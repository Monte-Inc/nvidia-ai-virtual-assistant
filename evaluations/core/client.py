# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agent client for evaluation.

Provides a client to invoke the agent programmatically, supporting both
HTTP API calls and direct graph invocation for evaluation purposes.
"""

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Response from the agent."""

    content: str
    session_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    interrupted: bool = False


class AgentClient:
    """
    Client for invoking the NVIDIA AI Virtual Assistant agent.

    Supports two modes:
    - HTTP mode: Calls the agent via the FastAPI /generate endpoint
    - Direct mode: Invokes the LangGraph directly (for evaluation with full trace access)
    """

    def __init__(
        self,
        base_url: str | None = None,
        mode: str = "http",
        auto_approve_interrupts: bool = True,
    ):
        """
        Initialize the agent client.

        Args:
            base_url: Base URL for HTTP mode (e.g., "http://localhost:8081")
            mode: "http" for API calls, "direct" for direct graph invocation
            auto_approve_interrupts: If True, automatically approve interrupts (like update_return)
        """
        self.base_url = base_url or os.getenv("AGENT_BASE_URL", "http://localhost:8081")
        self.mode = mode
        self.auto_approve_interrupts = auto_approve_interrupts
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()

    async def create_session(self) -> str:
        """Create a new session and return the session ID."""
        if self.mode == "http":
            response = await self._http_client.get(f"{self.base_url}/create_session")
            response.raise_for_status()
            return response.json()["session_id"]
        else:
            return str(uuid4())

    async def end_session(self, session_id: str) -> None:
        """End a session."""
        if self.mode == "http":
            await self._http_client.get(
                f"{self.base_url}/end_session",
                params={"session_id": session_id},
            )

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and clean up resources."""
        if self.mode == "http":
            await self._http_client.delete(
                f"{self.base_url}/delete_session",
                params={"session_id": session_id},
            )

    async def invoke(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Invoke the agent with a message.

        Args:
            message: The user's message
            user_id: The user ID for personalization
            session_id: Optional session ID (creates new if not provided)

        Returns:
            AgentResponse with the agent's response and metadata
        """
        if self.mode == "http":
            return await self._invoke_http(message, user_id, session_id)
        else:
            return await self._invoke_direct(message, user_id, session_id)

    async def _invoke_http(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        """Invoke agent via HTTP API."""
        if session_id is None:
            session_id = await self.create_session()

        start_time = time.time()

        payload = {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "session_id": session_id,
        }

        response = await self._http_client.post(
            f"{self.base_url}/generate",
            json=payload,
        )
        response.raise_for_status()

        # Parse SSE response
        content = self._parse_sse_response(response.text)
        latency_ms = (time.time() - start_time) * 1000

        # Check if we hit an interrupt (return confirmation prompt)
        interrupted = "approve" in content.lower() and "return" in content.lower()

        if interrupted and self.auto_approve_interrupts:
            # Auto-approve the interrupt
            approve_payload = {
                "messages": [{"role": "user", "content": "yes"}],
                "user_id": user_id,
                "session_id": session_id,
            }
            approve_response = await self._http_client.post(
                f"{self.base_url}/generate",
                json=approve_payload,
            )
            approve_response.raise_for_status()
            additional_content = self._parse_sse_response(approve_response.text)
            content = f"{content}\n[AUTO-APPROVED]\n{additional_content}"

        return AgentResponse(
            content=content,
            session_id=session_id,
            latency_ms=latency_ms,
            interrupted=interrupted,
        )

    async def _invoke_direct(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Invoke agent directly via LangGraph.

        This mode provides full access to trace data and tool calls.
        """
        # Import here to avoid circular imports and allow HTTP-only usage
        from src.agent.main import graph

        if session_id is None:
            session_id = str(uuid4())

        start_time = time.time()
        tool_calls = []
        trace_events = []

        config = {
            "recursion_limit": int(os.getenv("GRAPH_RECURSION_LIMIT", "6")),
            "configurable": {"thread_id": session_id, "chat_history": []},
        }

        input_for_graph = {
            "messages": [("human", message)],
            "user_id": user_id,
        }

        content = ""
        last_content = ""

        async for event in graph.astream_events(
            input_for_graph, version="v2", config=config
        ):
            kind = event["event"]
            tags = event.get("tags", [])
            trace_events.append({"event": kind, "name": event.get("name", "")})

            # Capture tool calls
            if kind == "on_tool_start":
                tool_calls.append({
                    "name": event.get("name", ""),
                    "input": event.get("data", {}).get("input", {}),
                })

            # Capture streaming content
            if kind == "on_chat_model_stream" and "should_stream" in tags:
                chunk = event["data"]["chunk"].content
                content += chunk

            # Capture final content
            if kind == "on_chain_end" and event["data"].get("output", "") == "__end__":
                end_msgs = event["data"]["input"]["messages"]
                if end_msgs:
                    last_content = end_msgs[-1].content

        # Use streamed content or fall back to last message
        if not content and last_content:
            content = last_content

        latency_ms = (time.time() - start_time) * 1000

        # Check for interrupt
        snapshot = await graph.aget_state(config)
        interrupted = bool(snapshot.next)

        if interrupted and self.auto_approve_interrupts:
            # Auto-approve the interrupt
            async for event in graph.astream_events(None, version="v2", config=config):
                kind = event["event"]
                tags = event.get("tags", [])

                if kind == "on_tool_start":
                    tool_calls.append({
                        "name": event.get("name", ""),
                        "input": event.get("data", {}).get("input", {}),
                    })

                if kind == "on_chat_model_stream" and "should_stream" in tags:
                    chunk = event["data"]["chunk"].content
                    content += chunk

        return AgentResponse(
            content=content,
            session_id=session_id,
            tool_calls=tool_calls,
            trace={"events": trace_events},
            latency_ms=latency_ms,
            interrupted=interrupted,
        )

    def _parse_sse_response(self, raw_response: str) -> str:
        """Parse Server-Sent Events response and extract content."""
        content_parts = []

        for line in raw_response.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    for choice in choices:
                        message = choice.get("message", {})
                        chunk = message.get("content", "")
                        finish_reason = choice.get("finish_reason", "")
                        if chunk and finish_reason != "[DONE]":
                            content_parts.append(chunk)
                except json.JSONDecodeError:
                    continue

        return "".join(content_parts)

    async def invoke_multi_turn(
        self,
        messages: list[str],
        user_id: str,
        session_id: str | None = None,
    ) -> list[AgentResponse]:
        """
        Invoke the agent with multiple turns.

        Args:
            messages: List of user messages to send in sequence
            user_id: The user ID for personalization
            session_id: Optional session ID (creates new if not provided)

        Returns:
            List of AgentResponses, one for each turn
        """
        if session_id is None:
            session_id = await self.create_session()

        responses = []
        for message in messages:
            response = await self.invoke(message, user_id, session_id)
            responses.append(response)

        return responses
