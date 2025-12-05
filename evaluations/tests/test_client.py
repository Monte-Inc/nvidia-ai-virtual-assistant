# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent client."""

import pytest

from evaluations.core.client import AgentClient, AgentResponse


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_values(self):
        """AgentResponse has sensible defaults."""
        response = AgentResponse(
            content="Hello, how can I help?",
            session_id="abc-123",
        )
        assert response.content == "Hello, how can I help?"
        assert response.session_id == "abc-123"
        assert response.tool_calls == []
        assert response.trace == {}
        assert response.latency_ms == 0.0
        assert response.interrupted is False

    def test_full_response(self):
        """AgentResponse with all fields populated."""
        response = AgentResponse(
            content="Your order is delivered.",
            session_id="session-456",
            tool_calls=[{"name": "get_purchase_history", "input": {}}],
            trace={"events": [{"event": "on_tool_start"}]},
            latency_ms=1500.0,
            interrupted=True,
        )
        assert len(response.tool_calls) == 1
        assert response.latency_ms == 1500.0
        assert response.interrupted is True


class TestAgentClient:
    """Tests for AgentClient class."""

    def test_init_default_values(self):
        """AgentClient initializes with default values."""
        client = AgentClient()
        assert client.base_url == "http://localhost:8081"
        assert client.mode == "http"
        assert client.auto_approve_interrupts is True

    def test_init_custom_values(self):
        """AgentClient accepts custom configuration."""
        client = AgentClient(
            base_url="http://custom:9000",
            mode="direct",
            auto_approve_interrupts=False,
        )
        assert client.base_url == "http://custom:9000"
        assert client.mode == "direct"
        assert client.auto_approve_interrupts is False

    def test_parse_sse_response_simple(self):
        """Parse a simple SSE response."""
        client = AgentClient()
        raw = 'data: {"choices": [{"message": {"content": "Hello"}, "finish_reason": ""}]}\n'
        result = client._parse_sse_response(raw)
        assert result == "Hello"

    def test_parse_sse_response_multiple_chunks(self):
        """Parse SSE response with multiple data chunks."""
        client = AgentClient()
        raw = (
            'data: {"choices": [{"message": {"content": "Your order "}, "finish_reason": ""}]}\n'
            'data: {"choices": [{"message": {"content": "is delivered."}, "finish_reason": ""}]}\n'
        )
        result = client._parse_sse_response(raw)
        assert result == "Your order is delivered."

    def test_parse_sse_response_ignores_done(self):
        """Parse SSE response ignores [DONE] finish reason."""
        client = AgentClient()
        raw = (
            'data: {"choices": [{"message": {"content": "Hello"}, "finish_reason": ""}]}\n'
            'data: {"choices": [{"message": {"content": ""}, "finish_reason": "[DONE]"}]}\n'
        )
        result = client._parse_sse_response(raw)
        assert result == "Hello"

    def test_parse_sse_response_handles_invalid_json(self):
        """Parse SSE response gracefully handles invalid JSON."""
        client = AgentClient()
        raw = (
            'data: {"choices": [{"message": {"content": "Valid"}, "finish_reason": ""}]}\n'
            'data: invalid json here\n'
            'data: {"choices": [{"message": {"content": " chunk"}, "finish_reason": ""}]}\n'
        )
        result = client._parse_sse_response(raw)
        assert result == "Valid chunk"

    def test_parse_sse_response_empty(self):
        """Parse empty SSE response."""
        client = AgentClient()
        result = client._parse_sse_response("")
        assert result == ""


class TestAgentClientIntegration:
    """
    Integration tests that require a running agent service.

    These tests are skipped by default. To run them:
    1. Ensure the agent service is running
    2. Run with: pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_and_end_session(self):
        """Test session lifecycle."""
        async with AgentClient() as client:
            session_id = await client.create_session()
            assert session_id is not None
            assert len(session_id) > 0

            await client.end_session(session_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invoke_order_status(self):
        """Test invoking agent for order status."""
        async with AgentClient() as client:
            session_id = await client.create_session()

            response = await client.invoke(
                message="What's the status of my Jetson Nano order?",
                user_id="4165",
                session_id=session_id,
            )

            assert response.content is not None
            assert len(response.content) > 0
            assert response.latency_ms > 0
            assert "delivered" in response.content.lower()

            await client.end_session(session_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invoke_multi_turn(self):
        """Test multi-turn conversation."""
        async with AgentClient() as client:
            responses = await client.invoke_multi_turn(
                messages=[
                    "What orders do I have?",
                    "What's the status of the Jetson Nano?",
                ],
                user_id="4165",
            )

            assert len(responses) == 2
            assert all(r.content for r in responses)
