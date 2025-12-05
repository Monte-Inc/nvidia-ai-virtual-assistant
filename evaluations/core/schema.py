# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation task schema definitions.

Defines the core data structures for representing evaluation tasks,
their expected outcomes, and evaluation results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TaskCategory(str, Enum):
    """Categories of evaluation tasks based on user goals."""

    ORDER_STATUS = "order_status"
    RETURN_STATUS = "return_status"
    RETURN_INIT = "return_init"
    PRODUCT_QA = "product_qa"
    OUT_OF_SCOPE = "out_of_scope"


class TaskStatus(str, Enum):
    """Status of an evaluation task."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class EvalTask(BaseModel):
    """
    Definition of an evaluation task.

    Each task represents a specific user scenario to test, with
    ground truth data and verification criteria.
    """

    # Identity
    id: str
    name: str
    category: TaskCategory

    # Test setup
    user_id: str
    prompt: str
    turns: Literal["single", "multi"] = "single"
    followup_prompts: list[str] = Field(default_factory=list)

    # Ground truth (populated from DB at task generation time)
    ground_truth: dict[str, Any] = Field(default_factory=dict)

    # Programmatic verification criteria
    response_must_contain: list[str] = Field(default_factory=list)
    response_must_not_contain: list[str] = Field(default_factory=list)
    tool_must_be_called: list[str] = Field(default_factory=list)
    tool_must_not_be_called: list[str] = Field(default_factory=list)
    expected_db_state: dict[str, Any] = Field(default_factory=dict)

    # LLM Judge configuration (for Product QA)
    use_llm_judge: bool = False
    judge_context: str | None = None
    judge_criteria: str | None = None

    model_config = {"use_enum_values": False}


class VerificationResult(BaseModel):
    """Result of a single verification check."""

    name: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvalResult(BaseModel):
    """
    Result of running an evaluation task.

    Contains the task outcome, all verification results,
    the agent's response, and trace data for debugging.
    """

    task_id: str
    category: TaskCategory
    status: TaskStatus

    # Verification outcomes
    verifications: list[VerificationResult] = Field(default_factory=list)

    # Agent output
    response: str = ""
    trace: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latency_ms: float | None = None

    # Failure analysis
    failure_summary: str | None = None
    error_message: str | None = None

    model_config = {"use_enum_values": False}

    @property
    def passed(self) -> bool:
        return self.status == TaskStatus.PASSED

    @property
    def all_verifications_passed(self) -> bool:
        return all(v.passed for v in self.verifications)
