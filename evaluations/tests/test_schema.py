# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluation schema definitions."""

import pytest
from datetime import datetime

from evaluations.core.schema import (
    EvalTask,
    EvalResult,
    TaskCategory,
    TaskStatus,
    VerificationResult,
)


class TestTaskCategory:
    """Tests for TaskCategory enum."""

    def test_all_categories_defined(self):
        """Verify all expected task categories exist."""
        expected = {"order_status", "return_status", "return_init", "product_qa", "out_of_scope"}
        actual = {c.value for c in TaskCategory}
        assert actual == expected

    def test_category_string_values(self):
        """Categories should have lowercase string values."""
        assert TaskCategory.ORDER_STATUS.value == "order_status"
        assert TaskCategory.RETURN_INIT.value == "return_init"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected task statuses exist."""
        expected = {"pending", "running", "passed", "failed", "error"}
        actual = {s.value for s in TaskStatus}
        assert actual == expected


class TestEvalTask:
    """Tests for EvalTask model."""

    def test_minimal_task_creation(self):
        """Create a task with only required fields."""
        task = EvalTask(
            id="test_001",
            name="Test task",
            category=TaskCategory.ORDER_STATUS,
            user_id="1234",
            prompt="What is my order status?",
        )
        assert task.id == "test_001"
        assert task.category == TaskCategory.ORDER_STATUS
        assert task.turns == "single"  # default
        assert task.response_must_contain == []  # default empty list

    def test_full_task_creation(self):
        """Create a task with all fields populated."""
        task = EvalTask(
            id="order_status_001",
            name="Check Jetson Nano order status",
            category=TaskCategory.ORDER_STATUS,
            user_id="4165",
            prompt="What's the status of my Jetson Nano order?",
            turns="single",
            ground_truth={
                "order_id": 52768,
                "product_name": "JETSON NANO DEVELOPER KIT",
                "order_status": "Delivered",
            },
            response_must_contain=["Delivered"],
            response_must_not_contain=["error", "not found"],
            tool_must_be_called=["get_purchase_history"],
            tool_must_not_be_called=["update_return"],
            expected_db_state={},
            use_llm_judge=False,
        )
        assert task.ground_truth["order_status"] == "Delivered"
        assert "Delivered" in task.response_must_contain
        assert "update_return" in task.tool_must_not_be_called

    def test_multi_turn_task(self):
        """Create a multi-turn task with follow-up prompts."""
        task = EvalTask(
            id="multi_001",
            name="Multi-turn order inquiry",
            category=TaskCategory.ORDER_STATUS,
            user_id="4165",
            prompt="What orders do I have?",
            turns="multi",
            followup_prompts=["What about the Jetson Nano?", "Is it delivered?"],
        )
        assert task.turns == "multi"
        assert len(task.followup_prompts) == 2

    def test_llm_judge_task(self):
        """Create a task that uses LLM judge for verification."""
        task = EvalTask(
            id="product_qa_001",
            name="Shield TV warranty question",
            category=TaskCategory.PRODUCT_QA,
            user_id="4165",
            prompt="What's the warranty on Shield TV?",
            use_llm_judge=True,
            judge_context="Shield TV comes with a 1-year limited warranty...",
            judge_criteria="Response accurately states the warranty period",
        )
        assert task.use_llm_judge is True
        assert task.judge_context is not None


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_passed_verification(self):
        """Create a passing verification result."""
        result = VerificationResult(
            name="response_contains_delivered",
            passed=True,
            details={"found": ["Delivered"], "missing": []},
        )
        assert result.passed is True
        assert result.error is None

    def test_failed_verification(self):
        """Create a failing verification result."""
        result = VerificationResult(
            name="response_contains_status",
            passed=False,
            details={"found": [], "missing": ["Delivered"]},
            error="Expected 'Delivered' not found in response",
        )
        assert result.passed is False
        assert "Delivered" in result.error


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_passed_result(self):
        """Create a passing evaluation result."""
        result = EvalResult(
            task_id="order_status_001",
            category=TaskCategory.ORDER_STATUS,
            status=TaskStatus.PASSED,
            response="Your Jetson Nano order has been Delivered.",
            latency_ms=1234.5,
        )
        assert result.passed is True
        assert result.status == TaskStatus.PASSED

    def test_failed_result(self):
        """Create a failing evaluation result."""
        result = EvalResult(
            task_id="order_status_002",
            category=TaskCategory.ORDER_STATUS,
            status=TaskStatus.FAILED,
            response="I couldn't find that order.",
            verifications=[
                VerificationResult(
                    name="contains_status",
                    passed=False,
                    details={"missing": ["Delivered"]},
                )
            ],
            failure_summary="Response missing expected order status",
        )
        assert result.passed is False
        assert result.all_verifications_passed is False

    def test_error_result(self):
        """Create an error evaluation result."""
        result = EvalResult(
            task_id="order_status_003",
            category=TaskCategory.ORDER_STATUS,
            status=TaskStatus.ERROR,
            error_message="Connection timeout to agent",
        )
        assert result.passed is False
        assert result.status == TaskStatus.ERROR

    def test_result_with_tool_calls(self):
        """Create a result that includes tool call trace."""
        result = EvalResult(
            task_id="return_init_001",
            category=TaskCategory.RETURN_INIT,
            status=TaskStatus.PASSED,
            response="I've initiated the return for your RTX 4070.",
            tool_calls=[
                {"name": "get_purchase_history", "input": {"user_id": "4165"}},
                {"name": "update_return", "input": {"order_id": 12345}},
            ],
        )
        assert len(result.tool_calls) == 2
        assert result.tool_calls[1]["name"] == "update_return"

    def test_all_verifications_passed_property(self):
        """Test the all_verifications_passed computed property."""
        # All pass
        result = EvalResult(
            task_id="test_001",
            category=TaskCategory.ORDER_STATUS,
            status=TaskStatus.PASSED,
            verifications=[
                VerificationResult(name="check1", passed=True),
                VerificationResult(name="check2", passed=True),
            ],
        )
        assert result.all_verifications_passed is True

        # One fails
        result.verifications.append(VerificationResult(name="check3", passed=False))
        assert result.all_verifications_passed is False
