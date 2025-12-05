# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for verification functions."""

import pytest

from evaluations.core.verifiers import (
    verify_response_contains,
    verify_response_not_contains,
    verify_response_matches_pattern,
    verify_tool_called,
    verify_tool_not_called,
    verify_tools_called,
    verify_tools_not_called,
    verify_tool_call_args,
    Verifier,
)
from evaluations.core.schema import EvalTask, TaskCategory


class TestVerifyResponseContains:
    """Tests for verify_response_contains."""

    def test_single_value_found(self):
        """Single expected value is found."""
        result = verify_response_contains(
            response="Your order status is Delivered.",
            values=["Delivered"],
        )
        assert result.passed is True
        assert result.details["found"] == ["Delivered"]
        assert result.details["missing"] == []

    def test_single_value_missing(self):
        """Single expected value is missing."""
        result = verify_response_contains(
            response="Your order status is Processing.",
            values=["Delivered"],
        )
        assert result.passed is False
        assert result.details["missing"] == ["Delivered"]
        assert "Delivered" in result.error

    def test_multiple_values_all_found(self):
        """All expected values are found."""
        result = verify_response_contains(
            response="Order 12345 for Jetson Nano is Delivered.",
            values=["12345", "Jetson Nano", "Delivered"],
        )
        assert result.passed is True
        assert len(result.details["found"]) == 3

    def test_multiple_values_some_missing(self):
        """Some expected values are missing."""
        result = verify_response_contains(
            response="Order 12345 is Processing.",
            values=["12345", "Jetson Nano", "Delivered"],
        )
        assert result.passed is False
        assert "12345" in result.details["found"]
        assert "Jetson Nano" in result.details["missing"]
        assert "Delivered" in result.details["missing"]

    def test_case_insensitive_by_default(self):
        """Matching is case-insensitive by default."""
        result = verify_response_contains(
            response="Your order is DELIVERED.",
            values=["delivered"],
        )
        assert result.passed is True

    def test_case_sensitive_when_specified(self):
        """Case-sensitive matching when specified."""
        result = verify_response_contains(
            response="Your order is DELIVERED.",
            values=["Delivered"],
            case_sensitive=True,
        )
        assert result.passed is False

    def test_empty_values_list(self):
        """Empty values list passes."""
        result = verify_response_contains(
            response="Any response.",
            values=[],
        )
        assert result.passed is True


class TestVerifyResponseNotContains:
    """Tests for verify_response_not_contains."""

    def test_forbidden_value_absent(self):
        """Forbidden value is correctly absent."""
        result = verify_response_not_contains(
            response="Your order is Delivered.",
            values=["error", "failed"],
        )
        assert result.passed is True
        assert result.details["forbidden_found"] == []

    def test_forbidden_value_present(self):
        """Forbidden value is found."""
        result = verify_response_not_contains(
            response="An error occurred.",
            values=["error", "failed"],
        )
        assert result.passed is False
        assert "error" in result.details["forbidden_found"]

    def test_case_insensitive_by_default(self):
        """Matching is case-insensitive by default."""
        result = verify_response_not_contains(
            response="An ERROR occurred.",
            values=["error"],
        )
        assert result.passed is False


class TestVerifyResponseMatchesPattern:
    """Tests for verify_response_matches_pattern."""

    def test_pattern_matches(self):
        """Regex pattern matches."""
        result = verify_response_matches_pattern(
            response="Order #12345 is Delivered.",
            pattern=r"Order #\d+",
        )
        assert result.passed is True
        assert result.details["matched"] == "Order #12345"

    def test_pattern_no_match(self):
        """Regex pattern doesn't match."""
        result = verify_response_matches_pattern(
            response="Your order is Delivered.",
            pattern=r"Order #\d+",
        )
        assert result.passed is False
        assert result.details["matched"] is None

    def test_invalid_pattern(self):
        """Invalid regex pattern is handled."""
        result = verify_response_matches_pattern(
            response="Some response.",
            pattern=r"[invalid",
        )
        assert result.passed is False
        assert "Invalid regex" in result.error


class TestVerifyToolCalled:
    """Tests for verify_tool_called."""

    def test_tool_was_called(self):
        """Tool is found in tool calls."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
            {"name": "structured_rag", "input": {}},
        ]
        result = verify_tool_called(tool_calls, "get_purchase_history")
        assert result.passed is True

    def test_tool_was_not_called(self):
        """Tool is not found in tool calls."""
        tool_calls = [
            {"name": "structured_rag", "input": {}},
        ]
        result = verify_tool_called(tool_calls, "get_purchase_history")
        assert result.passed is False
        assert "get_purchase_history" in result.error

    def test_empty_tool_calls(self):
        """Empty tool calls list."""
        result = verify_tool_called([], "get_purchase_history")
        assert result.passed is False


class TestVerifyToolNotCalled:
    """Tests for verify_tool_not_called."""

    def test_tool_correctly_not_called(self):
        """Forbidden tool was not called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
        ]
        result = verify_tool_not_called(tool_calls, "update_return")
        assert result.passed is True
        assert result.details["was_called"] is False

    def test_tool_incorrectly_called(self):
        """Forbidden tool was called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
            {"name": "update_return", "input": {"order_id": 123}},
        ]
        result = verify_tool_not_called(tool_calls, "update_return")
        assert result.passed is False
        assert "update_return" in result.error


class TestVerifyToolsCalled:
    """Tests for verify_tools_called."""

    def test_all_required_tools_called(self):
        """All required tools were called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
            {"name": "structured_rag", "input": {}},
            {"name": "update_return", "input": {}},
        ]
        result = verify_tools_called(
            tool_calls,
            ["get_purchase_history", "update_return"],
        )
        assert result.passed is True
        assert len(result.details["found"]) == 2
        assert len(result.details["missing"]) == 0

    def test_some_required_tools_missing(self):
        """Some required tools were not called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
        ]
        result = verify_tools_called(
            tool_calls,
            ["get_purchase_history", "update_return"],
        )
        assert result.passed is False
        assert "update_return" in result.details["missing"]


class TestVerifyToolsNotCalled:
    """Tests for verify_tools_not_called."""

    def test_no_forbidden_tools_called(self):
        """None of the forbidden tools were called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
        ]
        result = verify_tools_not_called(
            tool_calls,
            ["update_return", "delete_order"],
        )
        assert result.passed is True

    def test_forbidden_tool_was_called(self):
        """A forbidden tool was called."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
            {"name": "update_return", "input": {}},
        ]
        result = verify_tools_not_called(
            tool_calls,
            ["update_return", "delete_order"],
        )
        assert result.passed is False
        assert "update_return" in result.details["forbidden_called"]


class TestVerifyToolCallArgs:
    """Tests for verify_tool_call_args."""

    def test_correct_args(self):
        """Tool was called with correct arguments."""
        tool_calls = [
            {"name": "update_return", "input": {"order_id": 12345, "reason": "defective"}},
        ]
        result = verify_tool_call_args(
            tool_calls,
            "update_return",
            {"order_id": 12345},
        )
        assert result.passed is True

    def test_wrong_args(self):
        """Tool was called with wrong arguments."""
        tool_calls = [
            {"name": "update_return", "input": {"order_id": 99999}},
        ]
        result = verify_tool_call_args(
            tool_calls,
            "update_return",
            {"order_id": 12345},
        )
        assert result.passed is False
        assert "order_id" in result.details["mismatches"]

    def test_tool_not_called(self):
        """Tool was not called at all."""
        tool_calls = [
            {"name": "get_purchase_history", "input": {}},
        ]
        result = verify_tool_call_args(
            tool_calls,
            "update_return",
            {"order_id": 12345},
        )
        assert result.passed is False
        assert "not called" in result.error


class TestVerifier:
    """Tests for the Verifier class."""

    def test_verify_task_response_contains(self):
        """Verifier checks response_must_contain."""
        task = EvalTask(
            id="test_001",
            name="Test",
            category=TaskCategory.ORDER_STATUS,
            user_id="4165",
            prompt="What's my order status?",
            response_must_contain=["Delivered"],
        )

        verifier = Verifier()
        results = verifier.verify_task(
            task=task,
            response="Your order is Delivered.",
            tool_calls=[],
        )

        assert len(results) == 1
        assert results[0].name == "response_contains"
        assert results[0].passed is True

    def test_verify_task_response_not_contains(self):
        """Verifier checks response_must_not_contain."""
        task = EvalTask(
            id="test_002",
            name="Test",
            category=TaskCategory.ORDER_STATUS,
            user_id="4165",
            prompt="What's my order status?",
            response_must_not_contain=["error"],
        )

        verifier = Verifier()
        results = verifier.verify_task(
            task=task,
            response="Your order is Delivered.",
            tool_calls=[],
        )

        assert len(results) == 1
        assert results[0].name == "response_not_contains"
        assert results[0].passed is True

    def test_verify_task_tool_checks(self):
        """Verifier checks tool_must_be_called and tool_must_not_be_called."""
        task = EvalTask(
            id="test_003",
            name="Test",
            category=TaskCategory.RETURN_STATUS,
            user_id="4165",
            prompt="What's my return status?",
            tool_must_be_called=["get_recent_return_details"],
            tool_must_not_be_called=["update_return"],
        )

        verifier = Verifier()
        results = verifier.verify_task(
            task=task,
            response="Your return is Pending Approval.",
            tool_calls=[
                {"name": "get_recent_return_details", "input": {}},
            ],
        )

        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_verify_task_multiple_checks(self):
        """Verifier runs all applicable checks."""
        task = EvalTask(
            id="test_004",
            name="Test",
            category=TaskCategory.ORDER_STATUS,
            user_id="4165",
            prompt="What's my order status?",
            response_must_contain=["Delivered"],
            response_must_not_contain=["error"],
            tool_must_be_called=["get_purchase_history"],
        )

        verifier = Verifier()
        results = verifier.verify_task(
            task=task,
            response="Your order is Delivered.",
            tool_calls=[
                {"name": "get_purchase_history", "input": {}},
            ],
        )

        assert len(results) == 3
        assert all(r.passed for r in results)
