# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verification functions for evaluation tasks.

Provides both programmatic verifiers (response content, tool calls, DB state)
and LLM-based judges for cases requiring semantic understanding.
"""

import logging
import re
from typing import Any

from .schema import VerificationResult

logger = logging.getLogger(__name__)


def verify_response_contains(
    response: str,
    values: list[str],
    case_sensitive: bool = False,
) -> VerificationResult:
    """
    Check if response contains all expected values.

    Args:
        response: The agent's response text
        values: List of strings that must appear in the response
        case_sensitive: Whether to do case-sensitive matching

    Returns:
        VerificationResult with found/missing details
    """
    if not case_sensitive:
        response_check = response.lower()
        check_values = [(v, v.lower()) for v in values]
    else:
        response_check = response
        check_values = [(v, v) for v in values]

    found = []
    missing = []

    for original, check in check_values:
        if check in response_check:
            found.append(original)
        else:
            missing.append(original)

    passed = len(missing) == 0

    return VerificationResult(
        name="response_contains",
        passed=passed,
        details={
            "found": found,
            "missing": missing,
            "total_expected": len(values),
        },
        error=f"Missing expected values: {missing}" if not passed else None,
    )


def verify_response_not_contains(
    response: str,
    values: list[str],
    case_sensitive: bool = False,
) -> VerificationResult:
    """
    Check that response does NOT contain any forbidden values.

    Args:
        response: The agent's response text
        values: List of strings that must NOT appear in the response
        case_sensitive: Whether to do case-sensitive matching

    Returns:
        VerificationResult with forbidden values found
    """
    if not case_sensitive:
        response_check = response.lower()
        check_values = [(v, v.lower()) for v in values]
    else:
        response_check = response
        check_values = [(v, v) for v in values]

    forbidden_found = []

    for original, check in check_values:
        if check in response_check:
            forbidden_found.append(original)

    passed = len(forbidden_found) == 0

    return VerificationResult(
        name="response_not_contains",
        passed=passed,
        details={
            "forbidden_found": forbidden_found,
            "total_checked": len(values),
        },
        error=f"Found forbidden values: {forbidden_found}" if not passed else None,
    )


def verify_response_matches_pattern(
    response: str,
    pattern: str,
    flags: int = re.IGNORECASE,
) -> VerificationResult:
    """
    Check if response matches a regex pattern.

    Args:
        response: The agent's response text
        pattern: Regex pattern to match
        flags: Regex flags (default: case-insensitive)

    Returns:
        VerificationResult with match details
    """
    try:
        match = re.search(pattern, response, flags)
        passed = match is not None

        return VerificationResult(
            name="response_matches_pattern",
            passed=passed,
            details={
                "pattern": pattern,
                "matched": match.group(0) if match else None,
            },
            error=f"Pattern '{pattern}' not found in response" if not passed else None,
        )
    except re.error as e:
        return VerificationResult(
            name="response_matches_pattern",
            passed=False,
            details={"pattern": pattern},
            error=f"Invalid regex pattern: {e}",
        )


def verify_tool_called(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
) -> VerificationResult:
    """
    Check if a specific tool was called.

    Args:
        tool_calls: List of tool call records from the agent trace
        tool_name: Name of the tool that must have been called

    Returns:
        VerificationResult indicating if tool was called
    """
    called_tools = [tc.get("name", "") for tc in tool_calls]
    found = tool_name in called_tools

    return VerificationResult(
        name=f"tool_called_{tool_name}",
        passed=found,
        details={
            "required_tool": tool_name,
            "all_tools_called": called_tools,
        },
        error=f"Required tool '{tool_name}' was not called" if not found else None,
    )


def verify_tool_not_called(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
) -> VerificationResult:
    """
    Check that a specific tool was NOT called.

    This is critical for distinguishing between:
    - "Check return status" (should NOT call update_return)
    - "Initiate return" (SHOULD call update_return)

    Args:
        tool_calls: List of tool call records from the agent trace
        tool_name: Name of the tool that must NOT have been called

    Returns:
        VerificationResult indicating if tool was correctly not called
    """
    called_tools = [tc.get("name", "") for tc in tool_calls]
    found = tool_name in called_tools

    return VerificationResult(
        name=f"tool_not_called_{tool_name}",
        passed=not found,
        details={
            "forbidden_tool": tool_name,
            "all_tools_called": called_tools,
            "was_called": found,
        },
        error=f"Forbidden tool '{tool_name}' was called" if found else None,
    )


def verify_tools_called(
    tool_calls: list[dict[str, Any]],
    required_tools: list[str],
) -> VerificationResult:
    """
    Check if all required tools were called.

    Args:
        tool_calls: List of tool call records from the agent trace
        required_tools: List of tool names that must have been called

    Returns:
        VerificationResult with found/missing tool details
    """
    called_tools = set(tc.get("name", "") for tc in tool_calls)

    found = []
    missing = []

    for tool in required_tools:
        if tool in called_tools:
            found.append(tool)
        else:
            missing.append(tool)

    passed = len(missing) == 0

    return VerificationResult(
        name="tools_called",
        passed=passed,
        details={
            "required": required_tools,
            "found": found,
            "missing": missing,
            "all_tools_called": list(called_tools),
        },
        error=f"Missing required tools: {missing}" if not passed else None,
    )


def verify_tools_not_called(
    tool_calls: list[dict[str, Any]],
    forbidden_tools: list[str],
) -> VerificationResult:
    """
    Check that none of the forbidden tools were called.

    Args:
        tool_calls: List of tool call records from the agent trace
        forbidden_tools: List of tool names that must NOT have been called

    Returns:
        VerificationResult with any forbidden tools that were called
    """
    called_tools = set(tc.get("name", "") for tc in tool_calls)

    forbidden_called = [tool for tool in forbidden_tools if tool in called_tools]

    passed = len(forbidden_called) == 0

    return VerificationResult(
        name="tools_not_called",
        passed=passed,
        details={
            "forbidden": forbidden_tools,
            "forbidden_called": forbidden_called,
            "all_tools_called": list(called_tools),
        },
        error=f"Forbidden tools were called: {forbidden_called}" if not passed else None,
    )


def verify_tool_call_args(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
    expected_args: dict[str, Any],
) -> VerificationResult:
    """
    Verify that a tool was called with specific arguments.

    Args:
        tool_calls: List of tool call records from the agent trace
        tool_name: Name of the tool to check
        expected_args: Dict of argument names to expected values

    Returns:
        VerificationResult with argument matching details
    """
    # Find the tool call
    matching_calls = [tc for tc in tool_calls if tc.get("name") == tool_name]

    if not matching_calls:
        return VerificationResult(
            name=f"tool_args_{tool_name}",
            passed=False,
            details={"tool_name": tool_name, "expected_args": expected_args},
            error=f"Tool '{tool_name}' was not called",
        )

    # Check the last call (in case it was called multiple times)
    actual_args = matching_calls[-1].get("input", {})

    mismatches = {}
    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}

    passed = len(mismatches) == 0

    return VerificationResult(
        name=f"tool_args_{tool_name}",
        passed=passed,
        details={
            "tool_name": tool_name,
            "expected_args": expected_args,
            "actual_args": actual_args,
            "mismatches": mismatches,
        },
        error=f"Tool argument mismatches: {mismatches}" if not passed else None,
    )


def verify_db_state(
    db,  # TestDatabase instance
    customer_id: int | str,
    order_id: int | str,
    expected_state: dict[str, Any],
) -> VerificationResult:
    """
    Verify database state after a task.

    Useful for verifying that write operations (like update_return)
    correctly modified the database.

    Args:
        db: TestDatabase instance
        customer_id: Customer ID to query
        order_id: Order ID to query
        expected_state: Dict of field names to expected values

    Returns:
        VerificationResult with actual vs expected state
    """
    order = db.get_order(customer_id, order_id)

    if order is None:
        return VerificationResult(
            name="db_state",
            passed=False,
            details={
                "customer_id": customer_id,
                "order_id": order_id,
                "expected": expected_state,
            },
            error=f"Order not found: customer_id={customer_id}, order_id={order_id}",
        )

    # Convert order to dict for comparison
    order_dict = order.model_dump()

    mismatches = {}
    for field, expected_value in expected_state.items():
        actual_value = order_dict.get(field)

        # Handle case-insensitive string comparison
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if expected_value.lower() != actual_value.lower():
                mismatches[field] = {"expected": expected_value, "actual": actual_value}
        elif actual_value != expected_value:
            mismatches[field] = {"expected": expected_value, "actual": actual_value}

    passed = len(mismatches) == 0

    return VerificationResult(
        name="db_state",
        passed=passed,
        details={
            "customer_id": customer_id,
            "order_id": order_id,
            "expected": expected_state,
            "actual": {k: order_dict.get(k) for k in expected_state.keys()},
            "mismatches": mismatches,
        },
        error=f"Database state mismatches: {mismatches}" if not passed else None,
    )


class Verifier:
    """
    Orchestrates running all verifications for a task.

    Collects results from individual verification functions and
    aggregates them into a final pass/fail determination.
    """

    def __init__(self, db=None):
        """
        Initialize the verifier.

        Args:
            db: Optional TestDatabase instance for DB state verification
        """
        self.db = db

    def verify_task(
        self,
        task,  # EvalTask
        response: str,
        tool_calls: list[dict[str, Any]],
    ) -> list[VerificationResult]:
        """
        Run all applicable verifications for a task.

        Args:
            task: The EvalTask being verified
            response: The agent's response text
            tool_calls: List of tool calls from the agent trace

        Returns:
            List of VerificationResults from all checks
        """
        results = []

        # Response content checks
        if task.response_must_contain:
            results.append(
                verify_response_contains(response, task.response_must_contain)
            )

        if task.response_must_not_contain:
            results.append(
                verify_response_not_contains(response, task.response_must_not_contain)
            )

        # Tool call checks
        if task.tool_must_be_called:
            results.append(
                verify_tools_called(tool_calls, task.tool_must_be_called)
            )

        if task.tool_must_not_be_called:
            results.append(
                verify_tools_not_called(tool_calls, task.tool_must_not_be_called)
            )

        # Database state checks
        if task.expected_db_state and self.db is not None:
            order_id = task.ground_truth.get("order_id")
            if order_id:
                results.append(
                    verify_db_state(
                        self.db,
                        task.user_id,
                        order_id,
                        task.expected_db_state,
                    )
                )

        return results
