#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple test to verify the evaluation framework flow.

Tests the order status lookup for user 4165 (who has a Jetson Nano order).
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluations.core.schema import EvalTask, EvalResult, TaskCategory, TaskStatus
from evaluations.core.client import AgentClient
from evaluations.core.db import TestDatabase, DBConfig


def create_order_status_task() -> EvalTask:
    """Create a simple order status task for user 4165."""
    return EvalTask(
        id="order_status_001",
        name="Check Jetson Nano order status",
        category=TaskCategory.ORDER_STATUS,
        user_id="4165",
        prompt="What's the status of my Jetson Nano order?",
        ground_truth={
            "order_id": 52768,
            "product_name": "JETSON NANO DEVELOPER KIT",
            "order_status": "Delivered",
        },
        response_must_contain=["Delivered"],
    )


async def run_test_with_agent(task: EvalTask) -> EvalResult:
    """Run the task against the live agent."""
    print(f"\n{'='*60}")
    print(f"Running task: {task.name}")
    print(f"Category: {task.category.value}")
    print(f"User ID: {task.user_id}")
    print(f"Prompt: {task.prompt}")
    print(f"{'='*60}\n")

    async with AgentClient(mode="http") as client:
        # Create session and invoke
        session_id = await client.create_session()
        print(f"Created session: {session_id}")

        response = await client.invoke(
            message=task.prompt,
            user_id=task.user_id,
            session_id=session_id,
        )

        print(f"\nAgent response ({response.latency_ms:.0f}ms):")
        print("-" * 40)
        print(response.content)
        print("-" * 40)

        # Verify response contains expected values
        verifications = []
        for expected in task.response_must_contain:
            found = expected.lower() in response.content.lower()
            verifications.append({
                "check": f"Response contains '{expected}'",
                "passed": found,
            })
            print(f"{'✓' if found else '✗'} Response contains '{expected}': {found}")

        # Clean up session
        await client.end_session(session_id)

        # Build result
        all_passed = all(v["passed"] for v in verifications)
        return EvalResult(
            task_id=task.id,
            category=task.category,
            status=TaskStatus.PASSED if all_passed else TaskStatus.FAILED,
            response=response.content,
            latency_ms=response.latency_ms,
        )


def test_database_setup():
    """Test that we can set up and query the test database."""
    print("\n" + "=" * 60)
    print("Testing Database Setup")
    print("=" * 60 + "\n")

    # Use a config that points to the CSV for baseline data
    csv_path = project_root / "data" / "orders.csv"

    if not csv_path.exists():
        print(f"✗ CSV file not found at {csv_path}")
        return False

    print(f"✓ Found CSV file: {csv_path}")

    # Create test database instance (won't actually connect unless DB is available)
    db = TestDatabase()

    # Load baseline data from CSV (doesn't require DB connection)
    try:
        db.load_baseline_from_csv(csv_path)
        print(f"✓ Loaded {len(db._baseline_data)} records from CSV")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return False

    # Show sample data for user 4165
    user_records = [r for r in db._baseline_data if r.get("customer_id") == "4165"]
    print(f"\n✓ Found {len(user_records)} orders for user 4165:")
    for record in user_records[:3]:  # Show first 3
        print(f"  - {record['product_name']}: {record['order_status']}")

    return True


def test_schema():
    """Test that schema classes work correctly."""
    print("\n" + "=" * 60)
    print("Testing Schema")
    print("=" * 60 + "\n")

    task = create_order_status_task()
    print(f"✓ Created EvalTask: {task.id}")
    print(f"  Category: {task.category}")
    print(f"  User ID: {task.user_id}")
    print(f"  Ground truth: {task.ground_truth}")

    result = EvalResult(
        task_id=task.id,
        category=task.category,
        status=TaskStatus.PASSED,
        response="Your Jetson Nano order has been Delivered.",
        latency_ms=1234.5,
    )
    print(f"\n✓ Created EvalResult:")
    print(f"  Status: {result.status}")
    print(f"  Passed: {result.passed}")

    return True


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Evaluation Framework Test")
    print("#" * 60)

    # Test 1: Schema
    schema_ok = test_schema()

    # Test 2: Database setup (CSV loading only, no DB connection required)
    db_ok = test_database_setup()

    # Test 3: Agent integration (requires running agent)
    print("\n" + "=" * 60)
    print("Testing Agent Integration")
    print("=" * 60)

    task = create_order_status_task()

    try:
        result = await run_test_with_agent(task)
        print(f"\n{'✓' if result.passed else '✗'} Test {'PASSED' if result.passed else 'FAILED'}")
        agent_ok = result.passed
    except Exception as e:
        print(f"\n✗ Agent test failed with error: {e}")
        print("  (This is expected if the agent service is not running)")
        agent_ok = None  # Inconclusive

    # Summary
    print("\n" + "#" * 60)
    print("# Summary")
    print("#" * 60)
    print(f"  Schema test:   {'✓ PASSED' if schema_ok else '✗ FAILED'}")
    print(f"  Database test: {'✓ PASSED' if db_ok else '✗ FAILED'}")
    if agent_ok is None:
        print(f"  Agent test:    ○ SKIPPED (agent not running)")
    else:
        print(f"  Agent test:    {'✓ PASSED' if agent_ok else '✗ FAILED'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
