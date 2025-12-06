# Evaluation Core Module

This module provides the core infrastructure for evaluating the NVIDIA AI Virtual Assistant agent. It supports end-to-end testing of user tasks with programmatic verification and database state validation.

## Overview

The evaluation framework tests whether the agent **accomplishes user goals correctly**, not whether it takes specific internal paths. It supports:

- Single-turn and multi-turn conversations
- Programmatic verification (response content, tool calls, database state)
- Database isolation via reset between tasks
- Detailed reporting with failure analysis

## Module Structure

```
core/
├── schema.py      # Data models for tasks and results
├── client.py      # Agent invocation client
├── db.py          # Test database management
├── verifiers.py   # Verification functions
└── runner.py      # Task loading and evaluation orchestration
```

## Quick Start

```python
import asyncio
from evaluations.core import (
    EvalTask,
    TaskCategory,
    EvalRunner,
    RunConfig,
    run_evaluation,
)

# Define a task
task = EvalTask(
    id="order_status_001",
    name="Check Jetson Nano order status",
    category=TaskCategory.ORDER_STATUS,
    user_id="4165",
    prompt="What's the status of my Jetson Nano order?",
    ground_truth={
        "order_id": 52768,
        "order_status": "Delivered",
    },
    response_must_contain=["Delivered"],
    tool_must_not_be_called=["update_return"],
)

# Run evaluation
async def main():
    summary = await run_evaluation([task])
    print(summary.format_report())

asyncio.run(main())
```

## Components

### Schema (`schema.py`)

Defines the core data models using Pydantic.

#### TaskCategory

Categories of evaluation tasks based on user goals:

| Category | Description |
|----------|-------------|
| `ORDER_STATUS` | User checking order status |
| `RETURN_STATUS` | User checking existing return status |
| `RETURN_INIT` | User initiating a new return |
| `PRODUCT_QA` | User asking product questions |
| `OUT_OF_SCOPE` | Off-topic or unsupported requests |

#### TaskStatus

Status of an evaluation task:

| Status | Description |
|--------|-------------|
| `PENDING` | Task not yet started |
| `RUNNING` | Task currently executing |
| `PASSED` | All verifications passed |
| `FAILED` | One or more verifications failed |
| `ERROR` | Task failed due to exception |

#### EvalTask

Definition of an evaluation task:

```python
EvalTask(
    # Identity
    id="order_status_001",
    name="Check order status",
    category=TaskCategory.ORDER_STATUS,

    # Test setup
    user_id="4165",
    prompt="What's my order status?",
    turns="single",  # or "multi"
    followup_prompts=[],  # for multi-turn

    # Ground truth from database
    ground_truth={"order_id": 123, "order_status": "Delivered"},

    # Programmatic verification
    response_must_contain=["Delivered"],
    response_must_not_contain=["error"],
    tool_must_be_called=["get_purchase_history"],
    tool_must_not_be_called=["update_return"],
    expected_db_state={"return_status": "Requested"},
)
```

#### EvalResult

Result of running an evaluation task:

```python
result = EvalResult(
    task_id="order_status_001",
    category=TaskCategory.ORDER_STATUS,
    status=TaskStatus.PASSED,
    verifications=[...],  # List of VerificationResult
    response="Your order is Delivered.",
    tool_calls=[{"name": "get_purchase_history", "input": {...}}],
    trace={"events": [...]},
    latency_ms=1234.5,
    failure_summary=None,  # Set if failed
)

# Properties
result.passed  # True if status is PASSED
result.all_verifications_passed  # True if all checks passed
```

### Client (`client.py`)

Provides the `AgentClient` for invoking the agent via direct LangGraph invocation.

#### AgentClient

```python
from evaluations.core import AgentClient

async with AgentClient(
    mode="direct",
    auto_approve_interrupts=True,
) as client:
    # Create session
    session_id = await client.create_session()

    # Single invocation
    response = await client.invoke(
        message="What's my order status?",
        user_id="4165",
        session_id=session_id,
    )
    print(response.content)
    print(response.tool_calls)
    print(response.latency_ms)

    # Multi-turn conversation
    responses = await client.invoke_multi_turn(
        messages=["What orders do I have?", "What about the Jetson?"],
        user_id="4165",
    )

    # Cleanup
    await client.end_session(session_id)
```

#### Direct Mode

Evaluations use direct LangGraph invocation (not HTTP) to ensure:
- **Database isolation**: The test database environment variable is read by the agent
- **Full trace access**: Tool calls, inputs, and outputs are captured
- **No external dependencies**: No need for running Docker containers

#### Auto-Approve Interrupts

When `auto_approve_interrupts=True`, the client automatically approves interrupt prompts (e.g., return confirmation). This is essential for evaluating return initiation flows.

### Database (`db.py`)

Manages test database isolation and state verification.

#### TestDatabase

```python
from evaluations.core import TestDatabase
from pathlib import Path

db = TestDatabase()

# Load baseline data from CSV
db.load_baseline_from_csv(Path("data/orders.csv"))

# Full setup (create DB, schema, load data)
db.setup(csv_path=Path("data/orders.csv"))

# Reset to baseline between tests
db.reset_to_baseline()

# Query data
order = db.get_order(customer_id="4165", order_id=52768)
orders = db.get_orders_by_customer("4165")
orders = db.get_orders_by_product("4165", "Jetson")

# Verify state after write operations
result = db.verify_return_status("4165", 52768, "Requested")
# {"passed": True, "expected": "Requested", "actual": "Requested"}

# Get customer summary for task generation
summary = db.get_customer_summary("4165")
# {"total_orders": 10, "order_statuses": {...}, "return_statuses": {...}}

# Cleanup
db.drop_test_database()
```

#### OrderRecord

Pydantic model representing an order/return record:

```python
order = db.get_order("4165", 52768)
print(order.product_name)      # "JETSON NANO DEVELOPER KIT"
print(order.order_status)      # "Delivered"
print(order.return_status)     # None or "Requested", "Approved", etc.
print(order.return_reason)     # "Defective unit"
```

### Verifiers (`verifiers.py`)

Functions for verifying task outcomes.

#### Response Content Verifiers

```python
from evaluations.core import verify_response_contains, verify_response_not_contains

# Check response contains expected values
result = verify_response_contains(
    response="Your order is Delivered.",
    values=["Delivered", "order"],
    case_sensitive=False,  # default
)
# result.passed = True
# result.details = {"found": ["Delivered", "order"], "missing": []}

# Check response excludes forbidden values
result = verify_response_not_contains(
    response="Your order is Delivered.",
    values=["error", "failed"],
)
# result.passed = True
```

#### Tool Call Verifiers

```python
from evaluations.core import (
    verify_tool_called,
    verify_tool_not_called,
    verify_tools_called,
    verify_tools_not_called,
)

tool_calls = [
    {"name": "get_purchase_history", "input": {"user_id": "4165"}},
    {"name": "structured_rag", "input": {}},
]

# Single tool checks
result = verify_tool_called(tool_calls, "get_purchase_history")  # passed=True
result = verify_tool_not_called(tool_calls, "update_return")     # passed=True

# Multiple tool checks
result = verify_tools_called(tool_calls, ["get_purchase_history", "structured_rag"])
result = verify_tools_not_called(tool_calls, ["update_return", "delete_order"])
```

#### Database State Verifier

```python
from evaluations.core import verify_db_state

result = verify_db_state(
    db=test_database,
    customer_id="4165",
    order_id=52768,
    expected_state={"return_status": "Requested"},
)
# result.passed = True/False
# result.details = {"expected": {...}, "actual": {...}, "mismatches": {...}}
```

#### Verifier Class

Orchestrates all verifications for a task:

```python
from evaluations.core import Verifier

verifier = Verifier(db=test_database)

results = verifier.verify_task(
    task=eval_task,
    response="Your order is Delivered.",
    tool_calls=[...],
)

for result in results:
    print(f"{result.name}: {'PASS' if result.passed else result.error}")
```

### Runner (`runner.py`)

Orchestrates the full evaluation flow.

#### RunConfig

```python
from evaluations.core import RunConfig
from pathlib import Path

config = RunConfig(
    # Agent settings
    agent_mode="direct",
    auto_approve_interrupts=True,

    # Database settings
    use_test_db=True,
    reset_db_per_task=True,
    csv_path=Path("data/orders.csv"),

    # Execution settings
    timeout_seconds=120.0,

    # Output settings
    verbose=True,
)
```

#### EvalRunner

```python
from evaluations.core import EvalRunner, EvalTask

runner = EvalRunner(config)

# Run single task
result = await runner.run_task(task)

# Run multiple tasks
summary = await runner.run(tasks)
print(summary.format_report())
```

#### RunSummary

```python
summary = await runner.run(tasks)

# Metrics
print(summary.total_tasks)      # 40
print(summary.passed)           # 36
print(summary.failed)           # 3
print(summary.errors)           # 1
print(summary.pass_rate)        # 90.0

# By category
print(summary.by_category)
# {"order_status": {"total": 10, "passed": 10, "failed": 0, "errors": 0}, ...}

# Formatted report
print(summary.format_report())
```

#### Convenience Function

```python
from evaluations.core import run_evaluation

summary = await run_evaluation(tasks, config)
```

### Task Loading (`runner.py`)

Functions for loading tasks from JSON files.

#### JSON Task File Format

Task files should be JSON with a `tasks` array:

```json
{
  "description": "Tasks for evaluating order status lookup",
  "tasks": [
    {
      "id": "order_status_001",
      "name": "Check delivered order status",
      "category": "order_status",
      "user_id": "4165",
      "prompt": "What's the status of my Jetson Nano order?",
      "ground_truth": {
        "order_id": 52768,
        "order_status": "Delivered"
      },
      "response_must_contain": ["Delivered"],
      "tool_must_not_be_called": ["update_return"]
    }
  ],
  "metadata": {
    "total_tasks": 1
  }
}
```

#### Loading Tasks

```python
from evaluations.core import (
    load_tasks_from_file,
    load_tasks_from_directory,
    list_task_files,
)

# Load from a single JSON file
tasks = load_tasks_from_file("evaluations/tasks/order_status.json")

# Load all tasks from the tasks directory
tasks = load_tasks_from_directory()

# Load specific categories only
tasks = load_tasks_from_directory(categories=["order_status", "return_status"])

# Load from a custom directory
tasks = load_tasks_from_directory(tasks_dir="/path/to/tasks")

# List available task files
files = list_task_files()
# [Path("evaluations/tasks/order_status.json"), ...]
```

#### Running from Files

The `EvalRunner` can load and run tasks directly from files:

```python
from evaluations.core import EvalRunner, RunConfig

runner = EvalRunner(RunConfig(verbose=True))

# Run tasks from a specific file
summary = await runner.run_from_file("evaluations/tasks/order_status.json")

# Run all tasks from the tasks directory
summary = await runner.run_from_directory()

# Run specific categories
summary = await runner.run_from_directory(categories=["order_status"])
```

#### Convenience Function (Extended)

The `run_evaluation` function supports multiple calling patterns:

```python
from evaluations.core import run_evaluation

# Run with explicit task list
summary = await run_evaluation(tasks=my_tasks)

# Run from a JSON file
summary = await run_evaluation(file_path="evaluations/tasks/order_status.json")

# Run specific categories from tasks directory
summary = await run_evaluation(categories=["order_status", "return_status"])

# Run all tasks from tasks directory (default)
summary = await run_evaluation()
```

## Example: Complete Evaluation

```python
import asyncio
from evaluations.core import (
    EvalTask,
    TaskCategory,
    EvalRunner,
    RunConfig,
)

# Define tasks
tasks = [
    EvalTask(
        id="order_status_001",
        name="Check delivered order",
        category=TaskCategory.ORDER_STATUS,
        user_id="4165",
        prompt="What's the status of my Jetson Nano order?",
        ground_truth={"order_id": 52768, "order_status": "Delivered"},
        response_must_contain=["Delivered"],
    ),
    EvalTask(
        id="return_status_001",
        name="Check return status (should not initiate)",
        category=TaskCategory.RETURN_STATUS,
        user_id="4165",
        prompt="What's the return status for my RTX 4090?",
        ground_truth={"order_id": 4065, "return_status": "Requested"},
        response_must_contain=["Requested"],
        tool_must_not_be_called=["update_return"],
    ),
    EvalTask(
        id="return_init_001",
        name="Initiate new return",
        category=TaskCategory.RETURN_INIT,
        user_id="4165",
        prompt="I want to return my Jetson Nano",
        ground_truth={"order_id": 52768},
        tool_must_be_called=["update_return"],
        expected_db_state={"return_status": "Requested"},
    ),
]

async def main():
    config = RunConfig(verbose=True)
    runner = EvalRunner(config)
    summary = await runner.run(tasks)

    print(summary.format_report())

    # Check results
    if summary.pass_rate < 100:
        for result in summary.results:
            if not result.passed:
                print(f"FAILED: {result.task_id}")
                print(f"  {result.failure_summary}")

asyncio.run(main())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_BASE_URL` | `http://localhost:8081` | Agent API base URL |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `password` | Database password |
| `CUSTOMER_DATA_DB` | `customer_data` | Production database name |
| `TEST_CUSTOMER_DATA_DB` | `customer_data_test` | Test database name |
