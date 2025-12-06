# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation runner for executing tasks against the agent.

Orchestrates the full evaluation flow:
1. Load tasks from JSON files
2. Reset database to baseline state
3. Invoke the agent with the task prompt
4. Run all verification checks
5. Collect and report results
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .client import AgentClient
from .db import TestDatabase
from .schema import EvalTask, EvalResult, TaskCategory, TaskStatus, VerificationResult
from .verifiers import Verifier

logger = logging.getLogger(__name__)

# Default tasks directory
DEFAULT_TASKS_DIR = Path(__file__).parent.parent / "tasks"


# =============================================================================
# Task Loading Functions
# =============================================================================


def _parse_category(category_str: str) -> TaskCategory:
    """Parse a category string to TaskCategory enum."""
    category_map = {
        "order_status": TaskCategory.ORDER_STATUS,
        "return_status": TaskCategory.RETURN_STATUS,
        "return_init": TaskCategory.RETURN_INIT,
        "product_qa": TaskCategory.PRODUCT_QA,
        "out_of_scope": TaskCategory.OUT_OF_SCOPE,
    }
    return category_map.get(category_str.lower(), TaskCategory.ORDER_STATUS)


def load_task_from_dict(task_dict: dict[str, Any]) -> EvalTask:
    """
    Convert a task dictionary (from JSON) to an EvalTask object.

    Args:
        task_dict: Dictionary containing task data

    Returns:
        EvalTask object
    """
    return EvalTask(
        id=task_dict["id"],
        name=task_dict["name"],
        category=_parse_category(task_dict.get("category", "order_status")),
        user_id=task_dict["user_id"],
        prompt=task_dict["prompt"],
        turns=task_dict.get("turns", "single"),
        followup_prompts=task_dict.get("followup_prompts", []),
        ground_truth=task_dict.get("ground_truth", {}),
        response_must_contain=task_dict.get("response_must_contain", []),
        response_must_not_contain=task_dict.get("response_must_not_contain", []),
        tool_must_be_called=task_dict.get("tool_must_be_called", []),
        tool_must_not_be_called=task_dict.get("tool_must_not_be_called", []),
        expected_db_state=task_dict.get("expected_db_state", {}),
        use_llm_judge=task_dict.get("use_llm_judge", False),
        judge_context=task_dict.get("judge_context"),
        judge_criteria=task_dict.get("judge_criteria"),
    )


def load_tasks_from_file(file_path: Path | str) -> list[EvalTask]:
    """
    Load tasks from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of EvalTask objects
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Task file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    tasks = []
    for task_dict in data.get("tasks", []):
        try:
            task = load_task_from_dict(task_dict)
            tasks.append(task)
        except Exception as e:
            logger.error(f"Error loading task {task_dict.get('id', 'unknown')}: {e}")
            raise

    logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
    return tasks


def load_tasks_from_directory(
    tasks_dir: Path | str | None = None,
    categories: list[str] | None = None,
) -> list[EvalTask]:
    """
    Load all tasks from JSON files in a directory.

    Args:
        tasks_dir: Directory containing task JSON files.
                   Defaults to the evaluations/tasks directory.
        categories: Optional list of category names to filter by (e.g., ["order_status", "return_status"]).
                   If None, loads all categories.

    Returns:
        List of EvalTask objects from all files
    """
    if tasks_dir is None:
        tasks_dir = DEFAULT_TASKS_DIR
    else:
        tasks_dir = Path(tasks_dir)

    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    all_tasks = []
    json_files = sorted(tasks_dir.glob("*.json"))

    if categories:
        # Filter to only files matching the category names
        categories_lower = [c.lower() for c in categories]
        json_files = [f for f in json_files if f.stem.lower() in categories_lower]

    for json_file in json_files:
        try:
            tasks = load_tasks_from_file(json_file)
            all_tasks.extend(tasks)
        except Exception as e:
            logger.error(f"Error loading tasks from {json_file}: {e}")
            raise

    logger.info(f"Loaded {len(all_tasks)} total tasks from {len(json_files)} files")
    return all_tasks


def list_task_files(tasks_dir: Path | str | None = None) -> list[Path]:
    """
    List all task JSON files in the tasks directory.

    Args:
        tasks_dir: Directory containing task JSON files.
                   Defaults to the evaluations/tasks directory.

    Returns:
        List of Path objects for each JSON file
    """
    if tasks_dir is None:
        tasks_dir = DEFAULT_TASKS_DIR
    else:
        tasks_dir = Path(tasks_dir)

    if not tasks_dir.exists():
        return []

    return sorted(tasks_dir.glob("*.json"))


# =============================================================================
# Configuration and Summary Classes
# =============================================================================


class RunConfig(BaseModel):
    """Configuration for an evaluation run."""

    # Agent settings
    agent_base_url: str = "http://localhost:8081"
    agent_mode: str = "http"  # "http" or "direct"
    auto_approve_interrupts: bool = True

    # Database settings
    use_test_db: bool = True
    reset_db_per_task: bool = True
    csv_path: Path | None = None

    # Execution settings
    max_concurrent_tasks: int = 1  # Sequential by default for DB isolation
    timeout_seconds: float = 120.0

    # Output settings
    verbose: bool = True
    save_traces: bool = True

    model_config = {"arbitrary_types_allowed": True}


class RunSummary(BaseModel):
    """Summary of an evaluation run."""

    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0

    by_category: dict[str, dict[str, int]] = Field(default_factory=dict)

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    results: list[EvalResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate as a percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.passed / self.total_tasks) * 100

    def add_result(self, result: EvalResult) -> None:
        """Add a result and update counters."""
        self.results.append(result)
        self.total_tasks += 1

        if result.status == TaskStatus.PASSED:
            self.passed += 1
        elif result.status == TaskStatus.FAILED:
            self.failed += 1
        else:
            self.errors += 1

        # Update category counts
        category = result.category.value if isinstance(result.category, TaskCategory) else str(result.category)
        if category not in self.by_category:
            self.by_category[category] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}

        self.by_category[category]["total"] += 1
        if result.status == TaskStatus.PASSED:
            self.by_category[category]["passed"] += 1
        elif result.status == TaskStatus.FAILED:
            self.by_category[category]["failed"] += 1
        else:
            self.by_category[category]["errors"] += 1

    def format_report(self) -> str:
        """Generate a formatted text report."""
        lines = [
            "",
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Tasks: {self.total_tasks}",
            f"Passed:      {self.passed} ({self.pass_rate:.1f}%)",
            f"Failed:      {self.failed}",
            f"Errors:      {self.errors}",
            f"Duration:    {self.duration_seconds:.1f}s",
            "",
            "By Category:",
            "-" * 40,
        ]

        for category, counts in sorted(self.by_category.items()):
            total = counts["total"]
            passed = counts["passed"]
            rate = (passed / total * 100) if total > 0 else 0
            status = "✓" if passed == total else "✗"
            lines.append(f"  {status} {category}: {passed}/{total} ({rate:.0f}%)")

        # List failures
        failures = [r for r in self.results if r.status == TaskStatus.FAILED]
        if failures:
            lines.extend([
                "",
                "Failed Tasks:",
                "-" * 40,
            ])
            for result in failures:
                lines.append(f"  ✗ {result.task_id}")
                if result.failure_summary:
                    lines.append(f"    → {result.failure_summary}")
                for v in result.verifications:
                    if not v.passed:
                        lines.append(f"    - {v.name}: {v.error}")

        # List errors
        errors = [r for r in self.results if r.status == TaskStatus.ERROR]
        if errors:
            lines.extend([
                "",
                "Errors:",
                "-" * 40,
            ])
            for result in errors:
                lines.append(f"  ! {result.task_id}: {result.error_message}")

        lines.append("")
        return "\n".join(lines)


class EvalRunner:
    """
    Runs evaluation tasks against the agent.

    Handles:
    - Database setup and reset
    - Agent invocation
    - Verification orchestration
    - Result collection and reporting
    """

    def __init__(self, config: RunConfig | None = None):
        """
        Initialize the evaluation runner.

        Args:
            config: Optional RunConfig. Uses defaults if not provided.
        """
        self.config = config or RunConfig()
        self.db: TestDatabase | None = None
        self.verifier: Verifier | None = None
        self._client: AgentClient | None = None

    async def setup(self) -> None:
        """Set up database and verifier for evaluation."""
        if self.config.use_test_db:
            self.db = TestDatabase()
            csv_path = self.config.csv_path or Path(__file__).parent.parent.parent / "data" / "orders.csv"

            if csv_path.exists():
                self.db.load_baseline_from_csv(csv_path)
                logger.info(f"Loaded baseline data from {csv_path}")
            else:
                logger.warning(f"CSV file not found: {csv_path}")

        self.verifier = Verifier(db=self.db)

    async def teardown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def _get_client(self) -> AgentClient:
        """Get or create the agent client."""
        if self._client is None:
            self._client = AgentClient(
                base_url=self.config.agent_base_url,
                mode=self.config.agent_mode,
                auto_approve_interrupts=self.config.auto_approve_interrupts,
            )
            await self._client.__aenter__()
        return self._client

    async def run_task(self, task: EvalTask) -> EvalResult:
        """
        Run a single evaluation task.

        Args:
            task: The EvalTask to execute

        Returns:
            EvalResult with pass/fail status and verification details
        """
        started_at = datetime.now()

        if self.config.verbose:
            print(f"\n{'─' * 50}")
            print(f"Task: {task.name}")
            print(f"Category: {task.category.value}")
            print(f"Prompt: {task.prompt}")
            print(f"{'─' * 50}")

        try:
            # Reset database if configured
            if self.config.reset_db_per_task and self.db and self.db._baseline_data:
                self.db.reset_to_baseline()

            # Get agent client
            client = await self._get_client()

            # Create session and invoke agent
            session_id = await client.create_session()

            if task.turns == "single":
                response = await client.invoke(
                    message=task.prompt,
                    user_id=task.user_id,
                    session_id=session_id,
                )
            else:
                # Multi-turn: send all prompts
                all_prompts = [task.prompt] + task.followup_prompts
                responses = await client.invoke_multi_turn(
                    messages=all_prompts,
                    user_id=task.user_id,
                    session_id=session_id,
                )
                # Use the last response for verification
                response = responses[-1]

            # Clean up session
            await client.end_session(session_id)

            if self.config.verbose:
                print(f"\nResponse ({response.latency_ms:.0f}ms):")
                print(response.content[:500] + "..." if len(response.content) > 500 else response.content)

                # Debug: Show tool calls
                if response.tool_calls:
                    print(f"\nTool Calls ({len(response.tool_calls)}):")
                    for tc in response.tool_calls:
                        print(f"  → {tc.get('name', 'unknown')}")
                else:
                    print("\nTool Calls: (none)")

            # Run verifications
            verifications = self.verifier.verify_task(
                task=task,
                response=response.content,
                tool_calls=response.tool_calls,
            )

            # Determine overall status
            all_passed = all(v.passed for v in verifications)
            status = TaskStatus.PASSED if all_passed else TaskStatus.FAILED

            # Generate failure summary if failed
            failure_summary = None
            if not all_passed:
                failed_checks = [v for v in verifications if not v.passed]
                failure_summary = "; ".join(v.error for v in failed_checks if v.error)

            if self.config.verbose:
                print("\nVerifications:")
                for v in verifications:
                    symbol = "✓" if v.passed else "✗"
                    print(f"  {symbol} {v.name}: {'PASS' if v.passed else v.error}")

            completed_at = datetime.now()

            return EvalResult(
                task_id=task.id,
                category=task.category,
                status=status,
                verifications=verifications,
                response=response.content,
                tool_calls=response.tool_calls,
                trace=response.trace,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=response.latency_ms,
                failure_summary=failure_summary,
            )

        except Exception as e:
            logger.exception(f"Error running task {task.id}")

            return EvalResult(
                task_id=task.id,
                category=task.category,
                status=TaskStatus.ERROR,
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(e),
            )

    async def run_tasks(self, tasks: list[EvalTask]) -> RunSummary:
        """
        Run multiple evaluation tasks.

        Args:
            tasks: List of EvalTasks to execute

        Returns:
            RunSummary with aggregated results
        """
        summary = RunSummary()
        summary.started_at = datetime.now()

        await self.setup()

        try:
            for task in tasks:
                result = await self.run_task(task)
                summary.add_result(result)

                if self.config.verbose:
                    status_symbol = "✓" if result.passed else "✗"
                    print(f"\n{status_symbol} Task {result.task_id}: {'PASSED' if result.passed else 'FAILED'}")

        finally:
            await self.teardown()

        summary.completed_at = datetime.now()
        summary.duration_seconds = (summary.completed_at - summary.started_at).total_seconds()

        return summary

    async def run(self, tasks: list[EvalTask]) -> RunSummary:
        """
        Main entry point for running evaluations.

        Args:
            tasks: List of EvalTasks to execute

        Returns:
            RunSummary with full results and report
        """
        summary = await self.run_tasks(tasks)

        if self.config.verbose:
            print(summary.format_report())

        return summary

    async def run_from_file(self, file_path: Path | str) -> RunSummary:
        """
        Load tasks from a JSON file and run them.

        Args:
            file_path: Path to the task JSON file

        Returns:
            RunSummary with results
        """
        tasks = load_tasks_from_file(file_path)
        return await self.run(tasks)

    async def run_from_directory(
        self,
        tasks_dir: Path | str | None = None,
        categories: list[str] | None = None,
    ) -> RunSummary:
        """
        Load tasks from a directory and run them.

        Args:
            tasks_dir: Directory containing task JSON files.
                       Defaults to the evaluations/tasks directory.
            categories: Optional list of category names to filter by.

        Returns:
            RunSummary with results
        """
        tasks = load_tasks_from_directory(tasks_dir, categories)
        return await self.run(tasks)


async def run_evaluation(
    tasks: list[EvalTask] | None = None,
    config: RunConfig | None = None,
    file_path: Path | str | None = None,
    categories: list[str] | None = None,
) -> RunSummary:
    """
    Convenience function to run an evaluation.

    Can be called with:
    - tasks: Direct list of EvalTask objects
    - file_path: Path to a JSON task file
    - categories: List of category names to load from tasks directory

    Args:
        tasks: List of EvalTasks to execute (takes priority if provided)
        config: Optional RunConfig
        file_path: Path to a JSON task file
        categories: List of category names to load (e.g., ["order_status"])

    Returns:
        RunSummary with results
    """
    runner = EvalRunner(config)

    if tasks is not None:
        return await runner.run(tasks)
    elif file_path is not None:
        return await runner.run_from_file(file_path)
    elif categories is not None:
        return await runner.run_from_directory(categories=categories)
    else:
        # Default: run all tasks from the tasks directory
        return await runner.run_from_directory()
