# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation runner for executing tasks against the agent.

Orchestrates the full evaluation flow:
1. Reset database to baseline state
2. Invoke the agent with the task prompt
3. Run all verification checks
4. Collect and report results
"""

import asyncio
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


async def run_evaluation(
    tasks: list[EvalTask],
    config: RunConfig | None = None,
) -> RunSummary:
    """
    Convenience function to run an evaluation.

    Args:
        tasks: List of EvalTasks to execute
        config: Optional RunConfig

    Returns:
        RunSummary with results
    """
    runner = EvalRunner(config)
    return await runner.run(tasks)
