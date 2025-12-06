#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation CLI for running agent evaluations.

IMPORTANT: Evaluations run directly via LangGraph (not HTTP) to ensure database
isolation. The agent uses a separate test database during evals.

Setup (run once):
    python -m evaluations.main --setup-db

Usage examples:
    # Run all evaluations
    python -m evaluations.main

    # Run a specific category
    python -m evaluations.main --category order_status

    # Run multiple categories
    python -m evaluations.main --category order_status --category return_status

    # Run a specific task by ID
    python -m evaluations.main --task order_status_001

    # Run multiple specific tasks
    python -m evaluations.main --task order_status_001 --task order_status_002

    # Limit number of tasks (useful for quick testing)
    python -m evaluations.main --limit 5

    # Combine options
    python -m evaluations.main --category order_status --limit 3

    # Run in quiet mode (less output)
    python -m evaluations.main --quiet

    # List available categories and tasks
    python -m evaluations.main --list

    # Skip database reset between tasks (faster but less isolated)
    python -m evaluations.main --no-db-reset
"""

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

# Suppress ALL warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Some libraries bypass filters - suppress them by replacing showwarning
def _no_warning(*args, **kwargs):
    pass
warnings.showwarning = _no_warning


# =============================================================================
# CRITICAL: Load environment variables BEFORE any agent imports
# =============================================================================
# The agent reads various env vars at import time. We must configure
# the environment before importing anything from src.agent.

# Find the repo root (where .env file lives)
REPO_ROOT = Path(__file__).parent.parent


def _load_env_file():
    """Load environment variables from .env file."""
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # Don't override existing
                        os.environ[key] = value


_load_env_file()

# Set required defaults for agent to work outside Docker
if "EXAMPLE_PATH" not in os.environ:
    os.environ["EXAMPLE_PATH"] = "./src/agent"

# Use in-memory checkpointer for evals (avoids needing postgres checkpointer connection)
# This is fine for evals since we don't need persistent conversation state
if "APP_CHECKPOINTER_NAME" not in os.environ:
    os.environ["APP_CHECKPOINTER_NAME"] = "inmemory"

# Point to localhost for database connections (instead of Docker's "postgres" hostname)
if "APP_DATABASE_URL" not in os.environ:
    os.environ["APP_DATABASE_URL"] = "localhost:5432"

# Database credentials for the agent's tools (matches docker-compose defaults)
if "POSTGRES_USER_READONLY" not in os.environ:
    os.environ["POSTGRES_USER_READONLY"] = os.environ.get("POSTGRES_USER", "postgres")
if "POSTGRES_PASSWORD_READONLY" not in os.environ:
    os.environ["POSTGRES_PASSWORD_READONLY"] = os.environ.get("POSTGRES_PASSWORD", "password")

# LLM model name for NVIDIA AI Endpoints
if "APP_LLM_MODELNAME" not in os.environ:
    os.environ["APP_LLM_MODELNAME"] = "meta/llama-3.3-70b-instruct"

# Increase recursion limit for complex agent graphs
if "GRAPH_RECURSION_LIMIT" not in os.environ:
    os.environ["GRAPH_RECURSION_LIMIT"] = "25"

# Database configuration
TEST_DB_NAME = "customer_data_test"
PRODUCTION_DB_NAME = os.environ.get("CUSTOMER_DATA_DB", "customer_data")

# Store original value to restore later
_original_db = os.environ.get("CUSTOMER_DATA_DB")


def _set_test_db():
    """Set the database to test DB."""
    os.environ["CUSTOMER_DATA_DB"] = TEST_DB_NAME


def _restore_production_db():
    """Restore the database to production (original value)."""
    if _original_db is not None:
        os.environ["CUSTOMER_DATA_DB"] = _original_db
    elif "CUSTOMER_DATA_DB" in os.environ:
        del os.environ["CUSTOMER_DATA_DB"]


# =============================================================================
# Now safe to import evaluation modules
# =============================================================================

from evaluations.core import (
    EvalRunner,
    RunConfig,
    load_tasks_from_directory,
    load_tasks_from_file,
    list_task_files,
)
from evaluations.core.db import TestDatabase


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agent evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluations.main --setup-db                # Setup test database (run once)
  python -m evaluations.main                           # Run all tasks
  python -m evaluations.main --category order_status   # Run one category
  python -m evaluations.main --task order_status_001   # Run specific task
  python -m evaluations.main --limit 5                 # Run first 5 tasks
  python -m evaluations.main --list                    # List available tasks
        """,
    )

    # Database setup
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Setup the test database (create, load baseline data). Run this once before evals.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the test database to baseline state, then exit.",
    )
    parser.add_argument(
        "--drop-db",
        action="store_true",
        help="Drop the test database, then exit.",
    )

    # Task selection
    parser.add_argument(
        "-c", "--category",
        action="append",
        dest="categories",
        metavar="CATEGORY",
        help="Run tasks from specific category (can be repeated). "
             "Categories: order_status, return_status, return_init, product_qa, out_of_scope",
    )
    parser.add_argument(
        "-t", "--task",
        action="append",
        dest="tasks",
        metavar="TASK_ID",
        help="Run specific task by ID (can be repeated)",
    )
    parser.add_argument(
        "-f", "--file",
        type=Path,
        metavar="PATH",
        help="Run tasks from a specific JSON file",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        metavar="N",
        help="Limit to first N tasks",
    )

    # Database configuration
    parser.add_argument(
        "--no-db-reset",
        action="store_true",
        help="Disable database reset between tasks (faster but less isolated)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        metavar="PATH",
        help="Path to orders.csv for baseline data",
    )

    # Output configuration
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - only show summary",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode - show detailed output (default)",
    )

    # Utility commands
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available categories and tasks, then exit",
    )

    return parser.parse_args()


def get_csv_path(args: argparse.Namespace) -> Path:
    """Get the path to the orders.csv file."""
    if args.csv_path:
        return args.csv_path
    return Path(__file__).parent.parent / "data" / "orders.csv"


def setup_test_database(args: argparse.Namespace) -> int:
    """Setup the test database."""
    csv_path = get_csv_path(args)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    print("=" * 60)
    print("SETTING UP TEST DATABASE")
    print("=" * 60)
    print(f"Test database: {TEST_DB_NAME}")
    print(f"Baseline data: {csv_path}")
    print()

    db = TestDatabase()

    try:
        print("Creating test database...")
        db.create_test_database()
        print(f"  ✓ Database '{TEST_DB_NAME}' created")

        print("Creating schema...")
        db.create_schema()
        print("  ✓ customer_data table created")

        print("Loading baseline data...")
        db.load_baseline_from_csv(csv_path)
        print(f"  ✓ Loaded {len(db._baseline_data)} records from CSV")

        print("Populating database...")
        db.reset_to_baseline()
        print("  ✓ Database populated with baseline data")

        print()
        print("=" * 60)
        print("TEST DATABASE READY")
        print("=" * 60)
        print()
        print("To run evaluations: python -m evaluations.main")
        print()

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


def reset_test_database(args: argparse.Namespace) -> int:
    """Reset the test database to baseline state."""
    csv_path = get_csv_path(args)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    print("Resetting test database to baseline state...")

    db = TestDatabase()

    try:
        db.load_baseline_from_csv(csv_path)
        db.reset_to_baseline()
        print(f"✓ Database reset with {len(db._baseline_data)} records")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def drop_test_database() -> int:
    """Drop the test database."""
    print(f"Dropping test database '{TEST_DB_NAME}'...")

    db = TestDatabase()

    try:
        db.drop_test_database()
        print(f"✓ Database '{TEST_DB_NAME}' dropped")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def list_available_tasks() -> None:
    """List all available task categories and tasks."""
    print("\n" + "=" * 60)
    print("AVAILABLE EVALUATION TASKS")
    print("=" * 60)

    task_files = list_task_files()

    if not task_files:
        print("\nNo task files found in evaluations/tasks/")
        return

    total_tasks = 0

    for file_path in task_files:
        category = file_path.stem
        tasks = load_tasks_from_file(file_path)
        total_tasks += len(tasks)

        print(f"\n{category} ({len(tasks)} tasks)")
        print("-" * 40)

        for task in tasks:
            scenario = task.ground_truth.get("scenario", "")
            scenario_str = f" [{scenario}]" if scenario else ""
            print(f"  {task.id}: {task.name}{scenario_str}")

    print(f"\n{'=' * 60}")
    print(f"Total: {total_tasks} tasks across {len(task_files)} categories")
    print("=" * 60 + "\n")


def check_test_db_exists() -> bool:
    """Check if the test database exists."""
    db = TestDatabase()
    try:
        conn = db._get_connection(use_test_db=True)
        conn.close()
        return True
    except Exception:
        return False


def verify_using_test_db() -> bool:
    """
    Verify that the agent will actually use the test database.

    This queries the test DB directly to confirm it's accessible
    and has the expected data.
    """
    db = TestDatabase()
    try:
        conn = db._get_connection(use_test_db=True)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer_data")
            count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


async def run_evaluations(args: argparse.Namespace) -> int:
    """Run evaluations based on command line arguments."""

    # Check if test database exists
    if not check_test_db_exists():
        print("Error: Test database does not exist.")
        print()
        print("Run this first to set up the test database:")
        print("  python -m evaluations.main --setup-db")
        print()
        return 1

    # Set test database environment variable
    # This MUST happen before the agent module is imported
    _set_test_db()

    print(f"Database: {TEST_DB_NAME} (test)")
    print(f"Mode: direct (LangGraph)")

    # Verify the test DB is accessible and has data
    if not verify_using_test_db():
        print("Error: Test database exists but has no data or is inaccessible.")
        print("Try running: python -m evaluations.main --reset-db")
        _restore_production_db()
        return 1

    print(f"  ✓ Test database verified")

    # Build configuration - always use direct mode for evals
    csv_path = get_csv_path(args)
    config = RunConfig(
        agent_base_url="",  # Not used in direct mode
        agent_mode="direct",  # Always direct for proper DB isolation
        auto_approve_interrupts=True,
        reset_db_per_task=not args.no_db_reset,
        csv_path=csv_path if csv_path.exists() else None,
        verbose=not args.quiet,
    )

    # Load tasks based on arguments
    if args.file:
        # Load from specific file
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            _restore_production_db()
            return 1
        tasks = load_tasks_from_file(args.file)
        print(f"Loaded {len(tasks)} tasks from {args.file}")

    elif args.categories:
        # Load specific categories
        tasks = load_tasks_from_directory(categories=args.categories)
        print(f"Loaded {len(tasks)} tasks from categories: {', '.join(args.categories)}")

    else:
        # Load all tasks
        tasks = load_tasks_from_directory()
        print(f"Loaded {len(tasks)} tasks from all categories")

    # Filter to specific task IDs if provided
    if args.tasks:
        task_ids = set(args.tasks)
        tasks = [t for t in tasks if t.id in task_ids]
        if not tasks:
            print(f"Error: No tasks found matching IDs: {args.tasks}")
            _restore_production_db()
            return 1
        print(f"Filtered to {len(tasks)} specific tasks: {', '.join(args.tasks)}")

    # Apply limit if specified
    if args.limit and args.limit < len(tasks):
        tasks = tasks[:args.limit]
        print(f"Limited to first {args.limit} tasks")

    if not tasks:
        print("No tasks to run!")
        _restore_production_db()
        return 1

    # Run evaluations
    print(f"\nRunning {len(tasks)} evaluation tasks...")
    print("=" * 60)

    try:
        runner = EvalRunner(config)
        summary = await runner.run(tasks)

        # Print summary if in quiet mode (verbose mode already prints it)
        if args.quiet:
            print(summary.format_report())

        return 0  # Always return 0 - failures are informative, not errors

    finally:
        # ALWAYS restore production DB setting when done
        _restore_production_db()
        print(f"\nDatabase restored to: {PRODUCTION_DB_NAME}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle database management commands (these don't need test DB set)
    if args.setup_db:
        return setup_test_database(args)

    if args.reset_db:
        return reset_test_database(args)

    if args.drop_db:
        return drop_test_database()

    # Handle list command
    if args.list:
        list_available_tasks()
        return 0

    # Run evaluations
    try:
        return asyncio.run(run_evaluations(args))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        _restore_production_db()
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        _restore_production_db()
        return 1


if __name__ == "__main__":
    sys.exit(main())
