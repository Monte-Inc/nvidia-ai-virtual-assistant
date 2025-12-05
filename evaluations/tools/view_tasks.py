#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task viewer tool for evaluation tasks.

Provides an intelligent way to view, filter, and analyze evaluation tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def load_tasks(file_path: Path) -> dict:
    """Load tasks from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_scenario_color(scenario: str) -> str:
    """Get color for a scenario based on keywords."""
    scenario_lower = scenario.lower()
    if any(word in scenario_lower for word in ["approved", "valid", "delivered", "success", "specific"]):
        return Colors.GREEN
    elif any(word in scenario_lower for word in ["rejected", "canceled", "cancelled", "outside", "off_topic"]):
        return Colors.RED
    elif any(word in scenario_lower for word in ["pending", "notes", "in_progress", "already"]):
        return Colors.YELLOW
    elif any(word in scenario_lower for word in ["not_found", "no_return", "not_yet"]):
        return Colors.CYAN
    return Colors.BLUE


def get_status_field(ground_truth: dict) -> str:
    """Extract the most relevant status field from ground truth."""
    for field in ["order_status", "return_status", "status", "expected_status"]:
        if field in ground_truth and ground_truth[field]:
            return str(ground_truth[field])
    if ground_truth.get("exists") is False:
        return "Not Found"
    return "N/A"


def format_task_summary(task: dict, index: int) -> str:
    """Format a single task as a summary line."""
    ground_truth = task.get("ground_truth", {})
    scenario = ground_truth.get("scenario", "unknown")
    color = get_scenario_color(scenario)

    status = get_status_field(ground_truth)
    user = task.get("user_id", "N/A")

    return (
        f"{Colors.BOLD}{index:2}.{Colors.END} "
        f"[{color}{scenario:18}{Colors.END}] "
        f"User {user:4} | "
        f"Status: {status:16} | "
        f"{task['name']}"
    )


def format_task_detail(task: dict) -> str:
    """Format a single task with full details."""
    ground_truth = task.get("ground_truth", {})
    scenario = ground_truth.get("scenario", "unknown")

    lines = [
        f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}",
        f"{Colors.BOLD}Task: {task['id']}{Colors.END}",
        f"{Colors.BOLD}Name: {task['name']}{Colors.END}",
        f"{'='*70}",
        "",
        f"{Colors.CYAN}Category:{Colors.END} {task.get('category', 'unknown')}",
        f"{Colors.CYAN}Scenario:{Colors.END} {scenario}",
        f"{Colors.CYAN}User ID:{Colors.END}  {task.get('user_id', 'N/A')}",
        "",
        f"{Colors.GREEN}Prompt:{Colors.END}",
        f"  \"{task['prompt']}\"",
        "",
        f"{Colors.YELLOW}Ground Truth:{Colors.END}",
    ]

    for key, value in ground_truth.items():
        if key != "scenario":  # Don't duplicate scenario
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"{Colors.BLUE}Verification:{Colors.END}")

    if task.get("response_must_contain"):
        lines.append(f"  Must contain: {task['response_must_contain']}")
    if task.get("response_must_not_contain"):
        lines.append(f"  Must NOT contain: {task['response_must_not_contain']}")
    if task.get("tool_must_not_be_called"):
        lines.append(f"  Tools must NOT be called: {task['tool_must_not_be_called']}")
    if task.get("notes"):
        lines.append(f"\n{Colors.YELLOW}Notes:{Colors.END} {task['notes']}")

    return "\n".join(lines)


def format_metadata(metadata: dict) -> str:
    """Format metadata summary."""
    lines = [
        f"\n{Colors.BOLD}{Colors.HEADER}Summary{Colors.END}",
        f"{'─'*40}",
    ]

    if "total_tasks" in metadata:
        lines.append(f"Total tasks: {metadata['total_tasks']}")
        lines.append("")

    # Handle scenarios - could be a list or a dict with counts
    if "scenarios_covered" in metadata:
        lines.append(f"{Colors.CYAN}Scenarios:{Colors.END}")
        scenarios = metadata["scenarios_covered"]
        if isinstance(scenarios, list):
            lines.append(f"  {', '.join(scenarios)}")
        else:
            for scenario, count in scenarios.items():
                lines.append(f"  {scenario}: {count}")
        lines.append("")

    # Handle statuses - generic key name
    if "statuses_covered" in metadata:
        lines.append(f"{Colors.CYAN}Statuses covered:{Colors.END}")
        lines.append(f"  {', '.join(metadata['statuses_covered'])}")
        lines.append("")

    if "users_covered" in metadata:
        lines.append(f"{Colors.CYAN}Users:{Colors.END}")
        lines.append(f"  {', '.join(metadata['users_covered'])}")

    return "\n".join(lines)


def view_tasks(
    file_path: Path,
    task_id: str | None = None,
    scenario: str | None = None,
    user_id: str | None = None,
    show_detail: bool = False,
    show_metadata: bool = True,
) -> None:
    """View tasks with optional filtering."""
    data = load_tasks(file_path)
    tasks = data["tasks"]
    metadata = data.get("metadata", {})

    # Filter tasks
    filtered = tasks
    if task_id:
        filtered = [t for t in filtered if t["id"] == task_id]
    if scenario:
        filtered = [t for t in filtered if t.get("ground_truth", {}).get("scenario") == scenario]
    if user_id:
        filtered = [t for t in filtered if t.get("user_id") == user_id]

    # Display header - get category from first task if not at top level
    category = data.get("category") or (tasks[0].get("category", "unknown") if tasks else "unknown")
    print(f"\n{Colors.BOLD}{Colors.HEADER}Evaluation Tasks: {category}{Colors.END}")
    print(f"{Colors.CYAN}{data.get('description', '')}{Colors.END}")
    print(f"{'─'*70}")

    if not filtered:
        print(f"{Colors.RED}No tasks found matching criteria{Colors.END}")
        return

    # Display tasks
    if show_detail or task_id:
        for task in filtered:
            print(format_task_detail(task))
    else:
        for i, task in enumerate(filtered, 1):
            print(format_task_summary(task, i))

    # Display metadata
    if show_metadata and not task_id:
        print(format_metadata(metadata))


def list_task_files(tasks_dir: Path) -> list[Path]:
    """List all task JSON files in the tasks directory."""
    return sorted(tasks_dir.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_tasks.py                     # List all tasks in summary view
  python view_tasks.py -d                  # Show detailed view of all tasks
  python view_tasks.py -t order_status_001 # Show specific task
  python view_tasks.py -s canceled_order   # Filter by scenario
  python view_tasks.py -u 4165             # Filter by user ID
  python view_tasks.py -f return_status    # Use different task file
  python view_tasks.py --list              # List available task files
        """
    )

    parser.add_argument(
        "-f", "--file",
        default="order_status",
        help="Task file name (without .json extension)"
    )
    parser.add_argument(
        "-t", "--task",
        help="Show specific task by ID"
    )
    parser.add_argument(
        "-s", "--scenario",
        help="Filter by scenario (e.g., specific_product, canceled_order)"
    )
    parser.add_argument(
        "-u", "--user",
        help="Filter by user ID"
    )
    parser.add_argument(
        "-d", "--detail",
        action="store_true",
        help="Show detailed view"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Hide metadata summary"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available task files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (for programmatic use)"
    )

    args = parser.parse_args()

    # Find tasks directory
    script_dir = Path(__file__).parent.parent
    tasks_dir = script_dir / "tasks"

    if args.list:
        print(f"\n{Colors.BOLD}Available task files:{Colors.END}")
        for f in list_task_files(tasks_dir):
            print(f"  {f.stem}")
        return

    file_path = tasks_dir / f"{args.file}.json"
    if not file_path.exists():
        print(f"{Colors.RED}Error: Task file not found: {file_path}{Colors.END}")
        print(f"\nAvailable files:")
        for f in list_task_files(tasks_dir):
            print(f"  {f.stem}")
        sys.exit(1)

    if args.json:
        data = load_tasks(file_path)
        tasks = data["tasks"]
        if args.task:
            tasks = [t for t in tasks if t["id"] == args.task]
        if args.scenario:
            tasks = [t for t in tasks if t.get("ground_truth", {}).get("scenario") == args.scenario]
        if args.user:
            tasks = [t for t in tasks if t.get("user_id") == args.user]
        print(json.dumps(tasks, indent=2))
    else:
        view_tasks(
            file_path,
            task_id=args.task,
            scenario=args.scenario,
            user_id=args.user,
            show_detail=args.detail,
            show_metadata=not args.no_metadata,
        )


if __name__ == "__main__":
    main()
