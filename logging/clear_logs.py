#!/usr/bin/env python3
"""Clear LangSmith traces from a project."""

import os
from datetime import datetime, timedelta
from langsmith import Client

# Load from environment or .env file
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "nvidia-ai-virtual-assistant")


def clear_traces(
    project_name: str = None,
    confirm: bool = False,
):
    """
    Clear all LangSmith traces by deleting and recreating the project.

    Args:
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        confirm: If True, skip confirmation prompt
    """
    if not LANGCHAIN_API_KEY:
        raise ValueError(
            "LANGCHAIN_API_KEY not set. Export it or add to .env file."
        )

    project_name = project_name or LANGCHAIN_PROJECT
    client = Client(api_key=LANGCHAIN_API_KEY)

    # Count traces first
    try:
        runs = list(
            client.list_runs(
                project_name=project_name,
                is_root=True,
            )
        )
    except Exception:
        print(f"Project '{project_name}' not found or empty.")
        return

    if not runs:
        print(f"No traces found in project: {project_name}")
        return

    print(f"Found {len(runs)} traces in project: {project_name}")

    # Confirmation
    if not confirm:
        response = input(
            f"\nThis will delete the project and recreate it empty. "
            f"All {len(runs)} traces will be lost. Continue? [y/N]: "
        )
        if response.lower() != "y":
            print("Aborted.")
            return

    # Delete and recreate project
    print("Deleting project...")
    try:
        client.delete_project(project_name=project_name)
        print(f"Deleted project: {project_name}")
        
        # Recreate empty project
        client.create_project(project_name=project_name)
        print(f"Recreated empty project: {project_name}")
        print(f"\nCleared {len(runs)} traces successfully.")
    except Exception as e:
        print(f"Error: {e}")


def delete_project(project_name: str = None, confirm: bool = False):
    """
    Delete an entire LangSmith project permanently (without recreating).

    Args:
        project_name: LangSmith project name
        confirm: If True, skip confirmation prompt
    """
    if not LANGCHAIN_API_KEY:
        raise ValueError(
            "LANGCHAIN_API_KEY not set. Export it or add to .env file."
        )

    project_name = project_name or LANGCHAIN_PROJECT
    client = Client(api_key=LANGCHAIN_API_KEY)

    if not confirm:
        response = input(
            f"\nAre you sure you want to PERMANENTLY DELETE project '{project_name}'? "
            "This will remove ALL traces and the project itself. [y/N]: "
        )
        if response.lower() != "y":
            print("Aborted.")
            return

    try:
        client.delete_project(project_name=project_name)
        print(f"Permanently deleted project: {project_name}")
    except Exception as e:
        print(f"Failed to delete project: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clear LangSmith traces")
    parser.add_argument(
        "--project", "-p", help="Project name", default=LANGCHAIN_PROJECT
    )
    parser.add_argument(
        "--delete-project",
        action="store_true",
        help="Permanently delete the project (don't recreate)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    if args.delete_project:
        delete_project(project_name=args.project, confirm=args.yes)
    else:
        clear_traces(
            project_name=args.project,
            confirm=args.yes,
        )
