#!/usr/bin/env python3
"""Export LangSmith traces to JSON files."""

import os
import json
from datetime import datetime, timedelta
from langsmith import Client

# Load from environment or .env file
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "nvidia-ai-virtual-assistant")


def export_traces(
    project_name: str = None,
    output_dir: str = None,
    days_back: int = 7,
    limit: int = None,
):
    """
    Export LangSmith traces to JSON files.

    Args:
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        output_dir: Directory to save exports (defaults to ./exports)
        days_back: Number of days of history to export (default: 7)
        limit: Maximum number of traces to export (default: None = all)
    """
    if not LANGCHAIN_API_KEY:
        raise ValueError(
            "LANGCHAIN_API_KEY not set. Export it or add to .env file."
        )

    project_name = project_name or LANGCHAIN_PROJECT
    output_dir = output_dir or os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(output_dir, exist_ok=True)

    client = Client(api_key=LANGCHAIN_API_KEY)

    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    print(f"Exporting traces from project: {project_name}")
    print(f"Date range: {start_time.date()} to {end_time.date()}")

    # Fetch runs (traces)
    runs = list(
        client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            is_root=True,  # Only get root traces, not child spans
        )
    )

    print(f"Found {len(runs)} traces")

    if not runs:
        print("No traces to export.")
        return

    # Convert runs to serializable format
    traces = []
    for run in runs:
        trace = {
            "id": str(run.id),
            "name": run.name,
            "run_type": run.run_type,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "status": run.status,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
            "latency_ms": (
                (run.end_time - run.start_time).total_seconds() * 1000
                if run.end_time and run.start_time
                else None
            ),
            "total_tokens": run.total_tokens,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_cost": str(run.total_cost) if run.total_cost else None,
            "feedback_stats": run.feedback_stats,
            "tags": run.tags,
            "metadata": run.extra.get("metadata") if run.extra else None,
        }
        traces.append(trace)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"langsmith_export_{project_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(
            {
                "project": project_name,
                "exported_at": datetime.now().isoformat(),
                "date_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_traces": len(traces),
                "traces": traces,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Exported {len(traces)} traces to: {filepath}")
    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export LangSmith traces")
    parser.add_argument(
        "--project", "-p", help="Project name", default=LANGCHAIN_PROJECT
    )
    parser.add_argument(
        "--output", "-o", help="Output directory", default=None
    )
    parser.add_argument(
        "--days", "-d", type=int, help="Days of history to export", default=7
    )
    parser.add_argument(
        "--limit", "-l", type=int, help="Max traces to export", default=None
    )

    args = parser.parse_args()

    export_traces(
        project_name=args.project,
        output_dir=args.output,
        days_back=args.days,
        limit=args.limit,
    )
