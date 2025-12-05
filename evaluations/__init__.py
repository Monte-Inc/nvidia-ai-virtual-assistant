# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA AI Virtual Assistant Evaluation Framework.

This package provides tools for end-to-end evaluation of the agent,
including task definitions, agent client, and database management.
"""

from .core import (
    AgentClient,
    EvalResult,
    EvalTask,
    TaskCategory,
    TaskStatus,
    TestDatabase,
)

__all__ = [
    "EvalTask",
    "EvalResult",
    "TaskCategory",
    "TaskStatus",
    "AgentClient",
    "TestDatabase",
]
