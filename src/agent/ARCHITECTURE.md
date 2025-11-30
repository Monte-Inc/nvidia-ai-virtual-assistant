# Agent Architecture

This document covers the LangGraph agent architecture for the AI Virtual Assistant.

---

## Overview

The agent is a multi-agent system built with **LangGraph** that handles:
- **Order Status** - Checking order details and delivery status
- **Return Processing** - Initiating and tracking product returns
- **Product Q&A** - Answering questions about products using documentation

---

## Agent Graph

```
                         ┌─────────────────────┐
                         │        START        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │fetch_purchase_history│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                    ┌───▶│  primary_assistant  │◀──────────────────┐
                    │    └──────────┬──────────┘                   │
                    │               │                              │
                    │    ┌──────────┼──────────┬──────────┐        │
                    │    │          │          │          │        │
                    │    ▼          ▼          ▼          ▼        │
                    │ ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐   │
                    │ │Order │  │Return│  │Product│ │other_talk│   │
                    │ │Status│  │Proc. │  │  QA  │  └────┬─────┘   │
                    │ └──┬───┘  └──┬───┘  └──┬───┘       │         │
                    │    │         │         │           ▼         │
                    │    ▼         ▼         ▼          END        │
                    │ ┌──────┐  ┌──────┐  ┌──────┐                 │
                    │ │Valid-│  │Valid-│  │ RAG  │                 │
                    │ │ation │  │ation │  │Search│                 │
                    │ └──┬───┘  └──┬───┘  └──┬───┘                 │
                    │    │         │         │                     │
                    │    ▼         ▼         ▼                     │
                    │ ┌──────┐  ┌──────┐    END                    │
                    │ │Tools │  │Tools │                           │
                    │ └──┬───┘  └──┬───┘                           │
                    │    │         │                               │
                    │    │         ▼                               │
                    │    │    ┌─────────┐                          │
                    │    │    │Sensitive│ (interrupt_before)       │
                    │    │    │ Tools   │                          │
                    │    │    └────┬────┘                          │
                    │    │         │                               │
                    │    ▼         ▼                               │
                    │   END       END                              │
                    │                                              │
                    └────────── (clarification needed) ────────────┘
```

**File:** `src/agent/main.py`

---

## State

```python
class State(TypedDict):
    messages: list[AnyMessage]      # Conversation history
    user_id: str                    # Customer ID
    user_purchase_history: Dict     # Cached order/return data
    current_product: str            # Product being discussed
    needs_clarification: bool       # Ask follow-up?
    clarification_type: str         # "no_product" | "multiple_products"
    reason: str                     # Clarification details
```

---

## Routing

The **primary_assistant** routes to sub-agents via tool calls:

| Tool Call | Destination |
|-----------|-------------|
| `ToOrderStatusAssistant` | Order status sub-agent |
| `ToReturnProcessing` | Return processing sub-agent |
| `ToProductQAAssistant` | Product Q&A (RAG) |
| `HandleOtherTalk` | Greetings/off-topic |

**Product Validation:** Before processing order/return queries, the agent extracts the product name and validates it exists in the user's purchase history. If 0 or 2+ matches, it asks for clarification.

---

## Tools

| Tool | Type | Description |
|------|------|-------------|
| `structured_rag` | Safe | Text-to-SQL query for order data |
| `get_purchase_history` | Safe | Fetch user's orders |
| `return_window_validation` | Safe | Check return eligibility |
| `update_return` | **Sensitive** | Process return (requires approval) |

The graph interrupts before `update_return` for human-in-the-loop approval:

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["return_processing_sensitive_tools"]
)
```

---

## RAG Retrieval

| Type | Endpoint | Backend | Use Case |
|------|----------|---------|----------|
| Structured | `structured-retriever:8081` | Vanna.AI + PostgreSQL | Order/return queries |
| Unstructured | `unstructured-retriever:8081` | Milvus | Product documentation |

---

## Prompts

**File:** `src/agent/prompt.yaml`

| Key | Purpose |
|-----|---------|
| `primary_assistant_template` | Intent classification & routing |
| `order_status_template` | Order query responses |
| `return_processing_template` | Return handling |
| `rag_template` | Product Q&A from context |
| `ask_clarification.*` | Product disambiguation |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/agent/main.py` | Graph definition |
| `src/agent/server.py` | FastAPI endpoints |
| `src/agent/tools.py` | Tool definitions |
| `src/agent/utils.py` | Helper functions |
| `src/agent/prompt.yaml` | System prompts |
