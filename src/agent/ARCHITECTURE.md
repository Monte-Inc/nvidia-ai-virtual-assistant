# AI Virtual Assistant - Architecture Guide

This document provides a comprehensive overview of how the AI Virtual Assistant for Customer Service works. It covers the system architecture, request flow, components, and key design patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Request Lifecycle](#request-lifecycle)
4. [Core Components](#core-components)
   - [Server Layer](#1-server-layer)
   - [State Machine](#2-state-machine-langgraph)
   - [Routing Logic](#3-routing-logic)
   - [Tools](#4-tools)
   - [RAG Retrieval](#5-rag-retrieval)
   - [Prompts](#6-prompts)
5. [Key Workflows](#key-workflows)
   - [Order Status Workflow](#order-status-workflow)
   - [Return Processing Workflow](#return-processing-workflow)
   - [Product Q&A Workflow](#product-qa-workflow)
6. [Data Storage](#data-storage)
7. [Configuration](#configuration)
8. [Key Design Patterns](#key-design-patterns)

---

## Overview

This is an **NVIDIA NIM Agent Blueprint** for text-based customer service, built using:

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Agent orchestration and state machine |
| **LangChain** | LLM integration and retrieval components |
| **FastAPI** | REST API server |
| **NVIDIA NIMs** | LLM inference, embeddings, reranking |
| **PostgreSQL** | Customer data, conversation history, checkpointing |
| **Milvus** | Vector storage for unstructured documents |
| **Redis** | Session caching |

The agent handles three primary customer service tasks:
1. **Order Status** - Checking order details and delivery status
2. **Return Processing** - Initiating and tracking product returns
3. **Product Q&A** - Answering questions about products using documentation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client / UI                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API Gateway (:9000)                               │
│                    src/api_gateway/main.py                           │
└──────────────┬─────────────────────────────────┬────────────────────┘
               │                                 │
               ▼                                 ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│      Agent Server (:8081)    │   │    Analytics Server (:8082)      │
│      src/agent/server.py     │   │    src/analytics/main.py         │
└──────────────┬───────────────┘   └──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                           │
│                    src/agent/main.py                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │   Primary   │  │Order Status │  │   Return    │  │ Product QA │  │
│  │  Assistant  │─▶│  Assistant  │  │ Processing  │  │  Handler   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└──────────────┬─────────────┬─────────────┬──────────────────────────┘
               │             │             │
               ▼             ▼             ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│   Structured Retriever       │   │    Unstructured Retriever        │
│   (Vanna.AI + PostgreSQL)    │   │    (Milvus Vector Search)        │
│   src/retrievers/structured  │   │    src/retrievers/unstructured   │
└──────────────────────────────┘   └──────────────────────────────────┘
               │                             │
               ▼                             ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│        PostgreSQL            │   │           Milvus                 │
│   - customer_data            │   │   - Product manuals (PDF)        │
│   - conversation_history     │   │   - Documentation                │
│   - checkpoints              │   │   - FAQs                         │
└──────────────────────────────┘   └──────────────────────────────────┘
```

---

## Request Lifecycle

Here's what happens when a user sends a message:

```
1. POST /generate
   ├── Input: { session_id, user_id, messages[] }
   │
2. Session Validation
   ├── Check session exists in Redis/local cache
   ├── If invalid → Return fallback response
   │
3. Input Sanitization
   ├── Remove non-ASCII characters
   ├── Bleach HTML content
   ├── Normalize special characters
   │
4. Check for Interrupted State
   ├── Graph may be paused awaiting human approval
   ├── If user says "yes/y" → Resume execution
   ├── If user provides other input → Pass as feedback
   │
5. LangGraph Execution
   ├── START → fetch_purchase_history
   ├── → primary_assistant (intent classification)
   ├── → Route to specialized sub-agent
   ├── → Execute tools as needed
   ├── → Generate response
   │
6. Stream Response
   ├── Events tagged "should_stream" are sent to client
   ├── Uses Server-Sent Events (SSE) format
   │
7. Save Conversation
   ├── Store in SessionManager (Redis/local)
   ├── Include timestamps for analytics
   │
8. Return StreamingResponse
   └── Final message includes finish_reason: "[DONE]"
```

---

## Core Components

### 1. Server Layer

**File:** `src/agent/server.py`

The FastAPI server exposes these endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/generate` | POST | Main entry - processes queries, streams responses |
| `/create_session` | GET | Creates new session with UUID |
| `/end_session` | GET | Persists conversation to PostgreSQL, clears cache |
| `/delete_session` | DELETE | Removes all session data |
| `/feedback/response` | POST | Stores user feedback (-1, 0, +1) |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

**Key Data Models:**

```python
class Prompt(BaseModel):
    messages: List[Message]  # Conversation history
    user_id: str             # Customer identifier
    session_id: str          # Session UUID

class Message(BaseModel):
    role: str      # "user", "assistant", or "system"
    content: str   # Message text
```

**Error Handling:**
The server includes fallback responses for exceptions:
```python
FALLBACK_RESPONSES = [
    "Please try re-phrasing, I am likely having some trouble with that question.",
    "I will get better with time, please try with a different question.",
    ...
]
```

---

### 2. State Machine (LangGraph)

**File:** `src/agent/main.py`

The agent uses a LangGraph `StateGraph` with the following state:

```python
class State(TypedDict):
    messages: list[AnyMessage]      # Full conversation history
    user_id: str                    # Customer ID for database queries
    user_purchase_history: Dict     # Cached order/return data
    current_product: str            # Product being discussed
    needs_clarification: bool       # Whether to ask follow-up question
    clarification_type: str         # "no_product" or "multiple_products"
    reason: str                     # Details for clarification prompt
```

**Graph Nodes:**

| Node | Handler | Purpose |
|------|---------|---------|
| `fetch_purchase_history` | `user_info()` | Loads customer's orders at conversation start |
| `primary_assistant` | `Assistant()` | Main router - classifies intent and delegates |
| `enter_order_status` | `create_entry_node()` | Transition node to order assistant |
| `order_status` | `Assistant()` | Handles order-related queries |
| `order_validation` | `validate_product_info()` | Validates product exists in history |
| `order_status_safe_tools` | `ToolNode` | Executes order tools |
| `enter_return_processing` | `create_entry_node()` | Transition node to return assistant |
| `return_processing` | `Assistant()` | Handles return requests |
| `return_validation` | `validate_product_info()` | Validates product for returns |
| `return_processing_safe_tools` | `ToolNode` | Executes safe return tools |
| `return_processing_sensitive_tools` | `ToolNode` | Executes update_return (requires approval) |
| `enter_product_qa` | `handle_product_qa()` | Answers product questions via RAG |
| `ask_clarification` | `ask_clarification()` | Asks user to specify which product |
| `other_talk` | `handle_other_talk()` | Handles greetings and off-topic queries |

**Graph Visualization:**

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │ fetch_purchase_history │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   primary_assistant    │◄─────────────────┐
                    └───────────┬───────────┘                   │
                                │                               │
            ┌───────────────────┼───────────────────┬───────────┘
            │                   │                   │           │
            ▼                   ▼                   ▼           ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐  ┌────────────┐
    │enter_order_   │   │enter_return_  │   │enter_product_ │  │ other_talk │
    │status         │   │processing     │   │qa             │  └─────┬──────┘
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘        │
            │                   │                   │                ▼
            ▼                   ▼                   │               END
    ┌───────────────┐   ┌───────────────┐          │
    │ order_status  │   │return_        │          │
    │               │◄──│processing     │◄─┐       │
    └───────┬───────┘   └───────┬───────┘  │       │
            │                   │          │       ▼
    ┌───────┴───────┐   ┌───────┴───────┐  │      END
    │               │   │               │  │
    ▼               ▼   ▼               ▼  │
┌────────┐  ┌───────────┐ ┌──────────┐ ┌───┴────────┐
│order_  │  │order_     │ │return_   │ │return_     │
│status_ │  │validation │ │safe_     │ │sensitive_  │
│safe_   │  └─────┬─────┘ │tools     │ │tools       │
│tools   │        │       └──────────┘ └────────────┘
└────┬───┘        │              │            │
     │            ▼              │            │
     │    ┌───────────────┐      │            │
     │    │ask_clarification│◄───┘            │
     │    └───────┬───────┘                   │
     │            │                           │
     │            ▼                           ▼
     └──────────►END◄────────────────────────END
```

---

### 3. Routing Logic

**Primary Assistant Routing** (`route_primary_assistant` function):

The primary assistant uses tool calls to delegate to specialized sub-agents:

```python
def route_primary_assistant(state: State):
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == "ToProductQAAssistant":
            return "enter_product_qa"
        elif tool_calls[0]["name"] == "ToOrderStatusAssistant":
            return "enter_order_status"
        elif tool_calls[0]["name"] == "ToReturnProcessing":
            return "enter_return_processing"
        elif tool_calls[0]["name"] == "HandleOtherTalk":
            return "other_talk"
```

**Routing Tools** (defined as Pydantic models):

| Tool Class | Description |
|------------|-------------|
| `ToProductQAAssistant` | Routes product questions (specs, warranties, manuals) |
| `ToOrderStatusAssistant` | Routes order/purchase history queries |
| `ToReturnProcessing` | Routes return requests and status checks |
| `HandleOtherTalk` | Routes greetings and off-topic messages |
| `ProductValidation` | Triggers product disambiguation |

**Product Validation Flow** (`validate_product_info` function):

```
User asks about a product
         │
         ▼
Fetch user's purchase history
         │
         ▼
Extract product name from query (LLM call)
         │
         ▼
Match against purchase history
         │
    ┌────┴────┬────────────────┐
    │         │                │
    ▼         ▼                ▼
0 matches  1 match      2+ matches
    │         │                │
    ▼         ▼                ▼
Set needs_  Continue     Set needs_
clarification to sub-    clarification
="no_product" agent      ="multiple_products"
    │                          │
    └──────────┬───────────────┘
               ▼
        ask_clarification
               │
               ▼
              END
```

---

### 4. Tools

**File:** `src/agent/tools.py`

| Tool | Function | Description |
|------|----------|-------------|
| `structured_rag` | Query structured data | Sends NL query to structured retriever, returns SQL results |
| `get_purchase_history` | Fetch orders | Direct SQL query for user's order history (last 15 orders) |
| `get_recent_return_details` | Fetch returns | Wrapper around `get_purchase_history` |
| `return_window_validation` | Check eligibility | Validates if order is within return window (default: 15 days) |
| `update_return` | Process return | **SENSITIVE** - Updates return_status to "Requested" in DB |

**Tool Categories:**

```
Order Status Tools:
├── Safe: structured_rag
└── Validation: ProductValidation

Return Processing Tools:
├── Safe: get_recent_return_details, return_window_validation
├── Sensitive: update_return (requires human approval)
└── Validation: ProductValidation
```

**Human-in-the-Loop:**

The graph is compiled with an interrupt before sensitive operations:

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["return_processing_sensitive_tools"]
)
```

When triggered, the user sees:
> "Do you approve of the process the return? Type 'y' to continue; otherwise, explain your requested changes."

---

### 5. RAG Retrieval

The agent uses two retrieval systems:

#### Structured Data Retrieval

**Endpoint:** `http://structured-retriever:8081/search`

- Uses **Vanna.AI** for text-to-SQL conversion
- Queries PostgreSQL `customer_data` table
- Returns order details, status, dates, amounts

```python
@tool
def structured_rag(query: str, user_id: str) -> str:
    """Use this for answering personalized queries about orders, returns, refunds..."""
    entry_doc_search = {"query": query, "top_k": 4, "user_id": user_id}
    response = requests.post(structured_rag_search, json=entry_doc_search)
    return aggregated_content
```

#### Unstructured Data Retrieval

**Endpoint:** `http://unstructured-retriever:8081/search`

- Uses **Milvus** vector database
- Searches PDF manuals, product documentation, FAQs
- Returns semantically similar document chunks

```python
def canonical_rag(query: str, conv_history: list) -> str:
    """Use this for answering generic queries about products..."""
    entry_doc_search = {"query": query, "top_k": 4, "conv_history": conv_history}
    response = requests.post(canonical_rag_search, json=entry_doc_search)
    return aggregated_content
```

---

### 6. Prompts

**File:** `src/agent/prompt.yaml`

| Prompt | Purpose |
|--------|---------|
| `primary_assistant_template` | Main router - identifies intent and delegates to sub-agents |
| `order_status_template` | Answers order queries using tools and purchase history |
| `return_processing_template` | Handles returns with validation checks |
| `other_talk_template` | Responds to greetings, explains limitations |
| `rag_template` | Answers product questions from retrieved context |
| `ask_clarification.base_prompt` | Base prompt for follow-up questions |
| `ask_clarification.followup.no_product` | When product not in purchase history |
| `ask_clarification.followup.default` | When multiple products match |
| `get_product_name.base_prompt` | Extracts product name from query |
| `get_product_name.fallback_prompt` | Extracts from full conversation if not in query |

**Prompt Variables:**

| Variable | Injected Value |
|----------|----------------|
| `{user_id}` | Customer ID |
| `{current_product}` | Product being discussed |
| `{user_purchase_history}` | List of orders/returns |
| `{messages}` | Conversation history |
| `{context}` | Retrieved document chunks |

---

## Key Workflows

### Order Status Workflow

```
User: "What's the status of my GeForce order?"
                    │
                    ▼
        ┌───────────────────────┐
        │   primary_assistant   │
        │  → ToOrderStatusAssistant
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   enter_order_status  │
        │  (transition message) │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │     order_status      │
        │  → ProductValidation  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   order_validation    │
        │  Extract "GeForce"    │
        │  Match in history     │
        └───────────┬───────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
    1 match found      0 or 2+ matches
          │                   │
          ▼                   ▼
    ┌─────────────┐    ┌──────────────────┐
    │order_status │    │ ask_clarification │
    │→structured_ │    │ "Which product?" │
    │  rag tool   │    └────────┬─────────┘
    └──────┬──────┘             │
           │                    ▼
           ▼                   END
    ┌─────────────┐
    │ Response:   │
    │ "Your order │
    │ shipped..." │
    └──────┬──────┘
           │
           ▼
          END
```

### Return Processing Workflow

```
User: "I want to return my headset"
                    │
                    ▼
        ┌───────────────────────┐
        │   primary_assistant   │
        │  → ToReturnProcessing │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ enter_return_processing│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   return_processing   │
        │  → ProductValidation  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   return_validation   │
        │  Find "headset"       │
        └───────────┬───────────┘
                    │
                    ▼ (1 match found)
        ┌───────────────────────┐
        │   return_processing   │
        │→ return_window_       │
        │  validation           │
        └───────────┬───────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
    Within window       Outside window
    & Delivered              │
          │                   ▼
          ▼            ┌──────────────┐
    ┌─────────────┐    │ "Sorry, the  │
    │→ update_    │    │ return window│
    │  return     │    │ has expired" │
    │ (SENSITIVE) │    └──────────────┘
    └──────┬──────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ INTERRUPT: Human Approval    │
    │ "Do you approve the return?" │
    └──────────────┬───────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    User: "yes"         User: "no"
         │                   │
         ▼                   ▼
    ┌─────────────┐    ┌──────────────┐
    │ Execute     │    │ Cancel and   │
    │ update_     │    │ explain      │
    │ return      │    └──────────────┘
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ "Return     │
    │ initiated"  │
    └─────────────┘
```

### Product Q&A Workflow

```
User: "What's the warranty on NVIDIA Shield?"
                    │
                    ▼
        ┌───────────────────────┐
        │   primary_assistant   │
        │ → ToProductQAAssistant│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   enter_product_qa    │
        │   (handle_product_qa) │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   canonical_rag()     │
        │   Search Milvus for   │
        │   warranty info       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   LLM generates       │
        │   response from       │
        │   retrieved context   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ "The NVIDIA Shield    │
        │ comes with a 1-year   │
        │ limited warranty..."  │
        └───────────────────────┘
                    │
                    ▼
                   END
```

---

## Data Storage

### PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `customer_data` | Orders, returns, customer info |
| `conversation_history` | Persisted conversations |
| `checkpoints` | LangGraph state snapshots |
| `checkpoint_blobs` | Serialized checkpoint data |
| `checkpoint_writes` | Checkpoint write log |
| `analytics_*` | Summaries, sentiment, feedback |

### Redis Keys

| Key Pattern | Purpose |
|-------------|---------|
| `session:{session_id}` | Session metadata |
| `conversation:{session_id}` | Active conversation |
| `feedback:{session_id}` | Response feedback |

### Milvus Collections

| Collection | Content |
|------------|---------|
| Product manuals | PDF documentation |
| FAQs | Frequently asked questions |
| Policy documents | Return policies, warranties |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_LLM_MODELNAME` | `meta/llama-3.3-70b-instruct` | LLM model |
| `APP_LLM_MODELENGINE` | `nvidia-ai-endpoints` | Inference backend |
| `APP_CACHE_NAME` | `redis` | Session cache type |
| `APP_DATABASE_NAME` | `postgres` | Database type |
| `APP_CHECKPOINTER_NAME` | `postgres` | Checkpoint storage |
| `GRAPH_RECURSION_LIMIT` | `6` | Max graph iterations |
| `GRAPH_TIMEOUT_IN_SEC` | `20` | Step timeout |
| `RETURN_WINDOW_THRESHOLD_DAYS` | `15` | Return eligibility window |
| `STRUCTURED_RAG_URI` | `http://structured-retriever:8081` | Structured retriever URL |
| `CANONICAL_RAG_URL` | `http://unstructured-retriever:8081` | Unstructured retriever URL |

### LLM Settings

```python
default_llm_kwargs = {
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 1024
}
```

---

## Key Design Patterns

### 1. Multi-Agent Delegation
The primary assistant acts as a router, delegating to specialized sub-agents based on user intent. Users are unaware of the delegation - it happens silently via tool calls.

### 2. Human-in-the-Loop
Sensitive operations (like processing returns) require explicit user approval. The graph interrupts before executing `update_return` and waits for confirmation.

### 3. State Checkpointing
LangGraph state is persisted to PostgreSQL, enabling:
- Conversation resumption after interrupts
- Fault tolerance and recovery
- Session continuity across server restarts

### 4. Product Validation
Before answering order/return queries, the agent validates that the product exists in the user's purchase history. This prevents hallucination and ensures accurate responses.

### 5. Graceful Degradation
On errors, the system returns friendly fallback messages rather than exposing technical details. Multiple fallback responses add variety.

### 6. Streaming Responses
Long responses stream to the client using Server-Sent Events (SSE). Nodes tagged with `"should_stream"` send incremental updates.

### 7. Dual Retrieval Strategy
- **Structured retrieval** (Vanna.AI) for personalized order data
- **Unstructured retrieval** (Milvus) for product documentation

This separation ensures accurate personal data queries while supporting rich product knowledge.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/agent/main.py` | LangGraph state machine definition |
| `src/agent/server.py` | FastAPI server and endpoints |
| `src/agent/tools.py` | Tool definitions (structured_rag, update_return, etc.) |
| `src/agent/utils.py` | Utility functions (canonical_rag, get_product_name) |
| `src/agent/prompt.yaml` | All system prompts |
| `src/agent/cache/session_manager.py` | Redis/local session management |
| `src/agent/datastore/datastore.py` | PostgreSQL conversation storage |
| `src/retrievers/structured_data/chains.py` | Vanna.AI SQL retrieval |
| `src/retrievers/unstructured_data/chains.py` | Milvus vector retrieval |
| `src/common/configuration.py` | Configuration classes |
| `src/common/utils.py` | Shared utilities (get_llm, get_prompts) |
