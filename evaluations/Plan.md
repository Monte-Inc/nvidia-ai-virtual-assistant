# Agent Evaluation Framework Plan

## Overview

This document outlines an **end-to-end evaluation framework** for the NVIDIA AI Virtual Assistant agent. The goal is to create an automated test suite that:

1. Tests complete user tasks from start to finish (not internal mechanics)
2. Uses real database data as ground truth for test generation
3. Resets a test DB clone before each task for isolation
4. Uses programmatic verification for all checks
5. Supports both single-turn and multi-turn conversations

**Key Principle**: We evaluate whether the agent *accomplished the user's goal correctly*, not whether it took a specific internal path.

**Current Status**: Phase 1 (Core Infrastructure) is complete. Phase 2 (Task Generation) is partially complete with order_status tasks implemented.

---

## Agent Architecture Summary

Before designing evaluations, we must understand what we're testing.

### Agent Structure

The agent is a **hierarchical multi-agent system** built with LangGraph:

```
                              ┌─────────────────────┐
                              │  fetch_purchase_    │
                         ┌───▶│      history        │
                         │    └──────────┬──────────┘
                         │               │
                      START              ▼
                               ┌─────────────────────┐
                               │  Primary Assistant  │
                               │                     │
                               │  Tools:             │
                               │  - HandleOtherTalk  │
                               │  - ToProductQA      │
                               │  - ToOrderStatus    │
                               │  - ToReturnProc     │
                               └─────────┬───────────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   │                     │                     │
                   ▼                     ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │   Other Talk     │  │  Product QA      │  │  Order Status    │
        │   (greetings,    │  │  (RAG over       │  │  Assistant       │
        │   off-topic)     │  │   manuals/FAQs)  │  └────────┬─────────┘
        └────────┬─────────┘  └────────┬─────────┘           │
                 │                     │                     │
                 ▼                     ▼                     ▼
                END                   END            ┌──────────────────┐
                                                     │ Return Processing│
                                                     │ Assistant        │
                                                     └────────┬─────────┘
                                                              │
                                                              ▼
                                                             END
```

### Tools Reference

| Tool | Type | Purpose | Database Effect |
|------|------|---------|-----------------|
| `structured_rag` | Read | Query order/return data via RAG | None |
| `get_purchase_history` | Read | Fetch user's order history | None |
| `get_recent_return_details` | Read | Fetch user's return history | None |
| `return_window_validation` | Read | Check if order is within return window | None |
| `update_return` | **Write** | Set return_status to 'Requested' | **Writes to DB** |
| `ProductValidation` | Internal | Disambiguate which product user means | None |
| `canonical_rag` | Read | RAG over product manuals/FAQs | None |

---

## End-to-End Task Categories

We categorize by **what the user is trying to accomplish**, not internal routing.

**Important**: Categories are used for organization and reporting only. The verifier does NOT apply automatic checks based on category—each task must explicitly specify its verification criteria.

### Category 1: Order Status Lookup ✅ IMPLEMENTED

**User Goal**: Find out the status of an order.

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Specific product | "What's the status of my RTX 4070 SUPER order?" | Response contains correct `order_status` from DB |
| Order with notes | "Why is my tee shirt order delayed?" | Response includes relevant notes |
| Canceled order | "Why was my RTX 4060 Ti canceled?" | Response explains cancellation reason |
| Product not in history | "What about my RTX 5090?" | Response clarifies product not found |

**Verification** (must be explicitly set per task):
- `response_must_contain`: The order status value from ground truth
- `tool_must_not_be_called`: `["update_return"]` (optional, to ensure no side effects)

**Status**: 10 tasks implemented in `tasks/order_status.json`

---

### Category 2: Return Status Lookup 🔲 NOT YET IMPLEMENTED

**User Goal**: Check the status of an existing return (NOT initiate a new one).

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Pending return | "What's the return status for my RTX 4070 SUPER?" | Response contains "Pending Approval" |
| Approved return | "What's my Shield Remote return status?" | Response contains "Approved" |
| Rejected return | "What happened to my Computer Care Kit return?" | Response contains "Rejected" + reason |
| No return exists | "What's the return status for my mousepad?" | Response indicates no return initiated |

**Verification** (must be explicitly set per task):
- `response_must_contain`: The return status value
- `tool_must_not_be_called`: `["update_return"]` — **Critical** to distinguish from return initiation

---

### Category 3: Return Initiation 🔲 NOT YET IMPLEMENTED

**User Goal**: Start a return for a product.

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Valid return (delivered, in window) | "I want to return my Jetson Nano" | DB updated: `return_status = 'Requested'` |
| Already has return in progress | "I want to return my RTX 4090" | Response indicates return already exists, no DB change |
| Not yet delivered | "Return my tee shirt" | Response explains can't return yet, no DB change |
| Outside return window | "Return my [old order]" | Response explains window expired, no DB change |

**Verification** (must be explicitly set per task):
- `tool_must_be_called`: `["update_return"]` for valid returns
- `expected_db_state`: `{"return_status": "Requested"}` to verify DB was updated

**Note**: Agent uses `interrupt_before` for `update_return`. The `AgentClient` supports `auto_approve_interrupts=True` to handle this in eval mode.

---

### Category 4: Product QA 🔲 NOT YET IMPLEMENTED

**User Goal**: Get information about a product (specs, warranty, troubleshooting, policies).

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Warranty question | "What's the warranty on Shield TV?" | Response contains warranty info from docs |
| Specs question | "What are the specs of RTX 4090?" | Response contains accurate specs |
| Troubleshooting | "My Shield TV won't turn on" | Response contains troubleshooting steps |
| Policy question | "What's your return policy?" | Response contains policy info |

**Verification options**:
- Programmatic: `tool_must_be_called: ["canonical_rag"]`
- Programmatic: `response_must_contain` with key facts from source docs
- **LLM Judge** (planned): Given the source chunk, verify the response accurately communicates that information

---

### Category 5: Out-of-Scope Handling 🔲 NOT YET IMPLEMENTED

**User Goal**: (Various off-topic requests)

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Weather | "What's the weather?" | Polite redirect to NVIDIA support scope |
| Unrelated product | "Tell me about iPhones" | Explains scope limitation |
| General chat | "How are you?" | Friendly but redirects to assistance |

**Verification** (must be explicitly set per task):
- `response_must_contain`: Redirect/scope language keywords
- `tool_must_not_be_called`: Sub-agent tools (optional)

---

## Single-Turn vs Multi-Turn

### Single-Turn Tasks ✅ IMPLEMENTED

Most tasks can be evaluated in a single turn:
- User sends one message
- Agent responds
- We verify the response

**No LLM simulator needed** - just a static prompt.

### Multi-Turn Tasks ✅ SCHEMA READY, NO TASKS YET

The schema and runner support multi-turn via:
- `turns: "multi"`
- `followup_prompts: ["follow-up 1", "follow-up 2"]`

The `AgentClient.invoke_multi_turn()` method handles sequential invocation.

**For multi-turn**:
- Option A: Pre-scripted follow-up messages (deterministic) ← **Implemented**
- Option B: LLM-simulated user responses (more realistic but adds variance) ← **Not implemented**

---

## Test Data Strategy

### Using Real Database Data ✅ IMPLEMENTED

Tasks are generated from actual DB records, ensuring consistency. The `TestDatabase` class supports:
- Loading baseline data from CSV (`load_baseline_from_csv`)
- Querying customer orders (`get_orders_by_customer`, `get_order`)
- Getting customer summaries for task generation (`get_customer_summary`)

Example task generation pattern:
```python
def generate_order_status_task(user_id: str, order: dict) -> EvalTask:
    """Generate an order status task from real DB data."""
    return EvalTask(
        id=f"order_status_{user_id}_{order['order_id']}",
        name=f"Check {order['product_name']} order status",
        category=TaskCategory.ORDER_STATUS,
        user_id=user_id,
        prompt=f"What's the status of my {order['product_name']} order?",
        ground_truth={
            "order_status": order["order_status"],
            "order_id": order["order_id"],
        },
        response_must_contain=[order["order_status"]],
    )
```

### Database Reset Strategy ✅ IMPLEMENTED

The `EvalRunner` with `reset_db_per_task=True`:
1. Resets test DB to known baseline state before each task
2. Runs the evaluation task
3. Verifies final DB state (for write operations like `update_return`)

This allows:
- Any user's data to be used for testing
- Write operations to be tested without corrupting data
- Consistent, reproducible results

---

## Implementation Status

### What's Built

#### Core Infrastructure (`evaluations/core/`)

| Module | Status | Description |
|--------|--------|-------------|
| `schema.py` | ✅ Complete | Pydantic models: `EvalTask`, `EvalResult`, `TaskCategory`, `TaskStatus`, `VerificationResult` |
| `db.py` | ✅ Complete | `TestDatabase` class with CSV loading, reset, query, and verify methods |
| `client.py` | ✅ Complete | `AgentClient` with HTTP and direct modes, auto-approve interrupts |
| `verifiers.py` | ✅ Complete | All programmatic verifiers implemented |
| `runner.py` | ✅ Complete | `EvalRunner`, `RunConfig`, `RunSummary`, task loading from JSON |

#### Tasks

| Category | Status | File | Count |
|----------|--------|------|-------|
| Order Status | ✅ Complete | `tasks/order_status.json` | 10 tasks |
| Return Status | 🔲 Not started | - | - |
| Return Initiation | 🔲 Not started | - | - |
| Product QA | 🔲 Not started | - | - |
| Out-of-Scope | 🔲 Not started | - | - |

#### Tests (`evaluations/tests/`)

| File | Status | Tests |
|------|--------|-------|
| `test_schema.py` | ✅ Complete | 11 tests |
| `test_db.py` | ✅ Complete | 14 tests (2 integration) |
| `test_client.py` | ✅ Complete | 10 tests (3 integration) |
| `test_verifiers.py` | ✅ Complete | 23 tests |

#### Tools

| Tool | Status | Description |
|------|--------|-------------|
| `tools/view_tasks.py` | ✅ Complete | CLI for viewing/filtering tasks |
| `test_flow.py` | ✅ Complete | Integration test script |

---

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Task definition schema (`EvalTask` Pydantic model)
- [x] Agent client (invoke agent with user_id + message, HTTP and direct modes)
- [x] Database utilities (CSV loading, reset, query, verify)
- [x] Programmatic verifiers (response content, tool calls, DB state)
- [x] Task runner (`EvalRunner` with `run_task`, `run`, `run_from_file`, `run_from_directory`)
- [x] Task loading from JSON files
- [x] Report generation (`RunSummary.format_report()`)
- [x] Unit tests for all core modules

### Phase 2: Task Generation 🔶 IN PROGRESS
- [x] Order Status tasks (10 tasks covering various scenarios)
- [ ] Return Status tasks
- [ ] Return Initiation tasks
- [ ] Product QA tasks
- [ ] Out-of-Scope tasks
- Target: 30-40 tasks across categories

### Phase 3: LLM Judge 🔲 NOT STARTED
- [ ] Implement `llm_judge_rag_accuracy()` in verifiers
- [ ] Integrate LLM judge into `Verifier.verify_task()` when `use_llm_judge=True`
- [ ] Identify source docs indexed in RAG system
- [ ] Create Product QA tasks with `judge_context` populated
- [ ] Test on Product QA category

### Phase 4: Additional Verifiers 🔲 NOT STARTED
- [ ] Integrate regex pattern matching into Verifier class
- [ ] Integrate tool argument verification into Verifier class
- [ ] Implement LLM failure summarization

### Phase 5: Reporting Enhancements 🔲 NOT STARTED
- [ ] Failure mode clustering
- [ ] Latency percentiles (P95, P99)
- [ ] Export results to JSON/CSV

### Phase 6: Multi-Turn Tasks 🔲 NOT STARTED
- [ ] Create multi-turn task definitions
- [ ] Test disambiguation scenarios
- [ ] Test follow-up question handling
- [ ] (Future) LLM user simulator

### Phase 7: Failure Analytics 🔲 NOT STARTED
- [ ] LLM-powered failure summarization
- [ ] Failure pattern detection across runs
- [ ] Actionable insights for agent improvement

---

## Design Decisions

### Resolved

1. **Test data source**: Use real DB data from CSV, reset test DB before each task
2. **Single vs multi-turn**: Schema supports both; started with single-turn tasks
3. **Interrupt handling**: `AgentClient` supports `auto_approve_interrupts=True`
4. **Task-driven verification**: Each task explicitly specifies its verification criteria (no automatic category-based checks)
5. **Failure summary**: Generated from verification error messages (no LLM summarization)

### Design Evolution

During implementation, we made these changes from the original plan:

1. **Removed automatic category-based verification**: Originally planned for the verifier to automatically apply checks based on category (e.g., RETURN_STATUS always checks that `update_return` wasn't called). Changed to task-driven approach where each task explicitly specifies all verification criteria. This is more flexible and explicit.

2. **Deferred LLM judge**: Originally planned to implement LLM-based judging for Product QA accuracy. Deferred to focus on programmatic verification first. The schema has fields (`use_llm_judge`, `judge_context`, `judge_criteria`) but no implementation exists.

3. **Deferred LLM failure summarization**: Originally planned to use LLM to summarize failure modes. Currently using simple string concatenation of verification errors.

4. **Simplified reporting**: The report format is simpler than originally envisioned—focuses on pass/fail counts and failed task details rather than "common failure mode" clustering.

---

## Future Work

### LLM Judge for Product QA

For semantic accuracy verification, an LLM judge will:
- Compare agent response against source document chunks
- Score accuracy on a 0-1 scale
- Explain what was missing or incorrect

**Planned implementation**:

```python
def llm_judge_rag_accuracy(response: str, source_chunk: str, question: str) -> dict:
    """
    Use LLM to verify RAG response accuracy.

    Returns:
        - score: 1.0 (accurate), 0.0 (inaccurate/missing)
        - explanation: What was correct/incorrect
        - failure_summary: If failed, what went wrong
    """
    prompt = f"""You are evaluating whether a customer service response accurately
communicates information from a source document.

## User Question
{question}

## Source Document (contains the correct answer)
{source_chunk}

## Agent Response
{response}

## Evaluation
1. Does the response accurately convey the key information from the source?
2. Is there any incorrect or hallucinated information?
3. Did the response answer the user's question?

Provide:
- score: 1 if accurate and complete, 0 if inaccurate or missing key info
- explanation: Brief explanation of your assessment
- failure_summary: If score=0, summarize what was wrong (for debugging)

Return as JSON: {{"score": 0|1, "explanation": "...", "failure_summary": "..."}}
"""
    result = call_judge_llm(prompt)
    return result
```

**Prerequisites**:
- Identify which docs are indexed in the RAG system
- Create a mapping from product questions to source chunks
- Implement the judge LLM call in `verifiers.py`

The schema already has fields for this (`use_llm_judge`, `judge_context`, `judge_criteria`).

### LLM Failure Summarization

For debugging at scale, an LLM could:
- Analyze failed tasks and categorize failure modes
- Identify patterns across failures
- Suggest root causes

**Planned implementation**:

```python
def summarize_failure(result: EvalResult) -> str:
    """Use LLM to summarize failure mode for debugging."""
    if result.passed:
        return None

    prompt = f"""An evaluation task failed. Summarize what went wrong in 1-2 sentences.

Task: {result.task_id}
Category: {result.category}

Verification Results:
{json.dumps(result.verifications, indent=2)}

Agent Response:
{result.response}

What was the failure mode?"""

    return call_llm(prompt)
```

### LLM User Simulator

For realistic multi-turn testing:
- Simulate user responses to agent clarification questions
- Generate varied phrasings of the same intent
- Test edge cases and ambiguous inputs

This would replace pre-scripted `followup_prompts` with dynamic responses.

---

## Test Users

Key users identified for task generation:

| User ID | Orders | Notable Data |
|---------|--------|--------------|
| `4165` | 10 | Jetson Nano (Delivered), RTX 4090 (Return Requested), various statuses |
| `5603` | Multiple | Shield TV (Shipped), cancelled orders |
| `125` | Multiple | Pending orders |
| `6229` | Multiple | Delayed orders with notes |
| `7154` | Multiple | Out for Delivery status |

---

## Running Evaluations

### Quick Start

```python
import asyncio
from evaluations.core import run_evaluation

async def main():
    # Run all tasks from tasks directory
    summary = await run_evaluation()
    print(summary.format_report())

asyncio.run(main())
```

### Run Specific Categories

```python
summary = await run_evaluation(categories=["order_status"])
```

### Run from File

```python
summary = await run_evaluation(file_path="evaluations/tasks/order_status.json")
```

### Run Tests

```bash
# Unit tests only
pytest evaluations/tests/ -v -m "not integration"

# All tests (requires agent and DB)
pytest evaluations/tests/ -v
```

---

## Next Steps

1. **Create return_status tasks** - Query DB for users with existing returns, create task JSON
2. **Create return_init tasks** - Query DB for users with delivered orders that can be returned
3. **Create product_qa tasks** - Identify key product questions and expected answers
4. **Create out_of_scope tasks** - Hand-write common off-topic queries
5. **Run full evaluation** - Test against live agent, analyze results
