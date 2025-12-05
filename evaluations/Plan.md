# Agent Evaluation Framework Plan

## Overview

This document outlines an **end-to-end evaluation framework** for the NVIDIA AI Virtual Assistant agent. The goal is to create an automated test suite that:

1. Tests complete user tasks from start to finish (not internal mechanics)
2. Uses real database data as ground truth for test generation
3. Resets a test DB clone before each task for isolation
4. Uses programmatic verification where possible, LLM judge where needed
5. Supports both single-turn and multi-turn conversations
6. Integrates with LangSmith for tracing and analysis

**Key Principle**: We evaluate whether the agent *accomplished the user's goal correctly*, not whether it took a specific internal path.

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

### Category 1: Order Status Lookup

**User Goal**: Find out the status of an order.

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Specific product | "What's the status of my RTX 4070 SUPER order?" | Response contains correct `order_status` from DB |
| Order with notes | "Why is my tee shirt order delayed?" | Response includes relevant notes |
| Canceled order | "Why was my RTX 4060 Ti canceled?" | Response explains cancellation reason |
| Product not in history | "What about my RTX 5090?" | Response clarifies product not found |

**Verification**:
- Programmatic: Response contains exact `order_status` value from DB
- Ground truth: Query DB for user's order before test

---

### Category 2: Return Status Lookup

**User Goal**: Check the status of an existing return (NOT initiate a new one).

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Pending return | "What's the return status for my RTX 4070 SUPER?" | Response contains "Pending Approval" |
| Approved return | "What's my Shield Remote return status?" | Response contains "Approved" |
| Rejected return | "What happened to my Computer Care Kit return?" | Response contains "Rejected" + reason |
| No return exists | "What's the return status for my mousepad?" | Response indicates no return initiated |

**Verification**:
- Programmatic: Response contains correct `return_status` from DB
- **Critical check**: `update_return` tool must NOT be called (this distinguishes status lookup from initiation)

---

### Category 3: Return Initiation

**User Goal**: Start a return for a product.

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Valid return (delivered, in window) | "I want to return my Jetson Nano" | DB updated: `return_status = 'Requested'` |
| Already has return in progress | "I want to return my RTX 4090" | Response indicates return already exists, no DB change |
| Not yet delivered | "Return my tee shirt" | Response explains can't return yet, no DB change |
| Outside return window | "Return my [old order]" | Response explains window expired, no DB change |

**Verification**:
- Programmatic: Query DB after test to verify `return_status` changed (or didn't)
- Check `update_return` tool called only when appropriate

**Note**: Agent uses `interrupt_before` for `update_return`. For evals, we'll auto-approve the interrupt in eval mode.

---

### Category 4: Product QA

**User Goal**: Get information about a product (specs, warranty, troubleshooting, policies).

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Warranty question | "What's the warranty on Shield TV?" | Response contains warranty info from docs |
| Specs question | "What are the specs of RTX 4090?" | Response contains accurate specs |
| Troubleshooting | "My Shield TV won't turn on" | Response contains troubleshooting steps |
| Policy question | "What's your return policy?" | Response contains policy info |

**Verification**:
- Programmatic: `canonical_rag` tool was called
- **LLM Judge**: Given the source chunk that contains the answer, verify the response accurately communicates that information
  - Binary score: Did the response capture the key information?
  - Can provide more detailed scoring if needed
  - On failure, LLM summarizes what was missing or incorrect

---

### Category 5: Out-of-Scope Handling

**User Goal**: (Various off-topic requests)

| Scenario | Example Query | Success Criteria |
|----------|---------------|------------------|
| Weather | "What's the weather?" | Polite redirect to NVIDIA support scope |
| Unrelated product | "Tell me about iPhones" | Explains scope limitation |
| General chat | "How are you?" | Friendly but redirects to assistance |

**Verification**:
- Programmatic: No sub-agent tools called (or only `HandleOtherTalk`)
- Response contains redirect/scope language

---

## Single-Turn vs Multi-Turn

### Single-Turn Tasks

Most tasks can be evaluated in a single turn:
- User sends one message
- Agent responds
- We verify the response

**No LLM simulator needed** - just a static prompt.

### Multi-Turn Tasks

Some scenarios require multiple turns:
- Agent asks for clarification → User provides it → Agent responds
- Follow-up questions about the same topic
- Topic switching mid-conversation

**For multi-turn**:
- Option A: Pre-scripted follow-up messages (deterministic)
- Option B: LLM-simulated user responses (more realistic but adds variance)

We'll start with single-turn and add multi-turn later.

---

## Test Data Strategy

### Using Real Database Data

Tasks are generated from actual DB records, ensuring consistency:

```python
def generate_order_status_task(user_id: str, order: dict) -> EvalTask:
    """Generate an order status task from real DB data."""
    return EvalTask(
        id=f"order_status_{user_id}_{order['order_id']}",
        category="order_status",
        user_id=user_id,
        prompt=f"What's the status of my {order['product_name']} order?",
        ground_truth={
            "order_status": order["order_status"],
            "order_id": order["order_id"],
        },
        response_must_contain=[order["order_status"]],
    )
```

### Database Reset Strategy

Before each task:
1. Reset test DB to known baseline state (clone of production schema + test data)
2. Run the evaluation task
3. Verify final DB state (for write operations like `update_return`)

This allows:
- Any user's data to be used for testing
- Write operations to be tested without corrupting data
- Consistent, reproducible results

---

## Task Definition Schema

```python
@dataclass
class EvalTask:
    # Identity
    id: str
    name: str
    category: str  # order_status, return_status, return_init, product_qa, out_of_scope

    # Test setup
    user_id: str
    prompt: str  # The user's message
    turns: Literal["single", "multi"] = "single"
    followup_prompts: list[str] = None  # For multi-turn (scripted)

    # Ground truth (from DB, populated at task generation time)
    ground_truth: dict = None

    # Programmatic verification
    response_must_contain: list[str] = None
    response_must_not_contain: list[str] = None
    tool_must_be_called: list[str] = None
    tool_must_not_be_called: list[str] = None
    expected_db_state: dict = None  # For write operations

    # LLM Judge (for Product QA)
    use_llm_judge: bool = False
    judge_context: str = None  # The source chunk/doc to verify against
    judge_criteria: str = None  # What to evaluate
```

---

## Verification Functions

### Programmatic Verifiers

```python
def verify_response_contains(response: str, values: list[str]) -> dict:
    """Check if response contains expected values."""
    found = [v for v in values if v.lower() in response.lower()]
    missing = [v for v in values if v.lower() not in response.lower()]
    return {
        "passed": len(missing) == 0,
        "found": found,
        "missing": missing,
    }

def verify_tool_called(trace: dict, tool_name: str) -> bool:
    """Check if a specific tool was called."""
    all_tools = extract_all_tool_calls(trace)
    return tool_name in all_tools

def verify_tool_not_called(trace: dict, tool_name: str) -> bool:
    """Verify a tool was NOT called."""
    return not verify_tool_called(trace, tool_name)

def verify_db_state(user_id: str, order_id: str, expected: dict) -> dict:
    """Query database and verify expected state after task."""
    actual = query_order(user_id, order_id)
    matches = all(actual.get(k) == v for k, v in expected.items())
    return {
        "passed": matches,
        "expected": expected,
        "actual": actual,
    }
```

### LLM Judge (for Product QA)

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
    # Call LLM and parse response
    result = call_judge_llm(prompt)
    return result
```

---

## Execution Flow

```python
def run_eval_task(task: EvalTask) -> EvalResult:
    # 1. Reset database to baseline
    reset_test_database()

    # 2. Run the conversation
    if task.turns == "single":
        trace = call_agent(task.prompt, task.user_id)
        response = extract_final_response(trace)
    else:
        # Multi-turn: iterate through prompts
        response, trace = run_multi_turn(task)

    # 3. Run verifications
    results = {}

    # Response content checks
    if task.response_must_contain:
        results["content"] = verify_response_contains(response, task.response_must_contain)

    if task.response_must_not_contain:
        results["forbidden_content"] = verify_response_not_contains(response, task.response_must_not_contain)

    # Tool call checks
    if task.tool_must_be_called:
        results["required_tools"] = all(verify_tool_called(trace, t) for t in task.tool_must_be_called)

    if task.tool_must_not_be_called:
        results["forbidden_tools"] = all(verify_tool_not_called(trace, t) for t in task.tool_must_not_be_called)

    # Database state checks
    if task.expected_db_state:
        results["db_state"] = verify_db_state(task.user_id, task.ground_truth["order_id"], task.expected_db_state)

    # LLM Judge (for Product QA)
    if task.use_llm_judge:
        results["llm_judge"] = llm_judge_rag_accuracy(response, task.judge_context, task.prompt)

    # 4. Aggregate results
    passed = all(
        r if isinstance(r, bool) else r.get("passed", r.get("score", 0) == 1)
        for r in results.values()
    )

    return EvalResult(
        task_id=task.id,
        category=task.category,
        passed=passed,
        results=results,
        response=response,
        trace=trace,
    )
```

---

## Failure Analysis

When a task fails, we want actionable insights:

```python
@dataclass
class EvalResult:
    task_id: str
    category: str
    passed: bool
    results: dict
    response: str
    trace: dict
    failure_summary: str = None  # LLM-generated summary of what went wrong

def summarize_failure(result: EvalResult) -> str:
    """Use LLM to summarize failure mode for debugging."""
    if result.passed:
        return None

    prompt = f"""An evaluation task failed. Summarize what went wrong in 1-2 sentences.

Task: {result.task_id}
Category: {result.category}

Verification Results:
{json.dumps(result.results, indent=2)}

Agent Response:
{result.response}

What was the failure mode?"""

    return call_llm(prompt)
```

---

## Metrics and Reporting

### Per-Task Metrics

| Metric | Type | Description |
|--------|------|-------------|
| passed | bool | Overall pass/fail |
| response_accurate | bool | Response contains expected info |
| no_forbidden_tools | bool | No forbidden tools called |
| db_state_correct | bool | Database changes as expected |
| llm_judge_score | float | LLM judge score (0-1) |
| latency_ms | float | End-to-end time |

### Aggregate Metrics

| Metric | Formula |
|--------|---------|
| Overall Pass Rate | passed_count / total_count |
| Pass Rate by Category | per-category pass rates |
| Average Latency | mean(latency_ms) |
| P95 Latency | percentile(latency, 95) |

### Report Format

```
NVIDIA Agent Evaluation Report
==============================
Run Date: 2024-12-04
Total Tasks: 40
Passed: 36 (90%)

By Category:
  order_status:   10/10 (100%)
  return_status:  8/10 (80%)   <-- ATTENTION
  return_init:    8/10 (80%)   <-- ATTENTION
  product_qa:     5/5 (100%)
  out_of_scope:   5/5 (100%)

Failed Tasks:
  - return_status_003: update_return called when checking status
    → Agent misinterpreted status query as return request
  - return_status_007: Response missing "Pending Approval"
    → Agent returned wrong status value
  - return_init_002: DB not updated after return request
    → update_return tool failed silently

Common Failure Modes:
  - Return status vs initiation confusion (2 cases)
  - Missing information in response (1 case)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Task definition schema (`EvalTask` dataclass)
- [ ] Agent client (invoke agent with user_id + message)
- [ ] Database reset utility (reset test DB to baseline)
- [ ] Basic programmatic verifiers
- [ ] Single-task runner

### Phase 2: Task Generation
- [ ] Query DB to find users with diverse data patterns
- [ ] Generate tasks from real data for each category:
  - [ ] Order Status tasks
  - [ ] Return Status tasks
  - [ ] Return Initiation tasks
  - [ ] Product QA tasks (need to identify source docs)
  - [ ] Out-of-Scope tasks (can be hand-written)
- [ ] Target: 30-40 tasks across categories

### Phase 3: LLM Judge
- [ ] Implement RAG accuracy judge
- [ ] Implement failure summarizer
- [ ] Test on Product QA category

### Phase 4: Reporting & Analysis
- [ ] Aggregate metrics calculation
- [ ] Report generation
- [ ] Failure mode clustering

### Phase 5: Multi-Turn & Simulator (Future)
- [ ] Add scripted multi-turn tasks
- [ ] Implement LLM user simulator
- [ ] Add disambiguation scenarios

### Phase 6: CI/CD Integration (Future)
- [ ] Run evals on PR
- [ ] Regression detection
- [ ] Automated reporting

---

## Resolved Design Decisions

1. **Test data source**: Use real DB data, reset test DB before each task
2. **Single vs multi-turn**: Start with single-turn, add multi-turn later
3. **Product QA verification**: LLM judge with source chunk as context
4. **Failure analysis**: LLM summarizes failure modes as they happen
5. **Interrupt handling**: Auto-approve `update_return` interrupt in eval mode

---

## Open Questions

1. **Which users have the most diverse data for task generation?**
   - Need to query DB to find good candidates

2. **Source documents for Product QA**:
   - Which docs are indexed in the RAG system?
   - How do we get the "correct" chunk to give to the judge?

3. **Eval mode configuration**:
   - How do we configure agent to auto-approve interrupts?
   - Separate endpoint or config flag?

---

## Next Steps

1. Set up the test database reset mechanism
2. Query production DB to identify users with diverse order/return data
3. Implement core task runner and verifiers
4. Generate initial set of tasks from real data
