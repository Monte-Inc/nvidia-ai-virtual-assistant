# Evaluation Test Suite

This directory contains pytest tests for the evaluation framework core modules. The tests validate the schema definitions, database utilities, agent client, and verification functions.

## Test Structure

```
tests/
├── __init__.py
├── test_schema.py      # Tests for EvalTask, EvalResult, enums
├── test_db.py          # Tests for TestDatabase, OrderRecord
├── test_client.py      # Tests for AgentClient, AgentResponse
└── test_verifiers.py   # Tests for all verification functions
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest evaluations/tests/ -v

# With coverage
pytest evaluations/tests/ -v --cov=evaluations/core
```

### Run Specific Test Files

```bash
pytest evaluations/tests/test_schema.py -v
pytest evaluations/tests/test_verifiers.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run all tests in a class
pytest evaluations/tests/test_verifiers.py::TestVerifyResponseContains -v

# Run a specific test method
pytest evaluations/tests/test_verifiers.py::TestVerifyResponseContains::test_single_value_found -v
```

### Skip Integration Tests

Integration tests require a running agent or database. They are marked with `@pytest.mark.integration`:

```bash
# Run only unit tests (skip integration)
pytest evaluations/tests/ -v -m "not integration"

# Run only integration tests
pytest evaluations/tests/ -v -m integration
```

## Test Categories

### Unit Tests

These tests run without external dependencies (no agent, no database connection):

| File | Tests | Description |
|------|-------|-------------|
| `test_schema.py` | 11 | Schema model creation and validation |
| `test_db.py` (partial) | 14 | CSV loading, column mapping, config |
| `test_client.py` (partial) | 6 | SSE parsing, client initialization |
| `test_verifiers.py` | 23 | All verification function logic |

### Integration Tests

These tests require external services:

| File | Tests | Requirements |
|------|-------|--------------|
| `test_db.py` | 2 | PostgreSQL database |
| `test_client.py` | 3 | Running agent service |

## Test File Details

### test_schema.py

Tests for the Pydantic schema models:

```python
class TestTaskCategory:
    # Verifies all task categories are defined
    # Verifies category string values

class TestTaskStatus:
    # Verifies all task statuses are defined

class TestEvalTask:
    # test_minimal_task_creation - Required fields only
    # test_full_task_creation - All fields populated
    # test_multi_turn_task - Multi-turn with followup prompts
    # test_llm_judge_task - LLM judge configuration

class TestVerificationResult:
    # test_passed_verification - Passing result
    # test_failed_verification - Failing result with error

class TestEvalResult:
    # test_passed_result - Passed status
    # test_failed_result - Failed with verifications
    # test_error_result - Error status
    # test_result_with_tool_calls - Tool call capture
    # test_all_verifications_passed_property - Computed property
```

### test_db.py

Tests for database utilities:

```python
class TestDBConfig:
    # test_default_config - Default environment variable handling
    # test_connection_params - Connection parameter generation

class TestOrderRecord:
    # test_from_db_row_full - Complete record parsing
    # test_from_db_row_with_return - Record with return data
    # test_from_db_row_handles_missing_fields - Graceful handling

class TestCSVColumnMapping:
    # test_all_expected_columns_mapped - CSV column coverage
    # test_column_mappings_are_lowercase - DB column naming

class TestTestDatabase:
    # test_init_with_default_config - Default initialization
    # test_init_with_custom_config - Custom config
    # test_load_baseline_from_csv - CSV loading
    # test_baseline_data_has_correct_columns - Column mapping
    # test_baseline_data_for_user_4165 - Test user data
    # test_load_csv_file_not_found - Error handling
    # test_reset_without_baseline_raises_error - Error handling

class TestTestDatabaseIntegration:  # @pytest.mark.integration
    # test_full_setup_and_query - End-to-end DB test
    # test_verify_return_status - Return status verification
```

### test_client.py

Tests for the agent client:

```python
class TestAgentResponse:
    # test_default_values - Default field values
    # test_full_response - All fields populated

class TestAgentClient:
    # test_init_default_values - Default configuration
    # test_init_custom_values - Custom configuration
    # test_parse_sse_response_simple - Basic SSE parsing
    # test_parse_sse_response_multiple_chunks - Multi-chunk SSE
    # test_parse_sse_response_ignores_done - [DONE] handling
    # test_parse_sse_response_handles_invalid_json - Error handling
    # test_parse_sse_response_empty - Empty response

class TestAgentClientIntegration:  # @pytest.mark.integration
    # test_create_and_end_session - Session lifecycle
    # test_invoke_order_status - Order status query
    # test_invoke_multi_turn - Multi-turn conversation
```

### test_verifiers.py

Tests for verification functions:

```python
class TestVerifyResponseContains:
    # test_single_value_found - Single match
    # test_single_value_missing - Single missing
    # test_multiple_values_all_found - All present
    # test_multiple_values_some_missing - Partial match
    # test_case_insensitive_by_default - Default case handling
    # test_case_sensitive_when_specified - Case sensitivity option
    # test_empty_values_list - Empty list edge case

class TestVerifyResponseNotContains:
    # test_forbidden_value_absent - Correctly absent
    # test_forbidden_value_present - Found forbidden value
    # test_case_insensitive_by_default - Default case handling

class TestVerifyResponseMatchesPattern:
    # test_pattern_matches - Regex match
    # test_pattern_no_match - No match
    # test_invalid_pattern - Invalid regex handling

class TestVerifyToolCalled:
    # test_tool_was_called - Tool found
    # test_tool_was_not_called - Tool not found
    # test_empty_tool_calls - Empty list

class TestVerifyToolNotCalled:
    # test_tool_correctly_not_called - Correctly absent
    # test_tool_incorrectly_called - Found forbidden tool

class TestVerifyToolsCalled:
    # test_all_required_tools_called - All present
    # test_some_required_tools_missing - Partial match

class TestVerifyToolsNotCalled:
    # test_no_forbidden_tools_called - None found
    # test_forbidden_tool_was_called - Found forbidden

class TestVerifyToolCallArgs:
    # test_correct_args - Matching arguments
    # test_wrong_args - Mismatched arguments
    # test_tool_not_called - Tool not found

class TestVerifier:
    # test_verify_task_response_contains - Response check
    # test_verify_task_response_not_contains - Exclusion check
    # test_verify_task_tool_checks - Tool call checks
    # test_verify_task_multiple_checks - Combined checks
```

## Writing New Tests

### Test Naming Convention

```python
def test_<what>_<condition>():
    """Descriptive docstring."""
    pass

# Examples:
def test_response_contains_single_value():
def test_tool_called_when_required():
def test_db_state_mismatch_fails():
```

### Test Structure

```python
class TestFeatureName:
    """Tests for feature_name function/class."""

    def test_happy_path(self):
        """Basic successful case."""
        result = function_under_test(valid_input)
        assert result.passed is True

    def test_edge_case(self):
        """Edge case handling."""
        result = function_under_test(edge_input)
        assert result.passed is True

    def test_error_handling(self):
        """Error case handling."""
        result = function_under_test(invalid_input)
        assert result.passed is False
        assert "expected error" in result.error
```

### Integration Test Pattern

```python
class TestFeatureIntegration:
    """Integration tests requiring external services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_with_live_service(self):
        """Test with actual service."""
        async with SomeClient() as client:
            result = await client.do_something()
            assert result is not None
```

### Fixtures

Common fixtures can be added to `conftest.py`:

```python
# evaluations/tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def csv_path():
    """Path to test CSV file."""
    return Path(__file__).parent.parent.parent / "data" / "orders.csv"

@pytest.fixture
def sample_task():
    """Sample EvalTask for testing."""
    from evaluations.core import EvalTask, TaskCategory
    return EvalTask(
        id="test_001",
        name="Test task",
        category=TaskCategory.ORDER_STATUS,
        user_id="4165",
        prompt="Test prompt",
    )

@pytest.fixture
def sample_tool_calls():
    """Sample tool calls for testing."""
    return [
        {"name": "get_purchase_history", "input": {"user_id": "4165"}},
        {"name": "structured_rag", "input": {"query": "order status"}},
    ]
```

## Test Data

Tests use the CSV file at `data/orders.csv` for baseline data. Key test user:

| User ID | Orders | Use Cases |
|---------|--------|-----------|
| `4165` | 10 | Order status, return status, return initiation |

Key orders for user 4165:
- Jetson Nano (order 52768): Delivered, no return
- RTX 4090 (order 4065): Return Requested

## Continuous Integration

Tests should run on every PR. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install pytest pytest-asyncio pydantic httpx psycopg2-binary
      - run: pytest evaluations/tests/ -v -m "not integration"
```

## Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'evaluations'`**

Run from project root or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/ai-virtual-assistant:$PYTHONPATH
pytest evaluations/tests/ -v
```

**`PytestUnknownMarkWarning: Unknown pytest.mark.integration`**

Register the mark in `pytest.ini` or `pyproject.toml`:
```ini
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
```

**Integration tests failing**

Ensure services are running:
- Agent: `http://localhost:8081`
- PostgreSQL: `localhost:5432`

Or skip integration tests:
```bash
pytest evaluations/tests/ -v -m "not integration"
```
