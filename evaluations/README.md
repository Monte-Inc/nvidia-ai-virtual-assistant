# Evaluations

End-to-end evaluation framework for the NVIDIA AI Virtual Assistant agent.

## Setup

```bash
# Install agent dependencies (if not already installed)
pip install -r src/agent/requirements.txt

# Create the test database (run once)
python -m evaluations.main --setup-db
```

## Running Evaluations

```bash
# Run all evaluations
python -m evaluations.main

# Run specific category
python -m evaluations.main --category order_status

# Run specific task by ID
python -m evaluations.main --task order_status_001

# Run multiple categories
python -m evaluations.main -c order_status -c return_status

# Run multiple specific tasks
python -m evaluations.main -t order_status_001 -t order_status_002

# Limit to first N tasks (useful for quick testing)
python -m evaluations.main --limit 5

# Run from a specific JSON file
python -m evaluations.main --file evaluations/tasks/order_status.json

# List available tasks
python -m evaluations.main --list

# Quiet mode (summary only)
python -m evaluations.main --quiet
```

## Database Management

Evaluations use a separate test database (`customer_data_test`) for isolation.

```bash
# Setup test database (create + populate)
python -m evaluations.main --setup-db

# Reset test database to baseline
python -m evaluations.main --reset-db

# Drop test database
python -m evaluations.main --drop-db

# Skip database reset between tasks (faster, less isolated)
python -m evaluations.main --no-db-reset
```

## Directory Structure

```
evaluations/
├── main.py              # CLI entry point
├── core/                # Core evaluation infrastructure
├── tasks/               # JSON task definitions
├── tests/               # Unit tests
└── tools/               # Utility scripts
```

## Adding New Tasks

Create a JSON file in `evaluations/tasks/`:

```json
{
  "description": "Order status evaluation tasks",
  "tasks": [
    {
      "id": "order_status_001",
      "name": "Check delivered order status",
      "category": "order_status",
      "user_id": "4165",
      "prompt": "What's the status of my Jetson Nano order?",
      "ground_truth": {
        "order_id": 52768,
        "order_status": "Delivered"
      },
      "response_must_contain": ["Delivered"],
      "tool_must_not_be_called": ["update_return"]
    }
  ]
}
```

### Task Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique task identifier |
| `name` | Yes | Human-readable task name |
| `category` | Yes | Task category (order_status, return_status, return_init, product_qa, out_of_scope) |
| `user_id` | Yes | Customer ID for the test |
| `prompt` | Yes | User message to send to agent |
| `ground_truth` | No | Expected data from database |
| `response_must_contain` | No | Strings that must appear in response |
| `response_must_not_contain` | No | Strings that must NOT appear |
| `tool_must_be_called` | No | Tools that must be called |
| `tool_must_not_be_called` | No | Tools that must NOT be called |
| `expected_db_state` | No | Expected database state after task |

## Running Tests

```bash
# Run unit tests
python -m pytest evaluations/tests/ -v -m "not integration"
```
