# Gauntlet Evaluation Fixtures

Fixtures are deterministic inputs for `benchmarks/gauntlet_evaluation.py`.

Each JSON fixture includes:
- `input_content`: The artifact to stress-test
- `attack_responses`: Stubbed red-team outputs used by the harness
- `expected`: Expected severity counts for validation

Add new fixtures by copying an existing JSON file and editing values.
