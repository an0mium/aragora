# Task Bench Suite

This directory contains curated task cases for `benchmarks/task_bench.py`.
Each JSON file defines a single task with expected keywords for scoring.

Fields:
- id: unique task identifier
- kind: debate | review | gauntlet
- title: human-readable title
- input: task prompt / diff / policy text
- expected_keywords: list of keywords to match in outputs
- tags: classification tags
- context (optional): extra context for debate or gauntlet tasks
- attack_categories / probe_categories (optional): gauntlet config hints
- input_type (optional): gauntlet input_type override

Run:
```bash
python benchmarks/task_bench.py --mode demo --profile fast
python benchmarks/task_bench.py --mode demo --profile fast --enable-trending  # optional
```
