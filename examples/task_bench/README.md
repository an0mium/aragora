# Task Bench Artifacts

This directory stores example outputs from the task benchmark harness.

Generate (demo agents, fast profile):
```bash
python benchmarks/task_bench.py --mode demo --profile fast --output-dir examples/task_bench/demo
```

Optional: include Pulse trending context (requires network access):
```bash
python benchmarks/task_bench.py --mode demo --profile fast --enable-trending --output-dir examples/task_bench/demo
```

Artifacts:
- `results.jsonl`
- `summary.json`
