# Aragora Load Testing Framework

Comprehensive load testing framework for validating Aragora's performance under various load conditions.

## Overview

The load testing framework provides:
- **Multiple test tools**: Locust (Python), k6 (JavaScript), pytest (Python)
- **Configurable profiles**: Smoke, light, medium, heavy, spike, soak
- **SLO validation**: Automated validation of Service Level Objectives
- **Comprehensive coverage**: API endpoints, WebSocket connections, authentication, knowledge management

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install locust aiohttp websockets

# For k6 tests (optional)
brew install k6  # macOS
# or download from https://k6.io/docs/getting-started/installation/
```

### Run Smoke Test

```bash
# Quick validation with minimal load
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
    --headless -u 5 -r 1 --run-time 1m
```

### Run with Profile

```bash
# Use predefined load profile
ARAGORA_LOAD_PROFILE=medium locust -f tests/load/locustfile.py \
    --host=http://localhost:8080 --headless -u 50 -r 5 --run-time 10m
```

## Test Files

| File | Description | Tool |
|------|-------------|------|
| `locustfile.py` | Main Locust test suite with all user scenarios | Locust |
| `auth_load.py` | Authentication flow load tests | pytest |
| `knowledge_load.py` | Knowledge management load tests | pytest |
| `websocket_load.py` | WebSocket connection tests | pytest |
| `test_concurrent_debates.py` | Concurrent debate processing tests | pytest |
| `test_resilience_load.py` | Resilience layer tests | pytest |
| `gauntlet_load.py` | Gauntlet stress testing | pytest |
| `test_bridge_load.py` | Cross-pollination bridge tests | pytest |
| `profiles.py` | Load test configuration profiles | Python module |
| `slo_validator.py` | SLO validation utilities | Python module |
| `scenarios/` | k6 test scenarios | k6 |

## Load Profiles

| Profile | Users | Duration | Description |
|---------|-------|----------|-------------|
| smoke | 5 | 1 min | Quick CI/CD validation |
| light | 20 | 5 min | Normal operational load |
| medium | 50 | 10 min | Peak traffic simulation |
| heavy | 100 | 15 min | Stress testing |
| spike | 200 | 5 min | Traffic burst testing |
| soak | 30 | 1 hour | Extended stability testing |

### Using Profiles

```bash
# View available profiles
python tests/load/profiles.py list

# Show profile details
python tests/load/profiles.py show --profile medium

# Get Locust command for profile
python tests/load/profiles.py show --profile medium --format locust

# Get k6 options for profile
python tests/load/profiles.py show --profile medium --format k6
```

## Running Tests

### Locust (Web UI)

```bash
# Start with web interface
locust -f tests/load/locustfile.py --host=http://localhost:8080

# Open http://localhost:8089 to configure and start test
```

### Locust (Headless)

```bash
# Basic run
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
    --headless -u 50 -r 5 --run-time 5m

# With authentication
ARAGORA_API_TOKEN=your_token locust -f tests/load/locustfile.py \
    --host=http://localhost:8080 --headless -u 50 -r 5 --run-time 5m

# With specific profile
ARAGORA_LOAD_PROFILE=heavy locust -f tests/load/locustfile.py \
    --host=http://localhost:8080 --headless -u 100 -r 10 --run-time 15m
```

### pytest Tests

```bash
# Run all load tests (requires ARAGORA_LOAD_TEST_ENABLED=1)
ARAGORA_LOAD_TEST_ENABLED=1 pytest tests/load/ -v --asyncio-mode=auto

# Run specific test file
pytest tests/load/auth_load.py -v --asyncio-mode=auto

# Run stress tests
pytest tests/load/ -v -k stress --asyncio-mode=auto

# Run with longer timeout
pytest tests/load/ -v --timeout=600 --asyncio-mode=auto
```

### k6 Tests

```bash
# Run API baseline test
k6 run tests/load/scenarios/api-baseline.js

# Run stress test
k6 run tests/load/scenarios/stress-test.js

# Run WebSocket test
k6 run tests/load/scenarios/websocket-streaming.js
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_API_URL` | API base URL | `http://localhost:8080` |
| `ARAGORA_API_TOKEN` | Authentication token | (none) |
| `ARAGORA_LOAD_PROFILE` | Load profile to use | `light` |
| `ARAGORA_ENABLE_SLO_VALIDATION` | Enable SLO validation | `1` |
| `ARAGORA_LOAD_TEST_ENABLED` | Enable expensive load tests | `0` |
| `ARAGORA_TEST_AGENT` | Agent for debate tests | `echo` |
| `ARAGORA_WS_URL` | WebSocket URL | `ws://localhost:8080/ws` |
| `ARAGORA_CONCURRENT_WS` | Concurrent WebSocket connections | `50` |
| `ARAGORA_AUTH_CONCURRENT` | Concurrent auth operations | `20` |
| `ARAGORA_KM_CONCURRENT` | Concurrent KM operations | `20` |

## SLO Thresholds

Default SLO thresholds for the `light` profile:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| HTTP p95 | 500ms | 95th percentile response time |
| HTTP p99 | 1000ms | 99th percentile response time |
| Error rate | 1% | Maximum acceptable error rate |
| Throughput | 10 rps | Minimum requests per second |

### Validating SLOs

```bash
# Enable SLO validation (default)
ARAGORA_ENABLE_SLO_VALIDATION=1 locust -f tests/load/locustfile.py ...

# Disable SLO validation
ARAGORA_ENABLE_SLO_VALIDATION=0 locust -f tests/load/locustfile.py ...

# Run SLO validator demo
python tests/load/slo_validator.py --demo --profile medium
```

## Test Scenarios

### Health Check User
- Lightweight health probes
- Kubernetes readiness/liveness checks
- Baseline load generation

### API Browsing User
- List debates with pagination
- View debate details
- Search debates
- View leaderboard

### Debate User
- Create new debates
- Poll debate status
- Retrieve debate messages

### Authentication User
- Login attempts
- Token refresh
- Session management
- SSO flow initiation

### Knowledge User
- Create knowledge nodes
- Search knowledge base
- Semantic queries
- Node retrieval

### Heavy Load User
- Burst health checks
- Large list requests
- Rapid debate creation

## Interpreting Results

### Locust Output

```
Name                          # reqs  # fails  Avg  Min  Max  Median  req/s
POST /api/debate                 150    3(2%)   892   45  4521   780   12.5
GET /api/debates/:id             450    0(0%)   156   12   892   145   37.5
GET /api/health                  600    0(0%)    23    5   156    18   50.0
```

- **# reqs**: Total requests made
- **# fails**: Failed requests (percentage)
- **Avg/Min/Max/Median**: Response time in milliseconds
- **req/s**: Requests per second

### SLO Validation Report

```
============================================================
SLO VALIDATION REPORT
============================================================
Result: PASSED

WARNINGS:
  [WARNING] p50 response time 250ms exceeds expected 200ms

METRICS SUMMARY:
  response_time_p95_ms: 450.00
  response_time_p99_ms: 890.00
  error_rate: 0.01
  throughput_rps: 25.50
============================================================
```

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start services
        run: docker-compose up -d

      - name: Run smoke test
        run: |
          pip install locust
          locust -f tests/load/locustfile.py \
            --host=http://localhost:8080 \
            --headless -u 5 -r 1 --run-time 1m
        env:
          ARAGORA_LOAD_PROFILE: smoke
          ARAGORA_ENABLE_SLO_VALIDATION: 1
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure the server is running
2. **401 Unauthorized**: Set `ARAGORA_API_TOKEN` for authenticated endpoints
3. **429 Rate Limited**: Reduce spawn rate or user count
4. **Timeouts**: Increase test timeout or check server performance

### Debug Mode

```bash
# Enable verbose output
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
    --headless -u 5 -r 1 --run-time 1m --loglevel DEBUG
```

## Adding New Tests

### Adding a Locust User

```python
class CustomTasks(TaskSet):
    @task(5)
    def my_task(self) -> None:
        self.client.get("/api/custom", name="GET /api/custom")

class CustomUser(HttpUser):
    tasks = [CustomTasks]
    wait_time = between(1, 5)
    weight = 2
```

### Adding a pytest Test

```python
@pytest.mark.asyncio
async def test_custom_load():
    metrics = await run_custom_load_test(
        concurrent=10,
        duration=30.0,
    )
    assert metrics.success_rate >= 0.95
```

## License

See the main project LICENSE file.
