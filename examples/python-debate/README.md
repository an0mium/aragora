# Python Debate CLI

A simple command-line tool demonstrating the Aragora Python SDK.

## Setup

```bash
# Install Aragora
pip install aragora

# Start the server (in another terminal)
python -m aragora.server.unified_server --port 8080

# Set API keys
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Usage

### Run a Debate

```bash
python main.py debate "Should we use Kubernetes or Docker Swarm?"
```

### Stream a Debate in Real-time

```bash
python main.py stream "Design a rate limiter API"
```

### Run Gauntlet Validation

```bash
python main.py gauntlet policy.md --persona gdpr
```

### View Agent Rankings

```bash
python main.py rankings
```

## Example Output

```
$ python main.py debate "React vs Vue for a new project?"

Creating debate...
  Debate ID: abc123
  Agents: anthropic-api, openai-api
  Rounds: 3

Waiting for completion...
  Round 1/3 complete
  Round 2/3 complete
  Round 3/3 complete

Results:
  Consensus: Yes
  Confidence: 87.5%

Final Answer:
  Both frameworks are excellent choices. React offers a larger ecosystem
  and job market, while Vue provides gentler learning curve and better
  documentation. For a team new to frontend development, Vue is recommended.
  For a team with React experience or needing enterprise support, React
  is the better choice.
```
