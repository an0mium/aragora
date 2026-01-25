# Python Debate CLI

A full-featured command-line tool demonstrating the Aragora Python SDK.

## Setup

```bash
# Install Aragora SDK
pip install aragora-client python-dotenv

# Start the server (in another terminal)
python -m aragora.server.unified_server --port 8080

# Set API keys (for debate functionality)
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Commands

### Debates

```bash
# Run a simple debate
python main.py debate "Should we use Kubernetes or Docker Swarm?"

# Specify agents and rounds
python main.py debate "Design a rate limiter" --agents claude gpt gemini --rounds 5

# Stream a debate in real-time (WebSocket)
python main.py stream "What's the best database for our use case?"
```

### Gauntlet Validation

```bash
# Validate a document
python main.py gauntlet policy.md --persona gdpr

# Different validation profiles
python main.py gauntlet contract.md --persona legal --profile thorough

# Code review
python main.py gauntlet src/auth.py --type code --persona security
```

### Agent Rankings

```bash
# View agent leaderboard
python main.py rankings

# Show more agents
python main.py rankings --limit 20
```

### Tournaments

```bash
# Create a tournament
python main.py tournament create --name "Q1 Showdown" --agents claude gpt gemini mistral

# Create and wait for completion
python main.py tournament create --name "Finals" --agents claude gpt --format round_robin --wait

# List tournaments
python main.py tournament list
```

### Authentication

```bash
# Login
python main.py auth login --email user@example.com

# Logout
python main.py auth logout

# Manage API keys
python main.py auth apikeys list
python main.py auth apikeys create --name "CI/CD" --scopes debates:read debates:write
```

### Onboarding

```bash
# Start or continue onboarding
python main.py onboarding

# Complete the current step
python main.py onboarding --complete

# Use a specific template
python main.py onboarding --template developer
```

### System

```bash
# Check server health
python main.py health

# View memory analytics
python main.py memory
```

## Example Output

```
$ python main.py debate "React vs Vue for a new project?"

Creating debate...
  Topic: React vs Vue for a new project?
  Agents: anthropic-api, openai-api
  Rounds: 3

Results:
  Consensus: Yes
  Confidence: 87.5%
  Rounds completed: 3

Final Answer:
  Both frameworks are excellent choices. React offers a larger ecosystem
  and job market, while Vue provides gentler learning curve and better
  documentation. For a team new to frontend development, Vue is recommended.
  For a team with React experience or needing enterprise support, React
  is the better choice.
```

```
$ python main.py tournament create --name "AI Olympics" --agents claude gpt gemini --wait

Creating tournament...
  Name: AI Olympics
  Participants: claude, gpt, gemini
  Format: single_elimination

Tournament created: abc123
  Status: in_progress
  Total matches: 3

Waiting for tournament completion...
  Progress: 1/3 matches
  Progress: 2/3 matches
  Progress: 3/3 matches

Final Standings:
  1. claude - 2W/0L
  2. gpt - 1W/1L
  3. gemini - 0W/1L
```

## Using a Different Server

```bash
# Connect to a different Aragora instance
python main.py --server https://aragora.example.com debate "Topic here"
```

## Environment Variables

Create a `.env` file for convenience:

```bash
# .env
ARAGORA_SERVER=http://localhost:8080
ARAGORA_API_KEY=your-api-key
```
