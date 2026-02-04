# Human Action Steps - Step-by-Step Instructions

**Last Updated:** February 4, 2026

These are the remaining actions that require human intervention.

---

## 1. Fix Local Claude-Mem Connection

**What:** The claude-mem worker service needs to be restarted on your local machine.

**Steps:**

```bash
# Step 1: Navigate to your claude-mem directory
cd /path/to/claude-mem

# Step 2: Restart the worker
npm run worker:restart

# Step 3: Verify it's running (should return {"status":"ok"})
curl http://127.0.0.1:37777/health
```

**Expected Result:** You should see `{"status":"ok"}` from the health check.

**If it fails:**
```bash
# Check the logs
npm run worker:logs

# Or start in foreground to see errors
npm run worker:start
```

---

## 2. Verify AWS Secrets in Both Regions

**What:** Confirm the supermemory API key and other secrets are replicated to both regions.

**Prerequisites:**
- AWS CLI installed (`aws --version`)
- AWS credentials configured (`aws configure` or environment variables)

**Steps:**

```bash
# Step 1: Check secrets exist in us-east-1
aws secretsmanager list-secrets \
  --region us-east-1 \
  --query "SecretList[?contains(Name, 'aragora')].Name" \
  --output table

# Step 2: Check secrets exist in us-east-2
aws secretsmanager list-secrets \
  --region us-east-2 \
  --query "SecretList[?contains(Name, 'aragora')].Name" \
  --output table

# Step 3: Verify supermemory key value (us-east-1)
aws secretsmanager get-secret-value \
  --region us-east-1 \
  --secret-id aragora/api/supermemory \
  --query 'SecretString' \
  --output text

# Step 4: Verify supermemory key value (us-east-2)
aws secretsmanager get-secret-value \
  --region us-east-2 \
  --secret-id aragora/api/supermemory \
  --query 'SecretString' \
  --output text
```

**Expected Result:** Both regions should show the same secret value.

**If secrets don't exist, create them:**
```bash
# Create the secret in us-east-1
aws secretsmanager create-secret \
  --region us-east-1 \
  --name aragora/api/supermemory \
  --secret-string '{"SUPERMEMORY_API_KEY":"your-api-key-here"}'

# Replicate to us-east-2
aws secretsmanager replicate-secret-to-regions \
  --region us-east-1 \
  --secret-id aragora/api/supermemory \
  --add-replica-regions Region=us-east-2
```

---

## 3. Register for Free APIs

### 3a. GovInfo API Key (Free)

**What:** Get a free API key for US government documents.

**Steps:**

1. Go to: https://api.data.gov/signup/
2. Fill in:
   - First Name: [your name]
   - Last Name: [your name]
   - Email: [your email]
3. Click "Sign Up"
4. Check your email for the API key
5. Add to your environment:
   ```bash
   export GOVINFO_API_KEY="your-api-key-here"
   ```
   Or add to `.env` file:
   ```
   GOVINFO_API_KEY=your-api-key-here
   ```

**Note:** The key works immediately after registration.

---

### 3b. NICE Guidance API Key (Free with Registration)

**What:** Get API access for UK clinical guidelines.

**Steps:**

1. Go to: https://developer.nice.org.uk/
2. Click "Register" or "Get Started"
3. Create a developer account
4. Navigate to "My Apps" → "Create New App"
5. Copy your API key
6. Add to environment:
   ```bash
   export NICE_API_KEY="your-api-key-here"
   ```

**Note:** May take up to 24 hours for approval.

---

## 4. Configure Premium Connectors (If Available)

**What:** These require enterprise licenses. Only configure if you have access.

### 4a. Westlaw (Thomson Reuters)

**If you have a Westlaw API license:**

```bash
# Add to .env or environment
export WESTLAW_API_BASE="https://api.westlaw.com/v1"  # Or your endpoint
export WESTLAW_API_KEY="your-westlaw-api-key"
```

**To verify configuration:**
```bash
python -c "
from aragora.connectors.legal.westlaw import get_config_status
print(get_config_status())
"
```

Should show `'configured': True`

---

### 4b. LexisNexis

**If you have a Lexis API license:**

```bash
export LEXIS_API_BASE="https://api.lexisnexis.com/v1"  # Or your endpoint
export LEXIS_API_KEY="your-lexis-api-key"
```

---

### 4c. FASB GAAP (Internal Proxy)

**If you have an internal FASB content proxy:**

```bash
export FASB_API_BASE="https://internal.yourcompany.com/api/gaap"
export FASB_API_KEY="your-api-key"  # Optional depending on your proxy

# Optional: customize search parameters
export FASB_SEARCH_METHOD="POST"  # GET or POST
export FASB_SEARCH_QUERY_PARAM="query"  # Parameter name for search query
export FASB_SEARCH_LIMIT_PARAM="maxResults"  # Parameter name for limit
```

---

### 4d. IRS Tax Guidance (Internal Proxy)

**If you have an internal IRS content proxy:**

```bash
export IRS_API_BASE="https://internal.yourcompany.com/api/irs"
export IRS_API_KEY="your-api-key"  # Optional

# Optional: customize search parameters
export IRS_SEARCH_METHOD="GET"
export IRS_SEARCH_QUERY_PARAM="q"
export IRS_SEARCH_LIMIT_PARAM="limit"
```

---

## 5. Deploy Claude-Mem to EC2 Instances

**What:** Install and configure the claude-mem worker service on each EC2 instance.

**Steps for each EC2 instance:**

```bash
# Step 1: SSH to the instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Step 2: Navigate to aragora
cd /opt/aragora

# Step 3: Copy the systemd service file
sudo cp deploy/systemd/claude-mem.service /etc/systemd/system/

# Step 4: Create config directory
sudo mkdir -p /etc/aragora

# Step 5: Create environment file from example
sudo cp deploy/systemd/claude-mem.env.example /etc/aragora/claude-mem.env

# Step 6: Edit the environment file with your values
sudo nano /etc/aragora/claude-mem.env
# Set: SUPERMEMORY_API_KEY=your-key
# Set: SUPERMEMORY_BASE_URL=https://api.supermemory.ai

# Step 7: Reload systemd
sudo systemctl daemon-reload

# Step 8: Enable and start the service
sudo systemctl enable claude-mem
sudo systemctl start claude-mem

# Step 9: Verify it's running
sudo systemctl status claude-mem
curl http://127.0.0.1:37777/health
```

**Expected Result:** Service shows "active (running)" and health check returns OK.

**Troubleshooting:**
```bash
# View logs
sudo journalctl -u claude-mem -f

# Restart if needed
sudo systemctl restart claude-mem
```

---

## Quick Verification Commands

After completing all steps, run these to verify everything is configured:

```bash
# From aragora directory
cd /path/to/aragora

# Check all connector configuration status
python -c "
from aragora.connectors.accounting.gaap import get_config_status as fasb
from aragora.connectors.accounting.irs import get_config_status as irs
from aragora.connectors.legal.westlaw import get_config_status as westlaw
from aragora.connectors.legal.lexis import get_config_status as lexis

print('=== Connector Configuration Status ===')
print(f'FASB GAAP: {\"Configured\" if fasb()[\"configured\"] else \"Not configured\"}')
print(f'IRS:       {\"Configured\" if irs()[\"configured\"] else \"Not configured\"}')
print(f'Westlaw:   {\"Configured\" if westlaw()[\"configured\"] else \"Not configured\"}')
print(f'Lexis:     {\"Configured\" if lexis()[\"configured\"] else \"Not configured\"}')
"

# Check local claude-mem
curl -s http://127.0.0.1:37777/health && echo " - Local claude-mem OK"
```

---

## Summary Checklist

| # | Task | Required? | Status |
|---|------|-----------|--------|
| 1 | Fix local claude-mem | Yes | [ ] |
| 2 | Verify AWS secrets | Yes | [ ] |
| 3a | Register GovInfo API | Recommended | [ ] |
| 3b | Register NICE API | Optional | [ ] |
| 4a | Configure Westlaw | If licensed | [ ] |
| 4b | Configure Lexis | If licensed | [ ] |
| 4c | Configure FASB proxy | If available | [ ] |
| 4d | Configure IRS proxy | If available | [ ] |
| 5 | Deploy claude-mem to EC2 | Yes (production) | [ ] |

---

*Generated by Aragora deployment automation*
