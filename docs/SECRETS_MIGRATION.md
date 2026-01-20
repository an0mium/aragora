# Secrets Management Migration Plan

> **Status:** Phase 1 Complete
> **Created:** 2026-01-19
> **Updated:** 2026-01-20
> **Target Completion:** Q2 2026

## ⚠️ IMMEDIATE ACTION: API Key Rotation

**Priority: HIGH** - API keys were found in the `.env` file which may have been exposed.

### Step 1: Rotate All API Keys

| Provider | Dashboard URL | Action |
|----------|---------------|--------|
| Anthropic | https://console.anthropic.com/settings/keys | Regenerate `ANTHROPIC_API_KEY` |
| OpenAI | https://platform.openai.com/api-keys | Regenerate `OPENAI_API_KEY` |
| Mistral | https://console.mistral.ai/api-keys | Regenerate `MISTRAL_API_KEY` |
| Google AI | https://aistudio.google.com/apikey | Regenerate `GEMINI_API_KEY` |
| xAI | https://console.x.ai/team/api-keys | Regenerate `XAI_API_KEY` |
| OpenRouter | https://openrouter.ai/keys | Regenerate `OPENROUTER_API_KEY` |

### Step 2: Update Local `.env`

After rotation, update your local `.env` file:
```bash
# In project root
cp .env .env.backup  # Backup old file
# Edit .env with new keys
```

### Step 3: Add Secrets to GitHub

✅ **Completed 2026-01-20** - All API keys added via `gh secret set`:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `XAI_API_KEY`
- `GROK_API_KEY`

To add more secrets securely (value never in command line):
```bash
grep "^SECRET_NAME=" .env | cut -d= -f2- | gh secret set SECRET_NAME
```

### Step 4: Verify `.gitignore`

Ensure `.env` is in `.gitignore`:
```bash
grep "^\.env$" .gitignore || echo ".env" >> .gitignore
```

### Step 5: Install Pre-commit Hooks

The gitleaks pre-commit hook prevents future secret commits:
```bash
pre-commit install
pre-commit run gitleaks --all-files  # Test it
```

---

## Current State

### Local Development
- Secrets stored in `.env` file (gitignored)
- Loaded via `python-dotenv` or environment variables
- Keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, etc.

### Production
- Environment variables set in deployment platform
- No centralized secrets management
- No audit trail for secret access

## Target State

### Development
- **GitHub Codespaces**: Use GitHub Secrets for cloud development
- **Local**: Continue using `.env` with `.env.example` template

### CI/CD (GitHub Actions)
- **GitHub Secrets** for all API keys and credentials
- Encrypted at rest, injected at runtime
- No secrets in logs (masked automatically)

### Production (Kubernetes)
- **AWS Secrets Manager** for centralized secrets
- Automatic rotation for supported credentials
- IAM-based access control
- Audit logging via CloudTrail

## Migration Phases

### Phase 1: GitHub Actions Integration (Week 1-2)

**Objective:** Move all CI/CD secrets to GitHub Secrets

**Status:** ✅ Workflows configured, awaiting secrets in GitHub settings

The following workflows are already configured to use GitHub Secrets:

| Workflow | Secrets Used | Status |
|----------|--------------|--------|
| `test.yml` | `CODECOV_TOKEN`, `GITHUB_TOKEN` | ✅ Configured |
| `build.yml` | `GITHUB_TOKEN` | ✅ Configured |
| `aragora-gauntlet.yml` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | ✅ Configured |
| `deploy-secure.yml` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `AWS_ACCOUNT_ID`, `AWS_DEPLOY_ROLE_NAME` | ✅ Configured |
| `e2e.yml` | None (uses demo mode) | ✅ Complete |

**Action Required:** Add secrets to GitHub repository settings (Settings → Secrets and variables → Actions)

1. **Inventory current secrets:**
   ```
   ANTHROPIC_API_KEY     - Required for Claude API
   OPENAI_API_KEY        - Required for GPT API
   MISTRAL_API_KEY       - Optional for Mistral API
   OPENROUTER_API_KEY    - Fallback provider
   GEMINI_API_KEY        - Optional for Gemini
   XAI_API_KEY           - Optional for Grok
   SUPABASE_URL          - Cloud database
   SUPABASE_KEY          - Cloud database auth
   DATABASE_URL          - PostgreSQL connection
   ```

2. **Add to GitHub repository settings:**
   - Navigate to Settings > Secrets and variables > Actions
   - Add each secret with appropriate scope (repository/environment)

3. **Update GitHub Actions workflows:**
   ```yaml
   # Example: .github/workflows/test.yml
   env:
     ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
   ```

4. **Test in CI pipeline:**
   - Verify secrets are accessible
   - Confirm masking works in logs

### Phase 2: AWS Secrets Manager Setup (Week 3-4)

**Objective:** Establish production secrets infrastructure

1. **Create secrets in AWS Secrets Manager:**
   ```bash
   aws secretsmanager create-secret \
     --name aragora/production/api-keys \
     --secret-string '{"ANTHROPIC_API_KEY":"...","OPENAI_API_KEY":"..."}'
   ```

2. **Set up IAM policies:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "secretsmanager:GetSecretValue"
         ],
         "Resource": "arn:aws:secretsmanager:*:*:secret:aragora/*"
       }
     ]
   }
   ```

3. **Create Kubernetes ExternalSecret:**
   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: ExternalSecret
   metadata:
     name: aragora-api-keys
   spec:
     refreshInterval: 1h
     secretStoreRef:
       name: aws-secrets-manager
       kind: ClusterSecretStore
     target:
       name: aragora-secrets
     data:
       - secretKey: ANTHROPIC_API_KEY
         remoteRef:
           key: aragora/production/api-keys
           property: ANTHROPIC_API_KEY
   ```

### Phase 3: Application Integration (Week 5-6)

**Objective:** Update application to use secrets manager

1. **Add secrets manager client:**
   ```python
   # aragora/config/secrets.py
   import boto3
   from functools import lru_cache

   @lru_cache(maxsize=1)
   def get_secrets_client():
       return boto3.client('secretsmanager')

   def get_secret(name: str) -> dict:
       client = get_secrets_client()
       response = client.get_secret_value(SecretId=name)
       return json.loads(response['SecretString'])
   ```

2. **Create fallback chain:**
   ```python
   def get_api_key(key_name: str) -> str:
       # 1. Check environment variable (local dev)
       if value := os.getenv(key_name):
           return value

       # 2. Check AWS Secrets Manager (production)
       if os.getenv('AWS_REGION'):
           secrets = get_secret('aragora/production/api-keys')
           if key_name in secrets:
               return secrets[key_name]

       raise ValueError(f"Secret {key_name} not found")
   ```

3. **Update configuration loading:**
   - Modify `aragora/config/__init__.py`
   - Add graceful fallback for local development

### Phase 4: Rotation Setup (Week 7-8)

**Objective:** Enable automatic secret rotation

1. **Configure rotation for API keys:**
   - Note: Most LLM providers don't support automatic rotation
   - Set up manual rotation reminders (90-day cycle)

2. **Enable CloudTrail logging:**
   ```bash
   aws cloudtrail create-trail \
     --name aragora-secrets-audit \
     --s3-bucket-name aragora-audit-logs
   ```

3. **Create rotation alerts:**
   - CloudWatch alarm for rotation failures
   - Slack notification for manual rotation reminders

## Verification Checklist

### GitHub Secrets
- [x] All secrets added to repository settings (2026-01-20)
- [x] CI workflows updated to use `${{ secrets.* }}`
- [ ] Test workflow passes with secrets
- [ ] Secrets masked in action logs

### AWS Secrets Manager
- [ ] Secrets created in Secrets Manager
- [ ] IAM policies configured
- [ ] Kubernetes ExternalSecret working
- [ ] Application can retrieve secrets
- [ ] CloudTrail logging enabled

### Application
- [ ] Fallback chain works (env → secrets manager)
- [ ] Local development still works with `.env`
- [ ] Production deployment uses secrets manager
- [ ] No secrets in application logs

## Rollback Plan

If issues occur during migration:

1. **Immediate:** Revert to environment variables
   ```bash
   kubectl set env deployment/aragora ANTHROPIC_API_KEY=<value>
   ```

2. **CI/CD:** Revert workflow changes
   ```bash
   git revert <commit-hash>
   ```

3. **Long-term:** Document issue and plan fix

## Cost Estimate

| Service | Monthly Cost |
|---------|-------------|
| AWS Secrets Manager | ~$5 (10 secrets, 10K API calls) |
| CloudTrail | ~$2 (audit logging) |
| **Total** | ~$7/month |

## Related Documentation

- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [GitHub Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [External Secrets Operator](https://external-secrets.io/)
