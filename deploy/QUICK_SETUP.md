# Quick Cloud Setup Guide

## 1. Cloudflare Pages (Frontend)

### Create Project
```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create Pages project
cd aragora/live
npm ci && npm run build
wrangler pages project create aragora
```

### Get Credentials
1. **API Token**: https://dash.cloudflare.com/profile/api-tokens
   - Click "Create Token"
   - Use "Edit Cloudflare Pages" template
   - Copy the token

2. **Account ID**:
   - Go to https://dash.cloudflare.com
   - Click any domain or Workers & Pages
   - Account ID is in the right sidebar

### Set GitHub Secrets
```bash
gh secret set CLOUDFLARE_API_TOKEN
gh secret set CLOUDFLARE_ACCOUNT_ID
```

---

## 2. AWS Lightsail (Backend)

### Create Instance
```bash
# Via AWS CLI
aws lightsail create-instances \
  --instance-names aragora-server \
  --availability-zone us-east-1a \
  --blueprint-id ubuntu_22_04 \
  --bundle-id small_3_0

# Or via Console: https://lightsail.aws.amazon.com
# 1. Click "Create instance"
# 2. Select Ubuntu 22.04
# 3. Choose $5-10/month plan (1-2GB RAM)
# 4. Name it "aragora-server"
# 5. Create and download SSH key
```

### Setup Server
```bash
# SSH into server
ssh -i your-key.pem ubuntu@<LIGHTSAIL_IP>

# Run setup script
git clone https://github.com/an0mium/aragora.git
cd aragora
chmod +x deploy/lightsail-setup.sh
sudo ./deploy/lightsail-setup.sh

# Set environment variables
sudo nano /etc/aragora/env
# Add: ANTHROPIC_API_KEY=sk-ant-...
# Add: OPENAI_API_KEY=sk-...

# Start service
sudo systemctl start aragora
```

### Set GitHub Secrets
```bash
# Set SSH key (paste private key contents)
gh secret set LIGHTSAIL_SSH_KEY < ~/.ssh/lightsail.pem

# Set host IP
gh secret set LIGHTSAIL_HOST --body "1.2.3.4"
```

---

## 3. AWS EC2 (Backend Alternative)

### Create Instance
```bash
# Via AWS CLI
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.small \
  --key-name aragora-key \
  --security-group-ids sg-xxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aragora}]'

# Or via Console: https://console.aws.amazon.com/ec2
# 1. Launch instance
# 2. Ubuntu 22.04 AMI
# 3. t3.small or larger
# 4. Create/select key pair
# 5. Allow ports 22, 80, 443, 8080
```

### Setup Server
```bash
# SSH into server
ssh -i your-key.pem ubuntu@<EC2_IP>

# Run setup (same as Lightsail)
git clone https://github.com/an0mium/aragora.git
cd aragora
chmod +x deploy/lightsail-setup.sh
sudo ./deploy/lightsail-setup.sh

# Configure and start
sudo nano /etc/aragora/env
sudo systemctl start aragora
```

### Set GitHub Secrets
```bash
gh secret set EC2_SSH_KEY < ~/.ssh/ec2.pem
gh secret set EC2_HOST --body "ec2-xx-xx-xx-xx.compute-1.amazonaws.com"
```

---

## 4. Trigger Deployment

Once secrets are configured:

```bash
# Push to main triggers auto-deploy
git push origin main

# Or manually trigger
gh workflow run deploy.yml

# Or deploy specific target
gh workflow run deploy.yml -f environment=cloudflare
gh workflow run deploy.yml -f environment=lightsail
gh workflow run deploy.yml -f environment=ec2
```

---

## Verify Deployment

```bash
# Check Cloudflare
curl https://aragora.pages.dev

# Check Lightsail/EC2
curl http://<SERVER_IP>:8080/api/health
curl http://<SERVER_IP>:8080/api/v1/health
```

---

## Environment Variables (Server)

Create `/etc/aragora/env`:
```bash
# Required (at least one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional
GEMINI_API_KEY=...
XAI_API_KEY=...
OPENROUTER_API_KEY=...

# Security
ARAGORA_API_TOKEN=your-secure-token
ARAGORA_ALLOWED_ORIGINS=https://aragora.pages.dev,https://yourdomain.com

# Performance
ARAGORA_MAX_CONCURRENT_DEBATES=5
```
