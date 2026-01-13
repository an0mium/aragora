# Stripe Payment Setup Guide

This guide walks you through connecting Stripe to receive payments for Aragora subscriptions.

## Prerequisites

- Company bank account (US checking account)
- Business information (EIN, address, beneficial owners)
- Domain verified (aragora.ai)
- Production server running (api.aragora.ai)

---

## Step 1: Create Stripe Account

1. Go to https://dashboard.stripe.com/register
2. Sign up with your business email
3. Select "Business" account type

## Step 2: Complete Business Verification

In Stripe Dashboard → **Settings** → **Business settings**:

### 2.1 Business Details
- Legal business name
- Business address
- Tax ID (EIN for US companies)
- Industry: "Software / SaaS"
- Website: https://aragora.ai

### 2.2 Beneficial Owners
- Add anyone owning 25%+ of the company
- Provide: Name, DOB, SSN (last 4), address

### 2.3 Representative
- Person authorized to manage the Stripe account
- Usually the founder/CEO

**Verification typically takes 1-2 business days.**

---

## Step 3: Connect Bank Account

In Stripe Dashboard → **Settings** → **Payouts** → **Add bank account**:

1. Enter routing number (9 digits)
2. Enter account number
3. Select "Checking" account type
4. Stripe will make 2 micro-deposits (1-2 days)
5. Return to verify the exact amounts

Once verified, payouts are automatic (2 business day rolling basis).

---

## Step 4: Get API Keys

In Stripe Dashboard → **Developers** → **API keys**:

| Key | Example | Use |
|-----|---------|-----|
| Publishable key | `pk_live_51...` | Frontend (checkout) |
| Secret key | `sk_live_51...` | Backend (API calls) |

**Important:** Use `pk_live_` and `sk_live_` keys for production, NOT `pk_test_`/`sk_test_`.

---

## Step 5: Create Products & Prices

In Stripe Dashboard → **Products** → **Add product**:

### Starter Plan
- Name: "Aragora Starter"
- Description: "100 debates/month, 3 agents, basic analytics"
- Price: $29.00 / month (recurring)
- Copy the Price ID: `price_...`

### Professional Plan
- Name: "Aragora Professional"
- Description: "1000 debates/month, unlimited agents, API access"
- Price: $99.00 / month (recurring)
- Copy the Price ID: `price_...`

### Enterprise Plan
- Name: "Aragora Enterprise"
- Description: "Unlimited debates, priority support, custom agents"
- Price: $299.00 / month (recurring)
- Copy the Price ID: `price_...`

---

## Step 6: Register Webhook Endpoint

In Stripe Dashboard → **Developers** → **Webhooks** → **Add endpoint**:

### Endpoint Configuration
- **URL:** `https://api.aragora.ai/api/billing/webhook`
- **Description:** "Aragora production webhook"
- **Events to send:**
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_succeeded`
  - `invoice.payment_failed`

### Get Signing Secret
After creating the endpoint:
1. Click on the endpoint
2. Click "Reveal" under Signing secret
3. Copy the `whsec_...` value

---

## Step 7: Configure Environment

Run the setup script:

```bash
./scripts/setup_stripe.sh
```

Or manually add to `/etc/aragora/.env`:

```bash
# Stripe Live Keys
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Product Price IDs
STRIPE_PRICE_STARTER=price_...
STRIPE_PRICE_PROFESSIONAL=price_...
STRIPE_PRICE_ENTERPRISE=price_...
```

---

## Step 8: Deploy & Restart

```bash
# SSH to production server
ssh ubuntu@api.aragora.ai

# Pull latest code
cd /opt/aragora
git pull origin main

# Restart service
sudo systemctl restart aragora

# Verify service is running
sudo systemctl status aragora
```

---

## Step 9: Verify Setup

Run the verification script:

```bash
./scripts/verify_stripe.py
```

Or manually test:

### 9.1 Test Webhook Connectivity
In Stripe Dashboard → Webhooks → Your endpoint → "Send test webhook"

Select `checkout.session.completed` and send.

Check server logs:
```bash
sudo journalctl -u aragora -f | grep -i stripe
```

### 9.2 Test Real Transaction
1. Go to https://aragora.ai
2. Register a new account
3. Subscribe to Starter plan ($29)
4. Use a real card (your own)
5. Verify:
   - Payment appears in Stripe Dashboard
   - User's org shows tier=STARTER
   - Webhook logs show checkout.session.completed

### 9.3 Verify Payout
After 2 business days, check:
- Stripe Dashboard → Balances → Payouts
- Your bank account for the deposit

---

## Payout Schedule

Default: **2 business days** (US)

To change: Stripe Dashboard → Settings → Payouts → Payout schedule

Options:
- Daily (automatic)
- Weekly (choose day)
- Monthly (choose date)
- Manual (you trigger payouts)

---

## Troubleshooting

### Webhook not receiving events
1. Check endpoint URL is correct (https, not http)
2. Verify server is accessible from internet
3. Check firewall allows inbound 443
4. Review webhook logs in Stripe Dashboard

### Payments not settling
1. Verify bank account is verified (green checkmark)
2. Check for holds in Stripe Dashboard → Balances
3. Review any risk flags in Dashboard → Radar

### Subscription not upgrading tier
1. Check webhook endpoint is receiving events
2. Verify `STRIPE_WEBHOOK_SECRET` is correct
3. Check server logs for errors:
   ```bash
   sudo journalctl -u aragora | grep -i "webhook\|billing"
   ```

---

## Security Checklist

- [ ] Using `sk_live_` keys (not test keys)
- [ ] Webhook secret is set correctly
- [ ] HTTPS only (no HTTP)
- [ ] API keys not committed to git
- [ ] Environment file has restricted permissions (600)
- [ ] Webhook signature verification enabled

---

## Support

- Stripe Documentation: https://stripe.com/docs
- Stripe Support: https://support.stripe.com
- Aragora Issues: https://github.com/aragora/aragora/issues
