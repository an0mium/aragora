---
title: Incident Communication Procedures
description: Incident Communication Procedures
---

# Incident Communication Procedures

**SOC 2 Controls:** CC2.2, CC2.3, CC7.4, CC7.5

This document outlines the communication procedures for service incidents affecting Aragora customers.

## Communication Channels

| Channel | Purpose | Audience |
|---------|---------|----------|
| status.aragora.ai | Real-time status updates | All users (public) |
| In-app banner | Active incident notification | Logged-in users |
| Email | Major incident notifications | Affected customers |
| Twitter @aragora_status | Quick updates | Public followers |
| Slack #ops-alerts | Internal coordination | Ops team |
| PagerDuty | Critical escalation | On-call engineer |

## Incident Severity Levels

### P1 - Critical
- **Definition:** Complete service outage or data breach
- **Response Time:** 15 minutes
- **Update Frequency:** Every 15 minutes during active incident
- **Notification:** Status page + Email + In-app banner + Twitter

### P2 - High
- **Definition:** Major feature unavailable, significant performance degradation
- **Response Time:** 30 minutes
- **Update Frequency:** Every 30 minutes during active incident
- **Notification:** Status page + In-app banner

### P3 - Medium
- **Definition:** Minor feature degradation, non-critical component failure
- **Response Time:** 2 hours
- **Update Frequency:** Every 2 hours during active incident
- **Notification:** Status page only

### P4 - Low
- **Definition:** Cosmetic issues, minor bugs with workarounds
- **Response Time:** Next business day
- **Notification:** None (tracked internally)

## Status Page Update Templates

### Investigating

```
Title: Investigating [Service Name] Issues
Status: Investigating
Body:
We are currently investigating reports of [brief description of symptoms].
Our team is working to identify the cause and restore normal service.

We will provide updates every [frequency] until this issue is resolved.

Affected services: [list affected services]
Started at: [time] UTC
```

### Identified

```
Title: [Service Name] Issue Identified
Status: Identified
Body:
We have identified the cause of the [service] issues reported earlier.

Root cause: [brief technical description appropriate for customers]

Our team is implementing a fix and we expect service to be restored by
[estimated time] UTC.

Current impact: [describe current user impact]
Started at: [time] UTC
```

### Monitoring

```
Title: [Service Name] Fix Deployed - Monitoring
Status: Monitoring
Body:
A fix has been deployed for the [service] issues.

We are actively monitoring the system to ensure stability. All services
should be operating normally.

If you continue to experience issues, please contact support@aragora.ai.

Impact duration: [duration]
```

### Resolved

```
Title: [Service Name] Issue Resolved
Status: Resolved
Body:
The [service] issue has been fully resolved. All services are operating
normally.

Summary:
- Duration: [start time] to [end time] UTC ([total duration])
- Impact: [brief description of customer impact]
- Root cause: [brief technical description]
- Resolution: [what was done to fix it]

We apologize for any inconvenience this may have caused. A detailed
post-incident review will be conducted.

For questions, contact support@aragora.ai.
```

### Scheduled Maintenance

```
Title: Scheduled Maintenance - [Date]
Status: Scheduled
Body:
We will be performing scheduled maintenance on [date] from [start time]
to [end time] UTC.

During this window, the following services may be briefly unavailable:
- [Service 1]
- [Service 2]

Expected impact: [description]
Duration: [estimated duration]

We recommend completing any active work before the maintenance window.
No action is required from customers.
```

## Email Templates

### Major Incident Notification

```
Subject: [Aragora] Service Incident - [Brief Description]

Dear [Customer Name],

We are experiencing an issue affecting [service/feature] that may impact
your ability to [affected functionality].

Current Status: [Investigating/Identified/Monitoring]
Started: [time] UTC
Estimated Resolution: [time if known, or "We are working to resolve this
as quickly as possible"]

What this means for you:
- [Specific impact on this customer's usage]
- [Workarounds if available]

We are actively working on a resolution and will keep you updated. You
can monitor real-time status at https://status.aragora.ai

If you have urgent questions, please contact support@aragora.ai or your
account manager.

We apologize for any inconvenience.

Aragora Operations Team
```

### Incident Resolution Notification

```
Subject: [Aragora] Incident Resolved - [Brief Description]

Dear [Customer Name],

The service incident affecting [service/feature] has been resolved.

Incident Summary:
- Duration: [start] to [end] UTC
- Total downtime: [duration]
- Services affected: [list]

Root Cause:
[Brief, customer-appropriate explanation]

Resolution:
[What was done to fix it]

Preventive Measures:
[Steps being taken to prevent recurrence]

We sincerely apologize for any disruption to your operations. If you
have questions or experienced issues not yet resolved, please contact
support@aragora.ai.

A detailed post-incident report is available upon request for Enterprise
customers.

Aragora Operations Team
```

## Communication Workflow

### Step 1: Initial Detection (0-5 minutes)
1. Alert received via monitoring or customer report
2. On-call engineer acknowledges alert in PagerDuty
3. Assess severity level (P1-P4)
4. Create incident channel in Slack (#incident-YYYY-MM-DD-N)

### Step 2: Initial Communication (5-15 minutes)
1. Post "Investigating" update to status page
2. For P1/P2: Enable in-app incident banner
3. For P1: Draft customer email notification
4. Assign communication owner (separate from technical responder)

### Step 3: Ongoing Updates
1. Maintain update frequency based on severity
2. Communication owner posts all external updates
3. Technical updates flow through communication owner
4. Never speculate on timelines without technical confirmation

### Step 4: Resolution
1. Confirm fix is deployed and stable (minimum 10 min monitoring)
2. Post "Resolved" update to status page
3. Disable in-app banner
4. Send resolution email to affected customers (P1/P2)
5. Post resolution to Twitter (P1 only)

### Step 5: Post-Incident
1. Schedule post-incident review within 48 hours
2. Document timeline, root cause, and action items
3. Share lessons learned with team
4. Update runbooks if needed
5. Send post-incident report to requesting Enterprise customers

## Escalation Contacts

| Role | Primary | Backup |
|------|---------|--------|
| On-Call Engineer | PagerDuty rotation | #ops-alerts |
| Incident Commander | [Name] | [Backup Name] |
| Communications Lead | [Name] | [Backup Name] |
| Executive Sponsor | [Name] | [Backup Name] |

## Customer SLA Considerations

| Tier | SLA | Credit Policy |
|------|-----|---------------|
| Enterprise | 99.9% uptime | Service credits per SLA.md |
| Professional | 99.5% uptime | Service credits per SLA.md |
| Starter | 99% uptime | Best effort |

See [SLA.md](../enterprise/sla) for detailed service credit calculations.

## Audit Trail

All incident communications are logged:
- Status page history: Uptime Kuma built-in
- Email: Retained in ops mailbox for 7 years
- Slack: Retained per workspace policy
- PagerDuty: Incident timeline retained

## Review Cadence

- **Weekly:** Review open incidents and communication effectiveness
- **Monthly:** Analyze incident trends and communication metrics
- **Quarterly:** Update templates and procedures
- **Annually:** Full communication procedure review

---

*Last updated: January 2026*
*Document owner: Operations Team*
*Review cycle: Quarterly*
