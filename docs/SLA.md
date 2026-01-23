# Aragora Service Level Agreement (SLA)

**Version:** 1.0.0
**Effective Date:** January 2026
**Last Updated:** January 14, 2026

---

## Overview

This Service Level Agreement (SLA) defines the service commitments for Aragora's multi-agent deliberation control plane. It establishes measurable targets for availability, performance, and support response times.

---

## Service Tiers

### Free Tier

| Metric | Target |
|--------|--------|
| **Availability** | Best effort |
| **Support** | Community forums |
| **Rate Limits** | 100 deliberations/day |
| **Data Retention** | 30 days |
| **SLA Credits** | None |

### Pro Tier ($99/month)

| Metric | Target |
|--------|--------|
| **Availability** | 99.5% monthly uptime |
| **Support Response** | 24 hours (business days) |
| **Rate Limits** | 1,000 deliberations/day |
| **Data Retention** | 1 year |
| **SLA Credits** | Pro-rated refund |

### Enterprise Tier (Custom Pricing)

| Metric | Target |
|--------|--------|
| **Availability** | 99.9% monthly uptime |
| **Support Response** | 4 hours (24/7) |
| **Rate Limits** | Unlimited |
| **Data Retention** | Custom (up to 7 years) |
| **SLA Credits** | Per contract |
| **Dedicated Support** | Named account manager |

---

## Availability Calculation

### Definition

**Availability** = (Total Minutes - Downtime Minutes) / Total Minutes × 100%

### Exclusions

The following are excluded from downtime calculations:

1. **Scheduled Maintenance**: Pre-announced maintenance windows (minimum 72 hours notice)
2. **Emergency Maintenance**: Critical security patches (minimum 4 hours notice when possible)
3. **Customer-Caused Issues**: Problems resulting from customer code, configurations, or API misuse
4. **Third-Party Failures**: Outages in upstream AI providers (Anthropic, OpenAI, etc.)
5. **Force Majeure**: Natural disasters, wars, government actions

### Monthly Uptime Targets

| Tier | Target | Max Downtime/Month |
|------|--------|-------------------|
| Free | Best effort | N/A |
| Pro | 99.5% | 3 hours 36 minutes |
| Enterprise | 99.9% | 43 minutes |

---

## Performance Targets

### API Response Times

| Endpoint Category | p50 | p95 | p99 |
|------------------|-----|-----|-----|
| Health/Status | 5ms | 20ms | 50ms |
| Authentication | 50ms | 150ms | 300ms |
| Debate Initiation | 200ms | 500ms | 1000ms |
| Audit Log Query | 20ms | 50ms | 100ms |
| Usage Summary | 20ms | 50ms | 100ms |

### Debate Completion Times

| Debate Type | Expected Duration |
|------------|-------------------|
| 3-round, 3 agents | 30-90 seconds |
| 5-round, 5 agents | 60-180 seconds |
| Complex consensus | 2-5 minutes |

*Note: Debate times depend on upstream AI provider response times and are not guaranteed.*

### Throughput Targets

| Tier | Concurrent Debates | Debates/Minute |
|------|-------------------|----------------|
| Free | 1 | 2 |
| Pro | 5 | 10 |
| Enterprise | 50+ | 100+ |

---

## Support Response Times

### Severity Definitions

| Severity | Definition | Examples |
|----------|------------|----------|
| **Critical (P1)** | Service completely unavailable | API returns 5xx for all requests |
| **High (P2)** | Major feature unavailable | Debate consensus failing |
| **Medium (P3)** | Feature degraded | Slow response times |
| **Low (P4)** | Minor issue | Documentation error |

### Response Time Targets

| Severity | Free | Pro | Enterprise |
|----------|------|-----|------------|
| Critical (P1) | Best effort | 4 hours | 1 hour |
| High (P2) | Best effort | 8 hours | 2 hours |
| Medium (P3) | Best effort | 24 hours | 4 hours |
| Low (P4) | Best effort | 72 hours | 24 hours |

### Resolution Time Targets

| Severity | Pro | Enterprise |
|----------|-----|------------|
| Critical (P1) | 24 hours | 4 hours |
| High (P2) | 72 hours | 24 hours |
| Medium (P3) | 1 week | 72 hours |
| Low (P4) | Best effort | 1 week |

---

## SLA Credits

### Eligibility

SLA credits are available to Pro and Enterprise tier customers when:

1. Monthly availability falls below the committed target
2. Customer reports the issue within 30 days
3. Issue is not excluded per the exclusions list above

### Credit Calculation

| Availability | Credit (% of monthly fee) |
|--------------|--------------------------|
| 99.0% - 99.5% | 10% |
| 98.0% - 99.0% | 25% |
| 95.0% - 98.0% | 50% |
| < 95.0% | 100% |

### Credit Limitations

- Maximum credit per month: 100% of monthly fee
- Credits are applied to future invoices only
- Credits do not accumulate across months
- Credits are not transferable or redeemable for cash

---

## Data Protection

### Backup Frequency

| Data Type | Backup Frequency | Retention |
|-----------|-----------------|-----------|
| Debate Transcripts | Real-time | Per tier |
| Audit Logs | Real-time | 7 years |
| User Data | Daily | Per tier |
| Configuration | Daily | 90 days |

### Recovery Time Objectives

| Tier | RTO (Recovery Time) | RPO (Data Loss) |
|------|---------------------|-----------------|
| Free | 24 hours | 24 hours |
| Pro | 4 hours | 1 hour |
| Enterprise | 1 hour | 15 minutes |

---

## Security Commitments

### Certifications

| Certification | Status |
|--------------|--------|
| SOC 2 Type II | In Progress |
| ISO 27001 | Planned |
| HIPAA BAA | Available (Enterprise) |
| GDPR Compliance | Yes |
| PCI-DSS | Not applicable |

### Incident Response

| Phase | Target |
|-------|--------|
| Detection | < 15 minutes |
| Acknowledgment | < 1 hour |
| Initial Response | < 4 hours |
| Root Cause Analysis | < 72 hours |
| Post-Incident Report | < 7 days |

---

## Maintenance Windows

### Scheduled Maintenance

- **Frequency**: Monthly (last Sunday, 02:00-06:00 UTC)
- **Notification**: 72 hours advance notice
- **Duration**: Maximum 4 hours

### Emergency Maintenance

- **Notification**: As soon as possible (target 4 hours)
- **Communication**: Status page + email to affected customers

---

## Monitoring and Reporting

### Status Page

Real-time service status is available at: `status.aragora.ai`

Components monitored:
- API Gateway
- Debate Engine
- Authentication Services
- Database Cluster
- AI Provider Connectivity

### Monthly Reports

Enterprise customers receive monthly reports including:
- Availability metrics
- Performance percentiles
- Incident summaries
- Usage statistics

---

## Escalation Path

### Pro Tier

1. **Level 1**: Support ticket (support@aragora.ai)
2. **Level 2**: Senior engineer (escalation within 4 hours)
3. **Level 3**: Engineering leadership (escalation within 24 hours)

### Enterprise Tier

1. **Level 1**: Dedicated support engineer
2. **Level 2**: Account manager + engineering lead
3. **Level 3**: VP Engineering (escalation within 2 hours)
4. **Level 4**: Executive escalation (CEO/CTO)

---

## Contact Information

| Channel | Details |
|---------|---------|
| Support Email | support@aragora.ai |
| Status Page | status.aragora.ai |
| Security Issues | security@aragora.ai |
| Sales | sales@aragora.ai |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |

---

*This SLA is subject to the Aragora Terms of Service. In case of conflict, the Terms of Service prevail.*
