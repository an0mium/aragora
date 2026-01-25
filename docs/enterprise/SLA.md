# Aragora Service Level Agreement (SLA)

**Effective Date:** January 2026
**Version:** 1.0

---

## 1. Service Commitment

Aragora commits to providing **99.9% monthly uptime** for the Aragora Platform ("Service") to customers on Enterprise plans.

### 1.1 Uptime Calculation

```
Monthly Uptime % = ((Total Minutes - Downtime Minutes) / Total Minutes) × 100
```

- **Total Minutes**: Total minutes in the calendar month
- **Downtime Minutes**: Minutes where the Service is unavailable (excluding Scheduled Maintenance and Exclusions)

### 1.2 Service Availability Tiers

| Tier | Uptime SLA | Max Monthly Downtime | Service Credits |
|------|------------|---------------------|-----------------|
| Enterprise | 99.9% | 43.8 minutes | Yes |
| Business | 99.5% | 3.65 hours | Limited |
| Standard | 99.0% | 7.3 hours | No |

---

## 2. Service Level Objectives (SLOs)

### 2.1 API Response Time

| Endpoint Category | P50 Latency | P99 Latency | Error Rate |
|-------------------|-------------|-------------|------------|
| Health/Status | < 50ms | < 200ms | < 0.01% |
| Read Operations | < 100ms | < 500ms | < 0.1% |
| Write Operations | < 200ms | < 1000ms | < 0.1% |
| Debate Creation | < 500ms | < 2000ms | < 0.5% |
| Streaming (WS) | < 100ms | < 500ms | < 0.1% |

### 2.2 Throughput Guarantees

| Resource | Guaranteed Capacity |
|----------|---------------------|
| API Requests | 10,000 req/min per tenant |
| Concurrent Debates | 100 per tenant |
| WebSocket Connections | 1,000 per tenant |
| Storage | 100 GB per tenant |

### 2.3 Data Durability

- **Durability**: 99.999999999% (11 nines) for stored data
- **Backup Frequency**: Every 6 hours
- **Point-in-Time Recovery**: Up to 30 days
- **Cross-Region Replication**: Available for Enterprise+

---

## 3. Service Credits

### 3.1 Credit Schedule

| Monthly Uptime | Service Credit |
|----------------|----------------|
| 99.0% - 99.9% | 10% of monthly fee |
| 95.0% - 99.0% | 25% of monthly fee |
| 90.0% - 95.0% | 50% of monthly fee |
| < 90.0% | 100% of monthly fee |

### 3.2 Credit Request Process

1. Submit credit request within 30 days of incident
2. Include: Tenant ID, incident timestamps, affected services
3. Submit via: support@aragora.ai or support portal
4. Credits applied to next billing cycle (non-transferable)

### 3.3 Credit Limitations

- Maximum credit: 100% of monthly fee for affected service
- Credits do not apply to: one-time fees, professional services, third-party costs
- Credits are sole remedy for SLA breaches

---

## 4. Exclusions

The following are **not** counted as downtime:

### 4.1 Scheduled Maintenance

- Announced 72 hours in advance
- Maximum 4 hours per month
- Performed during low-usage windows (02:00-06:00 UTC)

### 4.2 Emergency Maintenance

- Security patches or critical fixes
- Announced as soon as reasonably possible
- Limited to 2 hours per incident

### 4.3 External Factors

- Customer's internet connectivity issues
- Third-party service failures (LLM providers, payment processors)
- DNS propagation delays
- Customer-side firewall/security blocks

### 4.4 Customer-Caused Issues

- Exceeding rate limits or quotas
- Misuse or abuse of the Service
- Customer code or integration errors
- Failure to follow documentation

### 4.5 Force Majeure

- Natural disasters
- Government actions
- War, terrorism, civil unrest
- Pandemic-related disruptions

---

## 5. Support Response Times

### 5.1 Support Tiers

| Priority | Description | Response Time | Resolution Target |
|----------|-------------|---------------|-------------------|
| P1 - Critical | Service unavailable | 15 minutes | 4 hours |
| P2 - High | Major feature degraded | 1 hour | 8 hours |
| P3 - Medium | Minor feature issue | 4 hours | 24 hours |
| P4 - Low | Questions, enhancements | 24 hours | Best effort |

### 5.2 Support Channels

| Channel | Availability | Best For |
|---------|--------------|----------|
| Email | 24/7 | All issues |
| Portal | 24/7 | Ticket tracking |
| Phone | Business hours | P1/P2 issues |
| Slack Connect | Business hours | Enterprise customers |

---

## 6. Disaster Recovery

### 6.1 Recovery Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| RTO (Recovery Time Objective) | 4 hours | Time to restore service |
| RPO (Recovery Point Objective) | 1 hour | Maximum data loss window |
| Failover Time | 15 minutes | Time to switch to backup region |

### 6.2 Backup Strategy

- **Hot Standby**: Active-passive replication to secondary region
- **Database**: Continuous WAL archiving with point-in-time recovery
- **Object Storage**: Cross-region replication enabled
- **Configuration**: Infrastructure-as-code, version controlled

### 6.3 DR Testing

- **Frequency**: Quarterly
- **Scope**: Full failover simulation
- **Documentation**: Results shared with Enterprise customers upon request

---

## 7. Data Residency & Compliance

### 7.1 Data Locations

| Region | Primary | Backup | Compliance |
|--------|---------|--------|------------|
| US | us-east-1 | us-west-2 | SOC 2, HIPAA |
| EU | eu-west-1 | eu-central-1 | GDPR, SOC 2 |
| APAC | ap-southeast-1 | ap-northeast-1 | SOC 2 |

### 7.2 Data Handling

- **Encryption at Rest**: AES-256-GCM
- **Encryption in Transit**: TLS 1.3
- **Key Management**: AWS KMS with customer-managed keys (Enterprise+)
- **Data Retention**: Configurable per tenant (default 90 days)

### 7.3 Compliance Certifications

- SOC 2 Type II (in progress)
- ISO 27001 (planned Q3 2026)
- GDPR compliant
- HIPAA compliant (with BAA)

---

## 8. Monitoring & Transparency

### 8.1 Status Page

- **URL**: status.aragora.ai
- **Updates**: Real-time incident notifications
- **History**: 90-day incident history
- **Subscriptions**: Email, SMS, webhook notifications

### 8.2 Metrics Dashboard

Enterprise customers have access to:
- Real-time SLO metrics
- Historical uptime data
- Latency percentiles
- Error rate trends
- Capacity utilization

### 8.3 Incident Communication

| Severity | Initial Update | Ongoing Updates | Post-Mortem |
|----------|----------------|-----------------|-------------|
| P1 | 15 minutes | Every 30 minutes | Within 72 hours |
| P2 | 1 hour | Every 2 hours | Within 1 week |
| P3 | 4 hours | Daily | Upon request |

---

## 9. Terms & Conditions

### 9.1 Agreement Duration

This SLA is effective for the duration of your Enterprise subscription agreement.

### 9.2 Modifications

Aragora may modify this SLA with 30 days' notice. Changes will not reduce service levels for existing customers during their current term.

### 9.3 Termination

If Aragora fails to meet the 99.9% uptime SLA for 3 consecutive months, customer may terminate with 30 days' notice and receive prorated refund.

### 9.4 Governing Law

This SLA is governed by the laws of the State of Delaware, USA.

---

## 10. Contact Information

- **Support Portal**: support.aragora.ai
- **Email**: support@aragora.ai
- **Emergency Hotline**: +1-XXX-XXX-XXXX (Enterprise only)
- **Status Page**: status.aragora.ai

---

*Last Updated: January 2026*
