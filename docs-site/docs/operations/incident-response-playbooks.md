---
title: Incident Response Playbooks
description: Incident Response Playbooks
---

# Incident Response Playbooks

**SOC 2 Control:** CC7.2

This document provides step-by-step playbooks for responding to common security incidents.

---

## Playbook Index

| ID | Playbook | Trigger |
|----|----------|---------|
| PB-001 | [Data Breach Response](#pb-001-data-breach-response) | Confirmed data exposure |
| PB-002 | [Account Compromise](#pb-002-account-compromise) | Unauthorized access detected |
| PB-003 | [Ransomware Attack](#pb-003-ransomware-attack) | Encryption detected |
| PB-004 | [DDoS Attack](#pb-004-ddos-attack) | Service degradation |
| PB-005 | [API Key Exposure](#pb-005-api-key-exposure) | Key found in public repo |
| PB-006 | [Phishing Attack](#pb-006-phishing-attack) | Successful phishing |
| PB-007 | [Insider Threat](#pb-007-insider-threat) | Suspicious employee activity |
| PB-008 | [Third-Party Breach](#pb-008-third-party-breach) | Vendor reports breach |

---

## PB-001: Data Breach Response

### Severity: Critical (P1)

### Trigger Conditions
- Customer data exposed externally
- PII or credentials leaked
- Database dump discovered

### Immediate Actions (0-15 minutes)

```
[ ] 1. DECLARE INCIDENT
    - Page incident commander
    - Create incident channel: #inc-YYYYMMDD-breach
    - Start incident log

[ ] 2. ASSESS SCOPE
    - What data was exposed?
    - How many users affected?
    - Is exposure ongoing?

[ ] 3. CONTAIN
    - Revoke compromised credentials
    - Block unauthorized access paths
    - Isolate affected systems if needed
```

### Short-Term Actions (15 min - 4 hours)

```
[ ] 4. EVIDENCE PRESERVATION
    - Snapshot affected systems
    - Export relevant logs
    - Document timeline

[ ] 5. IMPACT ANALYSIS
    - Identify all affected records
    - Classify data types exposed
    - Determine exposure duration

[ ] 6. STAKEHOLDER NOTIFICATION
    - Legal team (within 1 hour)
    - Executive team (within 2 hours)
    - Prepare customer notification draft
```

### Recovery Actions (4-72 hours)

```
[ ] 7. REMEDIATION
    - Fix vulnerability that caused breach
    - Reset affected credentials
    - Implement additional controls

[ ] 8. CUSTOMER NOTIFICATION
    - Send within 72 hours (GDPR requirement)
    - Include: what happened, what data, what we're doing
    - Provide support contact

[ ] 9. REGULATORY NOTIFICATION
    - GDPR: 72 hours to supervisory authority
    - Other regulations as applicable
```

### Post-Incident

```
[ ] 10. POST-INCIDENT REVIEW (within 5 days)
     - Root cause analysis
     - Timeline documentation
     - Lessons learned
     - Control improvements

[ ] 11. DOCUMENTATION
     - Complete incident report
     - Update risk register
     - File audit evidence
```

### Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| Incident Commander | On-call rotation | Overall coordination |
| Legal | legal@aragora.ai | Regulatory compliance |
| Comms | comms@aragora.ai | Customer notification |
| Executive | CEO direct line | Business decisions |

---

## PB-002: Account Compromise

### Severity: High (P2)

### Trigger Conditions
- Impossible travel detected
- Login from unknown device/location
- User reports unauthorized activity
- Credential stuffing detected

### Immediate Actions (0-15 minutes)

```
[ ] 1. CONTAIN
    - Suspend affected account
    - Terminate all active sessions
    - Block source IP if known

[ ] 2. ASSESS
    - What actions were taken?
    - What data was accessed?
    - Are other accounts affected?

[ ] 3. NOTIFY USER
    - If legitimate user: require password reset + MFA
    - If attacker: preserve evidence
```

### Investigation (15 min - 2 hours)

```
[ ] 4. AUDIT TRAIL REVIEW
    - Last 7 days of activity
    - Data access patterns
    - Configuration changes

[ ] 5. LATERAL MOVEMENT CHECK
    - Related accounts
    - Shared resources
    - API key usage

[ ] 6. CREDENTIAL ANALYSIS
    - Was password reused?
    - Is password in known breaches?
    - MFA status at time of compromise
```

### Recovery (2-24 hours)

```
[ ] 7. CREDENTIAL RESET
    - Force password change
    - Rotate API keys
    - Revoke OAuth tokens

[ ] 8. ACCESS RESTORATION
    - Verify user identity
    - Enable MFA if not active
    - Restore appropriate access

[ ] 9. MONITORING
    - Enhanced logging for 30 days
    - Alert on suspicious patterns
```

### Root Cause Checklist

- [ ] Phishing attack?
- [ ] Credential reuse from other breach?
- [ ] Weak password?
- [ ] Malware on user device?
- [ ] Session hijacking?
- [ ] MFA bypass?

---

## PB-003: Ransomware Attack

### Severity: Critical (P1)

### Trigger Conditions
- Files being encrypted
- Ransom note displayed
- Systems becoming unresponsive

### Immediate Actions (0-5 minutes)

```
!!! DO NOT PAY RANSOM WITHOUT EXECUTIVE/LEGAL APPROVAL !!!

[ ] 1. ISOLATE
    - Disconnect affected systems from network
    - Do NOT power off (preserve memory)
    - Block lateral movement paths

[ ] 2. PAGE
    - All incident responders
    - Executive team
    - Legal counsel

[ ] 3. ASSESS SPREAD
    - What systems are affected?
    - Is encryption still spreading?
    - Are backups safe?
```

### Containment (5-60 minutes)

```
[ ] 4. NETWORK ISOLATION
    - Segment infected zone
    - Block C2 communications
    - Preserve network logs

[ ] 5. BACKUP VERIFICATION
    - Check backup integrity
    - Verify backups not compromised
    - Isolate backup systems

[ ] 6. EVIDENCE COLLECTION
    - Memory dump if possible
    - Ransom note copy
    - Network traffic capture
```

### Recovery Path Decision

```
DECISION TREE:

1. Are clean backups available?
   YES → Proceed to backup restoration
   NO → Evaluate decryption options

2. Is decryptor available?
   Check: nomoreransom.org
   YES → Test on non-critical system first
   NO → Evaluate ransom payment (last resort)

3. Ransom payment consideration:
   - Legal review required
   - Executive approval required
   - No guarantee of decryption
   - May fund criminal activity
```

### Recovery (hours to days)

```
[ ] 7. CLEAN RESTORATION
    - Build systems from clean images
    - Restore data from verified backups
    - Do NOT reconnect to network until verified clean

[ ] 8. SECURITY IMPROVEMENTS
    - Patch vulnerability used for entry
    - Implement additional endpoint protection
    - Review and improve backup strategy

[ ] 9. BUSINESS CONTINUITY
    - Prioritize critical systems
    - Communicate status to stakeholders
    - Document downtime impact
```

---

## PB-004: DDoS Attack

### Severity: High (P2)

### Trigger Conditions
- Sudden traffic spike
- Service degradation
- Upstream provider notification

### Immediate Actions (0-5 minutes)

```
[ ] 1. CONFIRM ATTACK
    - Check traffic patterns
    - Verify not legitimate spike
    - Identify attack type (volumetric, protocol, app layer)

[ ] 2. ENGAGE MITIGATION
    - Enable DDoS protection (Cloudflare/AWS Shield)
    - Rate limit aggressive IPs
    - Scale infrastructure if possible

[ ] 3. COMMUNICATE
    - Update status page
    - Notify support team
    - Alert affected customers
```

### Mitigation (5-60 minutes)

```
[ ] 4. TRAFFIC ANALYSIS
    - Identify attack signatures
    - Determine source (botnet, amplification)
    - Block malicious patterns

[ ] 5. FILTER RULES
    - Geographic filtering if applicable
    - Rate limiting
    - Challenge suspicious traffic

[ ] 6. UPSTREAM COORDINATION
    - Contact ISP if needed
    - Engage CDN support
    - Consider null routing as last resort
```

### Recovery

```
[ ] 7. VERIFY NORMAL OPERATION
    - Monitor traffic patterns
    - Check all endpoints
    - Verify no data exfiltration (DDoS as distraction)

[ ] 8. STRENGTHEN DEFENSES
    - Review DDoS protection config
    - Update rate limits
    - Improve traffic analysis
```

---

## PB-005: API Key Exposure

### Severity: High (P2)

### Trigger Conditions
- Key found in public GitHub repo
- Key detected by scanning service
- Key reported by security researcher

### Immediate Actions (0-5 minutes)

```
[ ] 1. ROTATE KEY IMMEDIATELY
    - Do not wait for investigation
    - Generate new key
    - Update deployed services

[ ] 2. REVOKE OLD KEY
    - Mark as compromised
    - Log revocation time

[ ] 3. ASSESS EXPOSURE
    - When was key exposed?
    - What access did it provide?
    - Was it used maliciously?
```

### Investigation (5-60 minutes)

```
[ ] 4. AUDIT KEY USAGE
    - Review all API calls with this key
    - Identify suspicious activity
    - Check for data exfiltration

[ ] 5. SOURCE IDENTIFICATION
    - How did key get exposed?
    - Developer mistake?
    - Build system leak?
    - Third-party compromise?

[ ] 6. IMPACT ASSESSMENT
    - What resources were accessible?
    - Was any action taken?
    - Customer data affected?
```

### Remediation

```
[ ] 7. FIX EXPOSURE SOURCE
    - Remove from repository history (BFG Repo-Cleaner)
    - Update .gitignore
    - Add pre-commit hooks

[ ] 8. PROCESS IMPROVEMENTS
    - Secret scanning in CI/CD
    - Developer training
    - Secret management review
```

### Prevention Checklist

- [ ] Pre-commit hooks for secret detection
- [ ] CI/CD secret scanning
- [ ] Regular key rotation policy
- [ ] Minimal-privilege key scopes
- [ ] Secret management tooling

---

## PB-006: Phishing Attack

### Severity: Medium-High (P2/P3)

### Trigger Conditions
- Employee clicked phishing link
- Credentials entered on fake site
- Malicious attachment opened

### Immediate Actions (0-15 minutes)

```
[ ] 1. CREDENTIAL RESET
    - Force password change immediately
    - Revoke all sessions
    - Enable/verify MFA

[ ] 2. DEVICE CHECK
    - Scan for malware
    - Check for unusual processes
    - Review browser history

[ ] 3. ASSESS SCOPE
    - What credentials were entered?
    - What access does this user have?
    - Were others targeted?
```

### Investigation

```
[ ] 4. PHISHING ANALYSIS
    - Capture phishing page (screenshot, source)
    - Identify indicators of compromise
    - Report to phishing databases

[ ] 5. RELATED ACTIVITY CHECK
    - Search logs for phishing domain
    - Check if others visited
    - Look for lateral movement

[ ] 6. CREDENTIAL ABUSE CHECK
    - Monitor for suspicious logins
    - Check OAuth authorizations
    - Review API key usage
```

### Remediation

```
[ ] 7. BLOCK INDICATORS
    - Block phishing domain
    - Block sender email
    - Update email filters

[ ] 8. USER SUPPORT
    - Additional training
    - No-blame discussion
    - Credit monitoring if PII exposed

[ ] 9. ORGANIZATION-WIDE
    - Send awareness reminder
    - Consider targeted training
    - Update phishing simulations
```

---

## PB-007: Insider Threat

### Severity: High (P2)

### Trigger Conditions
- Unusual data access patterns
- Mass data download
- Unauthorized configuration changes
- Tip from colleague

### Immediate Actions (0-15 minutes)

```
!!! HANDLE WITH EXTREME CONFIDENTIALITY !!!

[ ] 1. ASSESS THREAT LEVEL
    - Immediate risk to systems?
    - Data exfiltration in progress?
    - Is person on premises?

[ ] 2. LIMITED NOTIFICATION
    - Legal counsel
    - HR representative
    - Executive sponsor
    - DO NOT alert subject

[ ] 3. EVIDENCE PRESERVATION
    - Export audit logs
    - Document observed behavior
    - Preserve chain of custody
```

### Investigation

```
[ ] 4. ACTIVITY AUDIT
    - Access logs (last 90 days)
    - Data downloads
    - External transfers
    - Off-hours activity

[ ] 5. DEVICE ANALYSIS
    - With HR/Legal approval
    - USB device usage
    - Cloud sync activity
    - Email attachments

[ ] 6. MOTIVATION ASSESSMENT
    - Financial stress?
    - Notice given?
    - Grievances?
    - External pressure?
```

### Response Options

```
DECISION MATRIX:

If immediate threat:
→ Disable access, involve security

If investigation needed:
→ Enhanced monitoring, document evidence

If departure imminent:
→ Accelerate exit, secure access

If criminal activity:
→ Engage law enforcement, preserve evidence
```

### Important Notes

- Involve HR and Legal from the start
- Document everything
- Maintain confidentiality
- Avoid tipping off subject
- Consider defamation risks

---

## PB-008: Third-Party Breach

### Severity: Varies (P2-P3)

### Trigger Conditions
- Vendor notifies of breach
- Public disclosure
- Shared credentials compromised

### Immediate Actions (0-30 minutes)

```
[ ] 1. ASSESS RELATIONSHIP
    - What data do they have?
    - What access do they have?
    - What systems are integrated?

[ ] 2. ROTATE CREDENTIALS
    - API keys
    - OAuth tokens
    - Shared accounts

[ ] 3. REVIEW ACCESS LOGS
    - Vendor's access patterns
    - Any unusual activity?
    - Timeline overlap with breach
```

### Investigation

```
[ ] 4. VENDOR COMMUNICATION
    - Request incident report
    - Understand scope of breach
    - Get timeline of events

[ ] 5. DATA IMPACT ANALYSIS
    - What Aragora data was exposed?
    - Customer data affected?
    - System access compromised?

[ ] 6. DOWNSTREAM IMPACT
    - Were our customers affected?
    - Do we need to notify anyone?
    - Regulatory obligations?
```

### Response

```
[ ] 7. CONTAINMENT
    - Disable vendor access if necessary
    - Implement compensating controls
    - Monitor for anomalies

[ ] 8. VENDOR MANAGEMENT
    - Document incident
    - Update risk assessment
    - Review contract terms
    - Consider relationship continuation

[ ] 9. COMMUNICATION
    - Internal stakeholders
    - Affected customers (if needed)
    - Regulators (if required)
```

---

## General Incident Documentation Template

```markdown
# Incident Report: [INCIDENT-ID]

## Summary
- **Date/Time Detected:**
- **Date/Time Contained:**
- **Date/Time Resolved:**
- **Severity:**
- **Incident Commander:**

## Timeline
| Time | Event | Actor |
|------|-------|-------|
| | | |

## Impact
- Users affected:
- Data exposed:
- Services impacted:
- Duration:

## Root Cause
[Description of root cause]

## Actions Taken
1.
2.
3.

## Lessons Learned
- What worked well:
- What could be improved:

## Follow-up Items
- [ ] Action item 1 (Owner, Due date)
- [ ] Action item 2 (Owner, Due date)

## Appendices
- Logs
- Screenshots
- Communications
```

---

*Last updated: January 2026*
*Document owner: Security Team*
*Review cycle: Semi-annual*
