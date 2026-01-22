---
title: Remote Work Security Policy
description: Remote Work Security Policy
---

# Remote Work Security Policy

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Security Team

---

## Purpose

This policy establishes security requirements and best practices for Aragora team members working remotely. It ensures consistent security controls regardless of work location while maintaining productivity and flexibility.

**SOC 2 Control:** CC5-02 - Remote access security controls

---

## Scope

This policy applies to:
- All employees, contractors, and consultants
- Any work performed outside Aragora office locations
- All devices used to access Aragora systems remotely

---

## Device Security

### 1. Approved Devices

| Device Type | Requirements |
|-------------|-------------|
| Company-issued laptops | Primary work device, fully managed |
| Personal devices (BYOD) | Must meet security requirements below |
| Mobile devices | MDM enrollment required for email/Slack |

### 2. Minimum Device Requirements

All devices accessing Aragora systems must have:

- [ ] Full disk encryption enabled (FileVault, BitLocker, LUKS)
- [ ] Operating system auto-updates enabled
- [ ] Firewall enabled
- [ ] Antivirus/endpoint protection (company-approved)
- [ ] Screen lock after 5 minutes of inactivity
- [ ] Strong password/biometric authentication
- [ ] No jailbroken/rooted devices

### 3. Lost or Stolen Devices

**Immediate Actions:**
1. Report to security@aragora.ai within 1 hour
2. Initiate remote wipe if device had sensitive data
3. Change all passwords used on the device
4. Revoke any active sessions

**Process:**
```
1. Contact: security@aragora.ai
2. Subject: "Lost/Stolen Device - [Your Name]"
3. Include: Device type, last known location, data stored
4. IT will: Revoke certificates, wipe device, audit access logs
```

---

## Network Security

### 1. VPN Requirements

| Activity | VPN Required? |
|----------|---------------|
| Accessing internal tools (admin panels) | Yes |
| Accessing production databases | Yes |
| Code review and development | Recommended |
| Email and Slack | No (encrypted by default) |
| Public documentation | No |

**Approved VPN Clients:**
- WireGuard (preferred)
- OpenVPN
- Tailscale (for specific use cases)

### 2. Prohibited Networks

Do NOT access Aragora systems from:
- Public WiFi without VPN (airports, cafes, hotels)
- Shared computers (libraries, business centers)
- Networks in high-risk countries (refer to compliance list)
- Tor or anonymizing proxies

### 3. Home Network Security

**Required:**
- [ ] WPA3 or WPA2 encryption on WiFi
- [ ] Change default router password
- [ ] Disable WPS (WiFi Protected Setup)
- [ ] Keep router firmware updated

**Recommended:**
- [ ] Separate IoT devices on guest network
- [ ] Enable router firewall
- [ ] Use DNS filtering (e.g., NextDNS, 1.1.1.2)

---

## Authentication

### 1. Password Requirements

| System | Minimum Length | Complexity | Rotation |
|--------|----------------|------------|----------|
| Aragora accounts | 14 characters | Upper, lower, number, symbol | 90 days |
| SSH keys | 4096-bit RSA or Ed25519 | N/A | Annual |
| API keys | 32 characters | Random | On compromise |

### 2. Multi-Factor Authentication (MFA)

**Required for:**
- All Aragora accounts
- Admin/owner roles (enforced, SOC 2 CC5-01)
- SSH access to production servers
- AWS console access
- GitHub organization

**Approved MFA Methods:**
- TOTP authenticator apps (Authy, Google Authenticator)
- Hardware security keys (YubiKey)

**Not Allowed:**
- SMS-based MFA (SIM swap risk)
- Email-based MFA (phishing risk)

### 3. Session Management

| Session Type | Maximum Duration | Idle Timeout |
|--------------|------------------|--------------|
| Web dashboard | 24 hours | 30 minutes |
| API tokens | 1 year | N/A (revoke manually) |
| SSH sessions | 8 hours | 15 minutes |
| Admin sessions | 8 hours | 15 minutes |

---

## Data Handling

### 1. Data Classification Compliance

Remote workers must follow the [Data Classification Policy](./DATA_CLASSIFICATION.md):

| Classification | Remote Work Rules |
|----------------|-------------------|
| Public | No restrictions |
| Internal | Encrypted storage only |
| Confidential | VPN required, no local copies |
| Restricted | Approval required, audit logged |

### 2. Prohibited Activities

Do NOT:
- Store customer data on personal devices
- Share credentials or access tokens
- Take screenshots of sensitive data
- Print confidential documents at home
- Use personal cloud storage for work files
- Access production data from untrusted networks

### 3. Approved Tools

| Category | Approved Tools |
|----------|----------------|
| Code hosting | GitHub (aragora org) |
| Communication | Slack, Google Meet |
| Documentation | Notion, Google Docs |
| File sharing | Google Drive (company domain) |
| Password management | 1Password (team vault) |

---

## Physical Security

### 1. Workspace Requirements

- [ ] Private workspace (not visible to others)
- [ ] Lockable space for devices when unattended
- [ ] No shoulder-surfing risk during calls
- [ ] Webcam awareness during video calls

### 2. Screen Privacy

- Use privacy screen filter when working in public
- Position monitor away from windows/public view
- Lock screen when stepping away (Cmd+L / Win+L)
- Close sensitive tabs before screen sharing

### 3. Document Handling

- No printing of Confidential/Restricted data at home
- Shred any work-related documents before disposal
- Do not discuss sensitive matters in public spaces

---

## Incident Response

### 1. Security Incidents

Report immediately to security@aragora.ai:
- Suspected phishing attempts
- Malware infections
- Unauthorized access attempts
- Lost or stolen devices
- Data exposure

### 2. Incident Severity Levels

| Level | Examples | Response Time |
|-------|----------|---------------|
| Critical | Data breach, ransomware | Immediate |
| High | Account compromise, malware | 1 hour |
| Medium | Phishing attempt (clicked) | 4 hours |
| Low | Suspicious email (not clicked) | 24 hours |

### 3. Contact Information

| Contact | Use For |
|---------|---------|
| security@aragora.ai | Security incidents |
| it-support@aragora.ai | Device/access issues |
| #security-alerts (Slack) | Real-time updates |
| On-call: +1-XXX-XXX-XXXX | After-hours emergencies |

---

## Compliance Monitoring

### 1. Auditing

- Device compliance checked quarterly
- VPN usage logged and reviewed monthly
- Access patterns monitored for anomalies
- Annual security training completion tracked

### 2. Non-Compliance Consequences

| Violation | First Offense | Repeat Offense |
|-----------|---------------|----------------|
| Missing MFA | Warning + 24hr fix deadline | Access suspended |
| Unencrypted device | Warning + 48hr fix deadline | Device quarantined |
| Public WiFi without VPN | Warning + retraining | Access review |
| Data policy violation | Formal warning | Disciplinary action |

### 3. Exceptions

Request exceptions via security@aragora.ai:
- Business justification required
- Time-limited approval (max 30 days)
- Compensating controls documented
- Manager and Security approval

---

## Training Requirements

### 1. Required Training

| Training | Frequency | Duration |
|----------|-----------|----------|
| Security awareness | Annual | 1 hour |
| Phishing simulation | Quarterly | Ongoing |
| Remote work security | On hire + annual | 30 min |
| Data handling | Annual | 45 min |

### 2. Resources

- Security training portal: learn.aragora.ai/security
- Security FAQ: Notion > Security > FAQ
- Quick reference card: Notion > Security > Remote Work Checklist

---

## Quick Reference Checklist

### Daily Checklist
- [ ] VPN connected for sensitive work
- [ ] Screen locked when stepping away
- [ ] Suspicious emails reported
- [ ] Work data not on personal devices

### Weekly Checklist
- [ ] Device updates installed
- [ ] Browser extensions reviewed
- [ ] Downloaded files cleaned up

### Monthly Checklist
- [ ] Password manager audit
- [ ] Unused access revoked
- [ ] Home network security check

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |

---

## Related Documents

- [Data Classification Policy](./DATA_CLASSIFICATION.md)
- [Data Residency Policy](./DATA_RESIDENCY.md)
- [Incident Response Plan](./INCIDENT_RESPONSE.md)
- [Privacy Policy](./PRIVACY_POLICY.md)
