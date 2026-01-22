# Penetration Testing Requirements

This document outlines the penetration testing requirements for Aragora, ensuring compliance with SOC 2 CC4.1 (Security Monitoring) and CC4.2 (Monitoring Activities).

## Overview

Aragora requires regular penetration testing to:
- Identify security vulnerabilities before attackers do
- Validate security controls are functioning correctly
- Meet compliance requirements (SOC 2, ISO 27001)
- Demonstrate security due diligence to customers

## Testing Schedule

| Test Type | Frequency | Scope |
|-----------|-----------|-------|
| External Network Pentest | Annually | All public-facing infrastructure |
| Web Application Pentest | Bi-annually | API endpoints, WebSocket handlers |
| Internal Network Pentest | Annually | Internal services, agent communication |
| Social Engineering | Annually | Phishing simulations for staff |
| Cloud Configuration Review | Quarterly | AWS/GCP security configuration |
| API Security Assessment | Bi-annually | All REST and GraphQL endpoints |

## Scope Definition

### In Scope

#### 1. External Attack Surface
- Public API endpoints (`api.aragora.ai`)
- WebSocket connections (`wss://api.aragora.ai/ws`)
- OAuth/OIDC authentication endpoints
- Public documentation portals
- DNS configuration

#### 2. Web Application
- Authentication and session management
- Authorization and access controls
- Input validation and output encoding
- API rate limiting and abuse prevention
- File upload functionality
- WebSocket message handling

#### 3. API Security
- REST API (`/api/v1/*`)
- GraphQL endpoint (`/graphql`)
- Webhook handlers (`/webhooks/*`)
- Agent communication protocols
- Inter-service authentication

#### 4. Infrastructure
- Load balancers and CDN configuration
- Database access controls (Supabase/PostgreSQL)
- Redis cache security
- Message queue security
- Container security (if applicable)

#### 5. Authentication Systems
- JWT token generation and validation
- OAuth 2.0 / OIDC flows (Google, GitHub)
- API key management
- Session handling
- MFA implementation

### Out of Scope

- Third-party SaaS providers (Stripe, Anthropic, OpenAI)
- Customer production environments
- Physical security testing
- Denial of Service (DoS) attacks without prior coordination
- Social engineering of customers

## Test Methodologies

### OWASP Testing Guide

All web application tests must follow the [OWASP Testing Guide v4.2](https://owasp.org/www-project-web-security-testing-guide/):

1. **Information Gathering**
   - Fingerprint web server
   - Enumerate applications
   - Map application architecture

2. **Configuration and Deployment Management Testing**
   - Test network infrastructure
   - Test application configuration
   - Test file extensions handling
   - Review backup files

3. **Identity Management Testing**
   - Test role definitions
   - Test user registration process
   - Test account provisioning
   - Test account enumeration

4. **Authentication Testing**
   - Test for credentials transported over encrypted channel
   - Test for default credentials
   - Test for weak lockout mechanism
   - Test for bypassing authentication schema
   - Test for browser cache weakness

5. **Authorization Testing**
   - Test directory traversal
   - Test for bypassing authorization schema
   - Test for privilege escalation
   - Test for insecure direct object references (IDOR)

6. **Session Management Testing**
   - Test for session fixation
   - Test for cookie attributes
   - Test for session timeout
   - Test for CSRF

7. **Input Validation Testing**
   - Test for reflected XSS
   - Test for stored XSS
   - Test for DOM-based XSS
   - Test for SQL injection
   - Test for command injection
   - Test for HTTP parameter pollution

8. **Error Handling**
   - Test for improper error handling
   - Test for stack traces

9. **Cryptography**
   - Test for weak TLS/SSL
   - Test for sensitive data in storage
   - Test for insecure cryptographic algorithms

10. **Business Logic Testing**
    - Test for business logic flaws
    - Test for process timing
    - Test for function limits

### API-Specific Testing

Follow [OWASP API Security Top 10](https://owasp.org/www-project-api-security/):

| Risk | Testing Focus |
|------|---------------|
| API1:2023 Broken Object Level Authorization | Test IDOR across all endpoints |
| API2:2023 Broken Authentication | Test auth flows, token handling |
| API3:2023 Broken Object Property Level Authorization | Test mass assignment, excessive data exposure |
| API4:2023 Unrestricted Resource Consumption | Test rate limiting, pagination |
| API5:2023 Broken Function Level Authorization | Test admin endpoints, privilege escalation |
| API6:2023 Unrestricted Access to Sensitive Business Flows | Test business logic abuse |
| API7:2023 Server Side Request Forgery | Test URL parameters, webhooks |
| API8:2023 Security Misconfiguration | Test headers, CORS, debug endpoints |
| API9:2023 Improper Inventory Management | Test deprecated endpoints, documentation |
| API10:2023 Unsafe Consumption of APIs | Test third-party integrations |

### Agent Security Testing

Aragora's multi-agent architecture requires specific testing:

1. **Agent Communication**
   - Test inter-agent message integrity
   - Test agent authentication
   - Test prompt injection resistance
   - Test output sanitization

2. **Agent Isolation**
   - Test sandbox boundaries
   - Test resource limits enforcement
   - Test credential isolation

3. **AI-Specific Attacks**
   - Prompt injection attempts
   - Jailbreak attempts
   - Data exfiltration via agent responses
   - Model manipulation attempts

## Rules of Engagement

### Pre-Test Requirements

1. **Written Authorization**
   - Signed penetration testing agreement
   - Defined scope and timeline
   - Emergency contacts list
   - Escalation procedures

2. **Testing Windows**
   - Coordinate with DevOps for timing
   - Avoid high-traffic periods
   - Notify monitoring teams

3. **Credentials Provided**
   - Test accounts at each permission level:
     - Standard user
     - Workspace admin
     - Super admin
   - Test API keys with various scopes

### During Testing

1. **Communication**
   - Daily status updates during active testing
   - Immediate notification of critical findings
   - Document all test activities

2. **Restrictions**
   - No DoS/DDoS attacks without explicit approval
   - No data destruction or modification
   - No access to customer data
   - No lateral movement to customer environments

3. **Evidence Collection**
   - Screenshot all findings
   - Record request/response pairs
   - Document reproduction steps
   - Preserve log evidence

### Post-Test

1. **Reporting Timeline**
   - Critical findings: Within 24 hours
   - High findings: Within 48 hours
   - Full report: Within 2 weeks

2. **Cleanup**
   - Remove all test accounts
   - Delete uploaded test files
   - Clear any injected data
   - Provide list of IPs used

## Vulnerability Classification

### Severity Ratings

| Severity | CVSS Score | Response Time | Examples |
|----------|------------|---------------|----------|
| Critical | 9.0-10.0 | 24 hours | RCE, auth bypass, data breach |
| High | 7.0-8.9 | 72 hours | SQLi, privilege escalation, XSS (stored) |
| Medium | 4.0-6.9 | 2 weeks | XSS (reflected), CSRF, info disclosure |
| Low | 0.1-3.9 | 30 days | Missing headers, verbose errors |
| Informational | N/A | Quarterly review | Best practice recommendations |

### Finding Report Template

Each finding must include:

```markdown
## [SEVERITY] Finding Title

### Summary
Brief description of the vulnerability.

### Affected Component
- URL/Endpoint:
- Parameter:
- Authentication Required: Yes/No

### Technical Details
Detailed description with request/response examples.

### Proof of Concept
Step-by-step reproduction instructions.

### Impact
Business impact and risk assessment.

### Recommendations
Specific remediation guidance.

### References
- CVE (if applicable)
- OWASP reference
- CWE classification
```

## Testing Environments

### Staging Environment

- **URL**: `https://staging.aragora.ai`
- **Purpose**: Primary testing target
- **Data**: Synthetic test data only
- **Availability**: 24/7 during test window

### Production Considerations

Production testing is only permitted for:
- Passive reconnaissance
- TLS/certificate verification
- Header analysis
- DNS enumeration

Active testing against production requires explicit approval.

## Vendor Requirements

### Minimum Qualifications

Penetration testing vendors must:
- Hold CREST, OSCP, or equivalent certifications
- Carry minimum $2M professional liability insurance
- Sign NDA before engagement
- Provide references from similar SaaS companies
- Demonstrate AI/ML security testing experience (preferred)

### Approved Vendors

Contact security@aragora.ai for current approved vendor list.

## Remediation Process

### Workflow

1. **Triage**: Security team reviews findings within 24 hours
2. **Prioritize**: Assign severity and owner
3. **Remediate**: Fix within SLA based on severity
4. **Verify**: Retest to confirm fix
5. **Close**: Document resolution and lessons learned

### Tracking

All findings are tracked in:
- Internal security issue tracker
- SOC 2 audit evidence repository
- Risk register (for high/critical findings)

## Compliance Mapping

| Requirement | SOC 2 Control | Evidence |
|-------------|---------------|----------|
| Annual external pentest | CC4.1 | Pentest report |
| Vulnerability remediation | CC4.2 | Remediation tracking |
| Security monitoring | CC7.2 | Finding notification |
| Risk assessment | CC3.2 | Risk register updates |

## Contact Information

- **Security Team**: security@aragora.ai
- **Emergency Contact**: [On-call rotation]
- **Compliance Officer**: compliance@aragora.ai

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-21 | Security Team | Initial release |

---

*This document is reviewed quarterly and updated as needed.*
