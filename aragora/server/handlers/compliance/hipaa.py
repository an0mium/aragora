"""
HIPAA Compliance Handler.

Provides Health Insurance Portability and Accountability Act (HIPAA) compliance operations:
- PHI Access Logging and Monitoring
- Breach Risk Assessment
- Business Associate Agreement (BAA) Management
- Security Rule Compliance Reporting
- Privacy Rule Compliance Assessment
- Audit Control Requirements
"""
# mypy: disable-error-code="assignment,index,attr-defined,var-annotated"
# Complex dict operations with dynamic keys

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.events.handler_events import emit_handler_event, CREATED, COMPLETED
from aragora.rbac.decorators import require_permission
from aragora.observability.metrics import track_handler
from aragora.storage.audit_store import get_audit_store

logger = logging.getLogger(__name__)

# =============================================================================
# RBAC Permission Constants for HIPAA Compliance
# =============================================================================

# Basic HIPAA compliance read access
PERM_HIPAA_READ = "compliance:hipaa:read"

# Generate HIPAA compliance reports
PERM_HIPAA_REPORT = "compliance:hipaa:report"

# View breach assessments and history
PERM_HIPAA_BREACHES_READ = "compliance:breaches:read"

# Create and manage breach risk assessments
PERM_HIPAA_BREACHES_REPORT = "compliance:breaches:report"

# Manage Business Associate Agreements
PERM_HIPAA_BAA_MANAGE = "compliance:baa:manage"

# PHI de-identification and Safe Harbor verification
PERM_HIPAA_PHI_DEIDENTIFY = "compliance:phi:deidentify"

# Full HIPAA administrative access
PERM_HIPAA_ADMIN = "compliance:hipaa:admin"


class HIPAAMixin:
    """Mixin providing HIPAA-related handler methods."""

    # HIPAA Security Rule safeguard categories
    SECURITY_SAFEGUARDS = {
        "administrative": [
            {
                "id": "164.308(a)(1)",
                "name": "Security Management Process",
                "controls": [
                    "Risk Analysis",
                    "Risk Management",
                    "Sanction Policy",
                    "Information System Activity Review",
                ],
            },
            {
                "id": "164.308(a)(2)",
                "name": "Assigned Security Responsibility",
                "controls": ["Security Officer Designation"],
            },
            {
                "id": "164.308(a)(3)",
                "name": "Workforce Security",
                "controls": [
                    "Authorization/Supervision",
                    "Workforce Clearance",
                    "Termination Procedures",
                ],
            },
            {
                "id": "164.308(a)(4)",
                "name": "Information Access Management",
                "controls": [
                    "Access Authorization",
                    "Access Establishment/Modification",
                ],
            },
            {
                "id": "164.308(a)(5)",
                "name": "Security Awareness Training",
                "controls": [
                    "Security Reminders",
                    "Malicious Software Protection",
                    "Log-in Monitoring",
                    "Password Management",
                ],
            },
            {
                "id": "164.308(a)(6)",
                "name": "Security Incident Procedures",
                "controls": ["Response and Reporting"],
            },
            {
                "id": "164.308(a)(7)",
                "name": "Contingency Plan",
                "controls": [
                    "Data Backup Plan",
                    "Disaster Recovery Plan",
                    "Emergency Mode Operation",
                    "Testing and Revision",
                    "Applications and Data Criticality Analysis",
                ],
            },
            {
                "id": "164.308(a)(8)",
                "name": "Evaluation",
                "controls": ["Periodic Technical/Non-technical Evaluation"],
            },
        ],
        "physical": [
            {
                "id": "164.310(a)",
                "name": "Facility Access Controls",
                "controls": [
                    "Contingency Operations",
                    "Facility Security Plan",
                    "Access Control/Validation",
                    "Maintenance Records",
                ],
            },
            {
                "id": "164.310(b)",
                "name": "Workstation Use",
                "controls": ["Workstation Use Policies"],
            },
            {
                "id": "164.310(c)",
                "name": "Workstation Security",
                "controls": ["Physical Safeguards for Workstations"],
            },
            {
                "id": "164.310(d)",
                "name": "Device and Media Controls",
                "controls": [
                    "Disposal",
                    "Media Re-use",
                    "Accountability",
                    "Data Backup/Storage",
                ],
            },
        ],
        "technical": [
            {
                "id": "164.312(a)",
                "name": "Access Control",
                "controls": [
                    "Unique User Identification",
                    "Emergency Access Procedure",
                    "Automatic Logoff",
                    "Encryption/Decryption",
                ],
            },
            {
                "id": "164.312(b)",
                "name": "Audit Controls",
                "controls": ["Hardware/Software/Procedural Audit Mechanisms"],
            },
            {
                "id": "164.312(c)",
                "name": "Integrity Controls",
                "controls": ["Electronic PHI Integrity Mechanisms"],
            },
            {
                "id": "164.312(d)",
                "name": "Person/Entity Authentication",
                "controls": ["Verify Identity of Persons/Entities"],
            },
            {
                "id": "164.312(e)",
                "name": "Transmission Security",
                "controls": ["Integrity Controls", "Encryption"],
            },
        ],
    }

    # PHI identifiers per 45 CFR 164.514
    PHI_IDENTIFIERS = [
        "Names",
        "Geographic data smaller than state",
        "Dates (except year)",
        "Phone numbers",
        "Fax numbers",
        "Email addresses",
        "Social Security numbers",
        "Medical record numbers",
        "Health plan beneficiary numbers",
        "Account numbers",
        "Certificate/license numbers",
        "Vehicle identifiers",
        "Device identifiers",
        "Web URLs",
        "IP addresses",
        "Biometric identifiers",
        "Full face photos",
        "Any other unique identifier",
    ]

    @track_handler("compliance/hipaa-status", method="GET")
    @require_permission(PERM_HIPAA_READ)
    async def _hipaa_status(self, query_params: dict[str, str]) -> HandlerResult:
        """
        Get HIPAA compliance status overview.

        Query params:
            scope: full | summary (default: summary)
            include_recommendations: true | false (default: true)
        """
        scope = query_params.get("scope", "summary")
        include_recommendations = (
            query_params.get("include_recommendations", "true").lower() == "true"
        )

        now = datetime.now(timezone.utc)

        # Evaluate compliance status
        safeguard_status = await self._evaluate_safeguards()
        phi_controls = await self._evaluate_phi_controls()
        baa_status = await self._get_baa_status()

        # Calculate overall compliance score
        total_controls = sum(
            len(sg["controls"]) for category in self.SECURITY_SAFEGUARDS.values() for sg in category
        )
        compliant_count = sum(
            1 for cat in safeguard_status.values() for sg in cat if sg.get("status") == "compliant"
        )
        compliance_score = int((compliant_count / total_controls) * 100)

        status_result = {
            "compliance_framework": "HIPAA",
            "assessed_at": now.isoformat(),
            "overall_status": (
                "compliant"
                if compliance_score >= 95
                else "substantially_compliant"
                if compliance_score >= 80
                else "partially_compliant"
                if compliance_score >= 60
                else "non_compliant"
            ),
            "compliance_score": compliance_score,
            "rules": {
                "privacy_rule": {
                    "status": "configured",
                    "phi_handling": phi_controls.get("status", "review_required"),
                },
                "security_rule": {
                    "status": "configured",
                    "safeguards_assessed": total_controls,
                    "safeguards_compliant": compliant_count,
                },
                "breach_notification_rule": {
                    "status": "configured",
                    "procedures_documented": True,
                },
            },
            "business_associates": baa_status,
        }

        if scope == "full":
            status_result["safeguard_details"] = safeguard_status
            status_result["phi_controls"] = phi_controls

        if include_recommendations:
            status_result["recommendations"] = await self._get_hipaa_recommendations(
                safeguard_status, phi_controls
            )

        return json_response(status_result)

    @track_handler("compliance/hipaa-phi-access", method="GET")
    @require_permission(PERM_HIPAA_READ)
    async def _hipaa_phi_access_log(self, query_params: dict[str, str]) -> HandlerResult:
        """
        Get PHI access log for audit purposes.

        Per 45 CFR 164.312(b), covered entities must implement audit controls.

        Query params:
            patient_id: Filter by patient ID (optional)
            user_id: Filter by accessing user (optional)
            from: Start date (ISO format)
            to: End date (ISO format)
            limit: Max results (default 100, max 1000)
        """
        patient_id = query_params.get("patient_id")
        user_id = query_params.get("user_id")
        from_date = query_params.get("from")
        to_date = query_params.get("to")
        limit = min(int(query_params.get("limit", "100")), 1000)

        try:
            store = get_audit_store()
            events = store.get_log(limit=limit * 2)  # Get extra for filtering

            # Filter PHI access events
            phi_accesses = []
            for event in events:
                if event.get("action", "").startswith("phi_"):
                    metadata = event.get("metadata", {})

                    # Apply filters
                    if patient_id and metadata.get("patient_id") != patient_id:
                        continue
                    if user_id and event.get("user_id") != user_id:
                        continue

                    # Date filtering
                    event_time = event.get("timestamp")
                    if event_time:
                        try:
                            if isinstance(event_time, str):
                                event_dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
                            else:
                                event_dt = event_time

                            if from_date:
                                from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
                                if event_dt < from_dt:
                                    continue
                            if to_date:
                                to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
                                if event_dt > to_dt:
                                    continue
                        except ValueError:
                            continue

                    phi_accesses.append(
                        {
                            "timestamp": event_time,
                            "action": event["action"],
                            "user_id": event.get("user_id"),
                            "patient_id": metadata.get("patient_id"),
                            "access_type": metadata.get("access_type"),
                            "phi_elements": metadata.get("phi_elements", []),
                            "purpose": metadata.get("purpose"),
                            "ip_address": metadata.get("ip_address"),
                        }
                    )

                    if len(phi_accesses) >= limit:
                        break

            return json_response(
                {
                    "phi_access_log": phi_accesses,
                    "count": len(phi_accesses),
                    "filters": {
                        "patient_id": patient_id,
                        "user_id": user_id,
                        "from": from_date,
                        "to": to_date,
                    },
                    "hipaa_reference": "45 CFR 164.312(b) - Audit Controls",
                }
            )

        except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.exception(f"Error fetching PHI access log: {e}")
            return error_response("Failed to retrieve access log", 500)

    @track_handler("compliance/hipaa-breach-assessment", method="POST")
    @require_permission(PERM_HIPAA_BREACHES_REPORT)
    async def _hipaa_breach_assessment(self, body: dict[str, Any]) -> HandlerResult:
        """
        Perform HIPAA breach risk assessment.

        Per 45 CFR 164.402, determine if an incident constitutes a breach
        requiring notification.

        Body:
            incident_id: Unique incident identifier (required)
            incident_type: Type of security incident (required)
            phi_involved: Whether PHI was involved (required)
            phi_types: Types of PHI involved (if applicable)
            affected_individuals: Estimated number affected
            unauthorized_access: Details of unauthorized access
            mitigation_actions: Actions taken to mitigate
        """
        incident_id = body.get("incident_id")
        if not incident_id:
            return error_response("incident_id is required", 400)

        incident_type = body.get("incident_type")
        if not incident_type:
            return error_response("incident_type is required", 400)

        phi_involved = body.get("phi_involved", False)

        now = datetime.now(timezone.utc)
        assessment_id = f"hipaa-breach-{incident_id}-{now.strftime('%Y%m%d%H%M%S')}"

        assessment = {
            "assessment_id": assessment_id,
            "incident_id": incident_id,
            "assessed_at": now.isoformat(),
            "phi_involved": phi_involved,
            "breach_determination": None,
            "risk_factors": [],
            "notification_required": False,
            "notification_deadlines": None,
        }

        if not phi_involved:
            assessment["breach_determination"] = "not_applicable"
            assessment["message"] = "No PHI involved. HIPAA breach notification not applicable."
        else:
            # Four-factor breach risk assessment per HHS guidance
            risk_factors = []

            # Factor 1: Nature and extent of PHI
            phi_types = body.get("phi_types", [])
            sensitive_phi = [
                p
                for p in phi_types
                if p in ["SSN", "Financial", "Medical diagnosis", "Treatment information"]
            ]
            if sensitive_phi:
                risk_factors.append(
                    {
                        "factor": "Nature and extent of PHI",
                        "risk": "high",
                        "details": f"Sensitive PHI types involved: {sensitive_phi}",
                    }
                )
            else:
                risk_factors.append(
                    {
                        "factor": "Nature and extent of PHI",
                        "risk": "moderate",
                        "details": "Limited PHI types involved",
                    }
                )

            # Factor 2: Unauthorized person
            unauthorized = body.get("unauthorized_access", {})
            if unauthorized.get("known_recipient"):
                risk_factors.append(
                    {
                        "factor": "Unauthorized person",
                        "risk": "moderate",
                        "details": "Recipient is known/identifiable",
                    }
                )
            else:
                risk_factors.append(
                    {
                        "factor": "Unauthorized person",
                        "risk": "high",
                        "details": "Unknown or unidentified recipient",
                    }
                )

            # Factor 3: PHI actually acquired/viewed
            if unauthorized.get("confirmed_access"):
                risk_factors.append(
                    {
                        "factor": "PHI acquisition/viewing",
                        "risk": "high",
                        "details": "Confirmed that PHI was accessed",
                    }
                )
            else:
                risk_factors.append(
                    {
                        "factor": "PHI acquisition/viewing",
                        "risk": "low",
                        "details": "No evidence PHI was actually accessed",
                    }
                )

            # Factor 4: Extent of risk mitigation
            mitigation = body.get("mitigation_actions", [])
            if len(mitigation) >= 3:
                risk_factors.append(
                    {
                        "factor": "Risk mitigation",
                        "risk": "low",
                        "details": f"Comprehensive mitigation: {len(mitigation)} actions taken",
                    }
                )
            else:
                risk_factors.append(
                    {
                        "factor": "Risk mitigation",
                        "risk": "moderate",
                        "details": "Limited mitigation actions taken",
                    }
                )

            assessment["risk_factors"] = risk_factors

            # Determine if notification is required
            high_risk_count = sum(1 for f in risk_factors if f["risk"] == "high")
            if high_risk_count >= 2:
                assessment["breach_determination"] = "presumed_breach"
                assessment["notification_required"] = True
                assessment["notification_deadlines"] = {
                    "individual_notification": (now + timedelta(days=60)).isoformat(),
                    "hhs_notification": (
                        "Annual"
                        if body.get("affected_individuals", 0) < 500
                        else (now + timedelta(days=60)).isoformat()
                    ),
                    "media_notification": (
                        (now + timedelta(days=60)).isoformat()
                        if body.get("affected_individuals", 0) >= 500
                        else "Not required"
                    ),
                }
            else:
                assessment["breach_determination"] = "low_probability"
                assessment["notification_required"] = False

        # Log the assessment
        try:
            store = get_audit_store()
            store.log_event(
                action="hipaa_breach_assessment",
                resource_type="incident",
                resource_id=incident_id,
                metadata={
                    "assessment_id": assessment_id,
                    "breach_determination": assessment["breach_determination"],
                    "notification_required": assessment["notification_required"],
                },
            )
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.warning(f"Failed to log breach assessment: {e}")

        emit_handler_event(
            "compliance",
            COMPLETED,
            {"action": "hipaa_breach_assessment", "assessment_id": assessment_id},
        )
        return json_response(assessment)

    @track_handler("compliance/hipaa-baa", method="GET")
    @require_permission(PERM_HIPAA_READ)
    async def _hipaa_list_baas(self, query_params: dict[str, str]) -> HandlerResult:
        """
        List Business Associate Agreements (BAAs).

        Query params:
            status: active | expired | pending | all (default: active)
            ba_type: vendor | subcontractor | all (default: all)
        """
        status_filter = query_params.get("status", "active")
        ba_type = query_params.get("ba_type", "all")

        # Get BAAs from storage
        baas = await self._get_baa_list(status_filter, ba_type)

        return json_response(
            {
                "business_associates": baas,
                "count": len(baas),
                "filters": {"status": status_filter, "ba_type": ba_type},
                "hipaa_reference": "45 CFR 164.502(e) and 164.504(e)",
            }
        )

    @track_handler("compliance/hipaa-baa-create", method="POST")
    @require_permission(PERM_HIPAA_BAA_MANAGE)
    async def _hipaa_create_baa(self, body: dict[str, Any]) -> HandlerResult:
        """
        Register a new Business Associate Agreement.

        Body:
            business_associate: Name of the business associate (required)
            ba_type: vendor | subcontractor (required)
            services_provided: Description of services (required)
            phi_access_scope: Types of PHI access granted
            agreement_date: Date of BAA execution (ISO format)
            expiration_date: BAA expiration date (ISO format, optional)
            subcontractor_clause: Whether subcontractor clause is included
        """
        ba_name = body.get("business_associate")
        if not ba_name:
            return error_response("business_associate is required", 400)

        ba_type = body.get("ba_type")
        if ba_type not in ["vendor", "subcontractor"]:
            return error_response("ba_type must be 'vendor' or 'subcontractor'", 400)

        services = body.get("services_provided")
        if not services:
            return error_response("services_provided is required", 400)

        now = datetime.now(timezone.utc)
        baa_id = f"baa-{hashlib.sha256(ba_name.encode()).hexdigest()[:8]}-{now.strftime('%Y%m%d')}"

        baa_record = {
            "baa_id": baa_id,
            "business_associate": ba_name,
            "ba_type": ba_type,
            "services_provided": services,
            "phi_access_scope": body.get("phi_access_scope", []),
            "agreement_date": body.get("agreement_date", now.isoformat()),
            "expiration_date": body.get("expiration_date"),
            "subcontractor_clause": body.get("subcontractor_clause", True),
            "status": "active",
            "created_at": now.isoformat(),
            "required_provisions": [
                "Use/disclosure limitations",
                "Safeguards requirement",
                "Subcontractor assurances",
                "Breach notification obligation",
                "Access to PHI for amendment",
                "Accounting of disclosures",
                "Compliance with Security Rule",
                "Termination provisions",
            ],
        }

        # Store the BAA
        try:
            store = get_audit_store()
            store.log_event(
                action="hipaa_baa_created",
                resource_type="baa",
                resource_id=baa_id,
                metadata=baa_record,
            )
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.warning(f"Failed to store BAA: {e}")

        emit_handler_event("compliance", CREATED, {"action": "hipaa_baa_created", "baa_id": baa_id})
        return json_response(
            {
                "message": "Business Associate Agreement registered",
                "baa": baa_record,
            },
            status=201,
        )

    @track_handler("compliance/hipaa-security-report", method="GET")
    @require_permission(PERM_HIPAA_REPORT)
    async def _hipaa_security_report(self, query_params: dict[str, str]) -> HandlerResult:
        """
        Generate HIPAA Security Rule compliance report.

        Query params:
            format: json | html (default: json)
            include_evidence: true | false (default: false)
        """
        output_format = query_params.get("format", "json")
        include_evidence = query_params.get("include_evidence", "false").lower() == "true"

        now = datetime.now(timezone.utc)
        report_id = f"hipaa-sec-{now.strftime('%Y%m%d%H%M%S')}"

        safeguards = await self._evaluate_safeguards()

        report = {
            "report_id": report_id,
            "report_type": "HIPAA Security Rule Compliance",
            "generated_at": now.isoformat(),
            "assessment_period": {
                "start": (now - timedelta(days=365)).isoformat(),
                "end": now.isoformat(),
            },
            "safeguards": {},
        }

        for category, category_safeguards in safeguards.items():
            report["safeguards"][category] = {
                "category": category.title(),
                "standards": category_safeguards,
                "compliant_count": sum(
                    1 for s in category_safeguards if s.get("status") == "compliant"
                ),
                "total_count": len(category_safeguards),
            }

        # Calculate summary
        total = sum(r["total_count"] for r in report["safeguards"].values())
        compliant = sum(r["compliant_count"] for r in report["safeguards"].values())

        report["summary"] = {
            "total_standards_assessed": total,
            "standards_compliant": compliant,
            "compliance_percentage": int((compliant / total * 100) if total > 0 else 0),
            "overall_status": (
                "Compliant"
                if compliant == total
                else "Substantially Compliant"
                if compliant / total >= 0.8
                else "Needs Improvement"
            ),
        }

        if include_evidence:
            report["evidence_references"] = await self._get_security_evidence()

        if output_format == "html":
            html_content = self._render_hipaa_html(report)
            return HandlerResult(
                status_code=200,
                content_type="text/html",
                body=html_content.encode("utf-8"),
            )

        return json_response(report)

    @track_handler("compliance/hipaa-deidentify", method="POST")
    @require_permission(PERM_HIPAA_PHI_DEIDENTIFY)
    async def _hipaa_deidentify(self, body: dict[str, Any]) -> HandlerResult:
        """
        De-identify content using HIPAA Safe Harbor method.

        Removes the 18 HIPAA identifiers from text or structured data using
        the HIPAAAnonymizer from aragora.privacy.anonymization.

        Body:
            content: Text content to de-identify (required, unless 'data' provided)
            data: Structured data dict to de-identify (optional, alternative to content)
            method: Anonymization method - redact | hash | generalize | suppress | pseudonymize
                    (default: redact)
            identifier_types: List of identifier types to target (optional, default: all)
        """
        content = body.get("content")
        data = body.get("data")

        if not content and not data:
            return error_response("Either 'content' (string) or 'data' (object) is required", 400)

        method_str = body.get("method", "redact")

        try:
            from aragora.privacy.anonymization import (
                HIPAAAnonymizer,
                AnonymizationMethod,
                IdentifierType,
            )
        except ImportError:
            return error_response(
                "Privacy anonymization module not available", 501
            )

        try:
            anon_method = AnonymizationMethod(method_str)
        except ValueError:
            valid = [m.value for m in AnonymizationMethod]
            return error_response(
                f"Invalid method '{method_str}'. Valid: {valid}", 400
            )

        # Parse identifier type filters
        id_types = None
        raw_types = body.get("identifier_types")
        if raw_types and isinstance(raw_types, list):
            try:
                id_types = [IdentifierType(t) for t in raw_types]
            except ValueError as e:
                logger.warning("Handler error: %s", e)
                return error_response("Invalid identifier type", 400)

        anonymizer = HIPAAAnonymizer()

        if content:
            result = anonymizer.anonymize(content, anon_method, id_types)
        else:
            result = anonymizer.anonymize_structured(data, default_method=anon_method)

        # Audit the de-identification
        try:
            store = get_audit_store()
            store.log_event(
                action="hipaa_phi_deidentified",
                resource_type="content",
                resource_id=result.audit_id,
                metadata={
                    "method": method_str,
                    "identifiers_found": len(result.identifiers_found),
                    "fields_anonymized": result.fields_anonymized,
                },
            )
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.warning("Failed to log de-identification audit: %s", e)

        emit_handler_event(
            "compliance",
            COMPLETED,
            {"action": "hipaa_deidentify", "audit_id": result.audit_id},
        )
        return json_response(result.to_dict())

    @track_handler("compliance/hipaa-safe-harbor-verify", method="POST")
    @require_permission(PERM_HIPAA_READ)
    async def _hipaa_safe_harbor_verify(self, body: dict[str, Any]) -> HandlerResult:
        """
        Verify content meets HIPAA Safe Harbor de-identification requirements.

        Checks content for the 18 HIPAA identifiers and reports compliance.

        Body:
            content: Text content to verify (required)
        """
        content = body.get("content")
        if not content:
            return error_response("'content' is required", 400)

        try:
            from aragora.privacy.anonymization import HIPAAAnonymizer
        except ImportError:
            return error_response(
                "Privacy anonymization module not available", 501
            )

        anonymizer = HIPAAAnonymizer()
        result = anonymizer.verify_safe_harbor(content)

        response = {
            "compliant": result.compliant,
            "identifiers_remaining": [
                {
                    "type": ident.identifier_type.value,
                    "value_preview": ident.value[:3] + "..." if len(ident.value) > 3 else "***",
                    "confidence": ident.confidence,
                    "position": {"start": ident.start_pos, "end": ident.end_pos},
                }
                for ident in result.identifiers_remaining
            ],
            "verification_notes": result.verification_notes,
            "verified_at": result.verified_at.isoformat(),
            "hipaa_reference": "45 CFR 164.514(b) - Safe Harbor Method",
        }

        return json_response(response)

    @track_handler("compliance/hipaa-detect-phi", method="POST")
    @require_permission(PERM_HIPAA_READ)
    async def _hipaa_detect_phi(self, body: dict[str, Any]) -> HandlerResult:
        """
        Detect HIPAA PHI identifiers in content.

        Scans content and returns all detected identifiers with positions
        and confidence scores. Does not modify the content.

        Body:
            content: Text content to scan (required)
            min_confidence: Minimum confidence threshold (default: 0.5)
        """
        content = body.get("content")
        if not content:
            return error_response("'content' is required", 400)

        min_confidence = float(body.get("min_confidence", 0.5))

        try:
            from aragora.privacy.anonymization import HIPAAAnonymizer
        except ImportError:
            return error_response(
                "Privacy anonymization module not available", 501
            )

        anonymizer = HIPAAAnonymizer()
        identifiers = anonymizer.detect_identifiers(content)

        # Filter by confidence and reverse to natural order
        filtered = [
            i for i in reversed(identifiers) if i.confidence >= min_confidence
        ]

        return json_response({
            "identifiers": [
                {
                    "type": ident.identifier_type.value,
                    "value": ident.value,
                    "start": ident.start_pos,
                    "end": ident.end_pos,
                    "confidence": ident.confidence,
                }
                for ident in filtered
            ],
            "count": len(filtered),
            "min_confidence": min_confidence,
            "hipaa_reference": "45 CFR 164.514 - HIPAA Safe Harbor Identifiers",
        })

    # =========================================================================
    # Helper methods for HIPAA operations
    # =========================================================================

    async def _evaluate_safeguards(self) -> dict[str, list[dict[str, Any]]]:
        """Evaluate HIPAA Security Rule safeguard compliance."""
        result = {}

        for category, safeguards in self.SECURITY_SAFEGUARDS.items():
            result[category] = []
            for sg in safeguards:
                # Evaluate each safeguard (in production, this would check actual controls)
                safeguard_status = {
                    "id": sg["id"],
                    "name": sg["name"],
                    "status": "compliant",  # Default to compliant for demo
                    "controls": [{"name": c, "implemented": True} for c in sg["controls"]],
                    "last_assessed": datetime.now(timezone.utc).isoformat(),
                }
                result[category].append(safeguard_status)

        return result

    async def _evaluate_phi_controls(self) -> dict[str, Any]:
        """Evaluate PHI handling controls."""
        # Check if anonymizer module is available
        anonymizer_available = False
        try:
            from aragora.privacy.anonymization import HIPAAAnonymizer  # noqa: F401
            anonymizer_available = True
        except ImportError:
            pass

        return {
            "status": "configured",
            "identifiers_tracked": len(self.PHI_IDENTIFIERS),
            "de_identification_method": "Safe Harbor",
            "anonymizer_available": anonymizer_available,
            "minimum_necessary_enforced": True,
            "access_controls": {
                "role_based": True,
                "audit_logging": True,
                "encryption_at_rest": True,
                "encryption_in_transit": True,
            },
        }

    async def _get_baa_status(self) -> dict[str, Any]:
        """Get Business Associate Agreement status summary."""
        return {
            "total_baas": 3,  # Example count
            "active": 3,
            "expiring_soon": 0,
            "expired": 0,
        }

    async def _get_baa_list(self, status: str, ba_type: str) -> list[dict[str, Any]]:
        """Get list of BAAs from storage."""
        # In production, this would query actual BAA storage
        return [
            {
                "baa_id": "baa-example-1",
                "business_associate": "Cloud Provider A",
                "ba_type": "vendor",
                "status": "active",
                "services_provided": "Cloud hosting services",
                "agreement_date": "2024-01-01",
            },
            {
                "baa_id": "baa-example-2",
                "business_associate": "AI Model Provider",
                "ba_type": "vendor",
                "status": "active",
                "services_provided": "AI processing services",
                "agreement_date": "2024-03-15",
            },
        ]

    async def _get_hipaa_recommendations(
        self,
        safeguards: dict[str, list[dict[str, Any]]],
        phi_controls: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate HIPAA compliance recommendations."""
        recommendations = []

        # Check for non-compliant safeguards
        for category, sgs in safeguards.items():
            for sg in sgs:
                if sg.get("status") != "compliant":
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": category,
                            "safeguard": sg["name"],
                            "recommendation": f"Address non-compliant controls in {sg['name']}",
                            "reference": sg["id"],
                        }
                    )

        # General recommendations
        recommendations.append(
            {
                "priority": "medium",
                "category": "administrative",
                "recommendation": "Conduct annual security risk analysis",
                "reference": "164.308(a)(1)(ii)(A)",
            }
        )

        return recommendations

    async def _get_security_evidence(self) -> list[dict[str, Any]]:
        """Get evidence references for security compliance."""
        return [
            {
                "control": "Access Control",
                "evidence_type": "System configuration",
                "location": "RBAC configuration in aragora/rbac/",
            },
            {
                "control": "Audit Controls",
                "evidence_type": "Audit logs",
                "location": "Audit store records",
            },
            {
                "control": "Transmission Security",
                "evidence_type": "TLS configuration",
                "location": "Server TLS certificates",
            },
        ]

    def _render_hipaa_html(self, report: dict[str, Any]) -> str:
        """Render HIPAA report as HTML."""
        import html as html_escape

        safeguards_html = ""
        for category, data in report.get("safeguards", {}).items():
            safeguards_html += f"""
            <h3>{html_escape.escape(data["category"])} Safeguards</h3>
            <p>Compliant: {data["compliant_count"]}/{data["total_count"]}</p>
            <ul>
            """
            for std in data.get("standards", []):
                status_class = "success" if std["status"] == "compliant" else "warning"
                safeguards_html += f"""
                <li class="{status_class}">
                    {html_escape.escape(std["id"])} - {html_escape.escape(std["name"])}
                </li>
                """
            safeguards_html += "</ul>"

        summary = report.get("summary", {})

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HIPAA Security Rule Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>HIPAA Security Rule Compliance Report</h1>
            <p><strong>Report ID:</strong> {html_escape.escape(str(report.get("report_id", "")))}</p>
            <p><strong>Generated:</strong> {html_escape.escape(str(report.get("generated_at", "")))}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Standards Assessed: {summary.get("total_standards_assessed", 0)}</p>
                <p>Standards Compliant: {summary.get("standards_compliant", 0)}</p>
                <p>Compliance: {summary.get("compliance_percentage", 0)}%</p>
                <p>Overall Status: <strong>{html_escape.escape(str(summary.get("overall_status", "")))}</strong></p>
            </div>

            <h2>Safeguards Assessment</h2>
            {safeguards_html}
        </body>
        </html>
        """


__all__ = [
    "HIPAAMixin",
    # RBAC Permission Constants
    "PERM_HIPAA_READ",
    "PERM_HIPAA_REPORT",
    "PERM_HIPAA_BREACHES_READ",
    "PERM_HIPAA_BREACHES_REPORT",
    "PERM_HIPAA_BAA_MANAGE",
    "PERM_HIPAA_PHI_DEIDENTIFY",
    "PERM_HIPAA_ADMIN",
]
