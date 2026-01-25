"""
Healthcare Connectors for HIPAA-Compliant Data Integration.

Supports:
- HL7 FHIR R4 API integration
- HL7 v2.x message integration
- EHR vendor adapters (Epic, Cerner)
- SMART on FHIR authentication
- PHI redaction (Safe Harbor method)
- MLLP transport protocol
- Comprehensive audit logging
"""

from aragora.connectors.enterprise.healthcare.fhir import (
    FHIRConnector,
    PHIRedactor,
    FHIRAuditLogger,
)
from aragora.connectors.enterprise.healthcare.hl7v2 import (
    HL7v2Connector,
    HL7Parser,
    HL7Message,
    HL7Segment,
    HL7PHIRedactor,
    HL7MessageType,
    HL7SegmentType,
    MSHSegment,
    PIDSegment,
    PV1Segment,
    OBXSegment,
    ORCSegment,
    OBRSegment,
    SCHSegment,
)
from aragora.connectors.enterprise.healthcare.ehr import (
    EHRAdapter,
    EHRVendor,
    EHRCapability,
    EHRConnectionConfig,
    EpicAdapter,
    CernerAdapter,
)

__all__ = [
    # FHIR
    "FHIRConnector",
    "PHIRedactor",
    "FHIRAuditLogger",
    # HL7 v2
    "HL7v2Connector",
    "HL7Parser",
    "HL7Message",
    "HL7Segment",
    "HL7PHIRedactor",
    "HL7MessageType",
    "HL7SegmentType",
    "MSHSegment",
    "PIDSegment",
    "PV1Segment",
    "OBXSegment",
    "ORCSegment",
    "OBRSegment",
    "SCHSegment",
    # EHR Adapters
    "EHRAdapter",
    "EHRVendor",
    "EHRCapability",
    "EHRConnectionConfig",
    "EpicAdapter",
    "CernerAdapter",
]
