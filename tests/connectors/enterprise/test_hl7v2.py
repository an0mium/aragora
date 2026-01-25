"""
Tests for HL7 v2.x Healthcare Connector.

Tests cover:
- HL7 message parsing (segments, fields, components)
- Typed segment extraction (MSH, PID, PV1, OBX, ORC, OBR, SCH)
- PHI redaction using Safe Harbor method
- MLLP framing encode/decode
- Message type filtering
- Connector sync functionality
"""

from datetime import datetime

import pytest


# =============================================================================
# HL7 Parser Tests
# =============================================================================


class TestHL7Field:
    """Tests for HL7Field parsing."""

    def test_simple_field(self):
        """Parse simple field value."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Field

        field = HL7Field.parse("simple_value")
        assert field.value == "simple_value"
        assert field.components == ["simple_value"]

    def test_field_with_components(self):
        """Parse field with components."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Field

        field = HL7Field.parse("Doe^John^M")
        assert field.value == "Doe^John^M"
        assert field.components == ["Doe", "John", "M"]
        assert field.get_component(1) == "Doe"
        assert field.get_component(2) == "John"
        assert field.get_component(3) == "M"
        assert field.get_component(4) == ""  # Default for missing

    def test_field_with_repetitions(self):
        """Parse field with repetitions."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Field

        field = HL7Field.parse("ID1^MRN~ID2^SSN")
        # First repetition's value is stored in value
        assert field.value == "ID1^MRN"
        assert field.components == ["ID1", "MRN"]
        assert len(field.repetitions) == 1
        assert field.repetitions[0].components == ["ID2", "SSN"]


class TestHL7Segment:
    """Tests for HL7Segment parsing."""

    def test_parse_simple_segment(self):
        """Parse simple segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Segment

        segment = HL7Segment.parse("EVN|A01|20230101120000")
        assert segment.segment_type == "EVN"
        assert segment.get_field_value(1) == "A01"
        assert segment.get_field_value(2) == "20230101120000"

    def test_parse_msh_segment(self):
        """Parse MSH segment with encoding characters."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Segment

        msh = "MSH|^~\\&|SENDER|FACILITY|RECEIVER|DEST|20230101120000||ADT^A01|12345|P|2.5.1"
        segment = HL7Segment.parse(msh)

        assert segment.segment_type == "MSH"
        assert segment.get_field_value(1) == "|"  # Field separator
        assert segment.get_field_value(2) == "^~\\&"  # Encoding chars
        assert segment.get_field_value(3) == "SENDER"
        assert segment.get_field_value(9) == "ADT^A01"

    def test_parse_pid_segment(self):
        """Parse PID segment with patient data."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Segment

        pid = "PID|1||123456^^^HOSPITAL^MR||DOE^JOHN^M||19800101|M|||123 MAIN ST^^CITY^ST^12345"
        segment = HL7Segment.parse(pid)

        assert segment.segment_type == "PID"
        assert segment.get_field_value(1) == "1"
        assert segment.get_component(3, 1) == "123456"  # Patient ID
        assert segment.get_component(5, 1) == "DOE"  # Last name
        assert segment.get_component(5, 2) == "JOHN"  # First name

    def test_get_component(self):
        """Get component from field."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Segment

        segment = HL7Segment.parse("OBX|1|NM|12345-6^CHOLESTEROL^LN||200|mg/dL")
        assert segment.get_component(3, 1) == "12345-6"
        assert segment.get_component(3, 2) == "CHOLESTEROL"
        assert segment.get_component(3, 3) == "LN"


class TestHL7Parser:
    """Tests for HL7Parser."""

    def test_parse_simple_message(self):
        """Parse simple ADT message."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|SENDER|FACILITY|RECEIVER|DEST|20230101120000||ADT^A01|12345|P|2.5.1
EVN|A01|20230101120000
PID|1||123456^^^HOSPITAL^MR||DOE^JOHN^M||19800101|M
PV1|1|I|ICU^01^01"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.message_type == "ADT^A01"
        assert result.message_control_id == "12345"
        assert len(result.segments) == 4

    def test_parse_msh_typed(self):
        """Parse MSH into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = "MSH|^~\\&|APP|FAC|RECV|DEST|20230615143022||ADT^A08|MSG001|P|2.5.1"
        parser = HL7Parser()
        result = parser.parse(message)

        assert result.msh is not None
        assert result.msh.sending_application == "APP"
        assert result.msh.sending_facility == "FAC"
        assert result.msh.receiving_application == "RECV"
        assert result.msh.message_type == "ADT^A08"
        assert result.msh.version_id == "2.5.1"
        assert result.msh.message_datetime == datetime(2023, 6, 15, 14, 30, 22)

    def test_parse_pid_typed(self):
        """Parse PID into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
PID|1||12345^^^HOSP^MR~98765^^^SSA^SS||SMITH^JANE^A||19850315|F|||100 OAK AVE^^BOSTON^MA^02101||5551234567||EN|M"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.pid is not None
        assert result.pid.patient_identifier_list == ["12345", "98765"]
        assert result.pid.patient_name == "SMITH^JANE^A"
        assert result.pid.date_of_birth == datetime(1985, 3, 15)
        assert result.pid.administrative_sex == "F"
        assert result.pid.primary_language == "EN"

    def test_parse_pv1_typed(self):
        """Parse PV1 into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
PV1|1|I|ICU^01^BED1|||||||CARDIO|||||||||VN123456"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.pv1 is not None
        assert result.pv1.patient_class == "I"  # Inpatient
        assert result.pv1.assigned_patient_location == "ICU^01^BED1"
        assert result.pv1.hospital_service == "CARDIO"
        assert result.pv1.visit_number == "VN123456"

    def test_parse_obx_typed(self):
        """Parse OBX into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|LAB|FAC|||20230101||ORU^R01|MSG001|P|2.5.1
OBX|1|NM|2093-3^CHOLESTEROL^LN||185|mg/dL|<200||||F
OBX|2|NM|2085-9^HDL^LN||55|mg/dL|>40||||F"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert len(result.obx_list) == 2
        assert result.obx_list[0].observation_identifier == "2093-3^CHOLESTEROL^LN"
        assert result.obx_list[0].observation_value == "185"
        assert result.obx_list[0].units == "mg/dL"
        assert result.obx_list[1].observation_identifier == "2085-9^HDL^LN"

    def test_parse_orc_typed(self):
        """Parse ORC into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        # ORC-12 is ordering_provider, so we need fields 1-12
        message = """MSH|^~\\&|ORDER|FAC|||20230101||ORM^O01|MSG001|P|2.5.1
ORC|NW|ORD001|FILL001|||CM||20230101120000||NURSE1|NURSE2|DR_SMITH"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.orc is not None
        assert result.orc.order_control == "NW"
        assert result.orc.placer_order_number == "ORD001"
        assert result.orc.filler_order_number == "FILL001"
        assert result.orc.ordering_provider == "DR_SMITH"

    def test_parse_sch_typed(self):
        """Parse SCH into typed segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|SCHED|FAC|||20230101||SIU^S12|MSG001|P|2.5.1
SCH|APT001|FILL001|||SCHED001||CHECKUP|ROUTINE|30|MIN"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.sch is not None
        assert result.sch.placer_appointment_id == "APT001"
        assert result.sch.appointment_reason == "CHECKUP"
        assert result.sch.appointment_type == "ROUTINE"
        assert result.sch.appointment_duration == "30"

    def test_parse_strict_mode(self):
        """Strict mode raises on invalid segments."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        parser = HL7Parser(strict=True)

        with pytest.raises(ValueError, match="must start with MSH"):
            parser.parse("INVALID|SEGMENT")

    def test_parse_non_strict_skips_bad_segments(self):
        """Non-strict mode skips invalid segments."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
BAD_SEGMENT_WITH_NO_PROPER_FORMAT
EVN|A01|20230101"""

        parser = HL7Parser(strict=False)
        result = parser.parse(message)

        # Should have MSH and EVN, skipping bad segment
        assert len(result.segments) >= 2


# =============================================================================
# MLLP Tests
# =============================================================================


class TestMLLP:
    """Tests for MLLP framing."""

    def test_parse_mllp_single_message(self):
        """Parse single MLLP-framed message."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import (
            HL7Parser,
            MLLP_START_BLOCK,
            MLLP_END_BLOCK,
            MLLP_CARRIAGE_RETURN,
        )

        raw = "MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1"
        mllp_data = MLLP_START_BLOCK + raw.encode() + MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN

        parser = HL7Parser()
        messages = parser.parse_mllp(mllp_data)

        assert len(messages) == 1
        assert messages[0].message_type == "ADT^A01"

    def test_encode_mllp(self):
        """Encode message with MLLP framing."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import (
            HL7Parser,
            MLLP_START_BLOCK,
            MLLP_END_BLOCK,
            MLLP_CARRIAGE_RETURN,
        )

        parser = HL7Parser()
        message = parser.parse("MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1")

        encoded = parser.encode_mllp(message)

        assert encoded.startswith(MLLP_START_BLOCK)
        assert encoded.endswith(MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN)


# =============================================================================
# PHI Redaction Tests
# =============================================================================


class TestHL7PHIRedactor:
    """Tests for HL7 PHI redaction."""

    def test_redact_patient_name(self):
        """Redact patient name from PID segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
PID|1||12345||DOE^JOHN^M||19800101|M"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        redactor = HL7PHIRedactor()
        result = redactor.redact(parsed)

        assert "[REDACTED-PATIENT_NAME]" in result.redacted_message
        assert "DOE^JOHN^M" not in result.redacted_message
        assert result.redactions_count > 0

    def test_redact_ssn(self):
        """Redact SSN from PID segment."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
PID|1||12345||DOE^JOHN||19800101|M|||||||||||123-45-6789"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        redactor = HL7PHIRedactor()
        result = redactor.redact(parsed)

        assert "123-45-6789" not in result.redacted_message
        assert "[REDACTED-SSN]" in result.redacted_message

    def test_redact_preserves_year(self):
        """Redact date but preserve year when configured."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
PID|1||12345||DOE^JOHN||19800315|M"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        redactor = HL7PHIRedactor(preserve_year=True)
        result = redactor.redact(parsed)

        # Year should be preserved, month/day should be 0101
        assert "19800101" in result.redacted_message
        assert "19800315" not in result.redacted_message

    def test_redact_phone_in_free_text(self):
        """Redact phone numbers in NTE segments."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor

        message = """MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1
NTE|1||Patient callback: 555-123-4567"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        redactor = HL7PHIRedactor(redact_free_text=True)
        result = redactor.redact(parsed)

        assert "555-123-4567" not in result.redacted_message
        assert "[REDACTED-PHONE]" in result.redacted_message

    def test_original_hash_preserved(self):
        """Original message hash is preserved for audit."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor
        import hashlib

        message = "MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1"
        expected_hash = hashlib.sha256(message.encode()).hexdigest()

        parser = HL7Parser()
        parsed = parser.parse(message)

        redactor = HL7PHIRedactor()
        result = redactor.redact(parsed)

        assert result.original_hash == expected_hash


# =============================================================================
# Connector Tests
# =============================================================================


class TestHL7v2Connector:
    """Tests for HL7v2Connector."""

    def test_connector_initialization(self):
        """Connector initializes with correct settings."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7v2Connector

        connector = HL7v2Connector(
            connector_id="test-hl7",
            tenant_id="hospital-1",
            source_type="file",
            message_types=["ADT", "ORU"],
            enable_phi_redaction=True,
        )

        assert connector.connector_id == "test-hl7"
        assert connector.tenant_id == "hospital-1"
        assert connector.hl7_source_type == "file"
        assert connector.message_types == {"ADT", "ORU"}
        assert connector.enable_phi_redaction is True

    def test_connector_stats(self):
        """Connector provides statistics."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7v2Connector

        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ADT"],
        )

        stats = connector.get_stats()

        assert stats["messages_processed"] == 0
        assert stats["phi_redaction_enabled"] is True
        assert stats["message_type_filter"] == ["ADT"]

    @pytest.mark.asyncio
    async def test_process_message(self):
        """Process a message into SyncItem."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7v2Connector, HL7Parser

        connector = HL7v2Connector(
            connector_id="test-hl7",
            enable_phi_redaction=True,
        )

        message = """MSH|^~\\&|SENDER|HOSPITAL|||20230615140000||ADT^A01|MSG12345|P|2.5.1
PID|1||PAT001||DOE^JOHN||19800101|M"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        item = await connector._process_message(parsed)

        assert item is not None
        assert item.id == "MSG12345"
        assert item.domain == "healthcare"
        assert item.metadata["message_type"] == "ADT^A01"
        assert item.metadata["phi_redacted"] is True

    @pytest.mark.asyncio
    async def test_message_type_filter(self):
        """Message type filtering works."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7v2Connector, HL7Parser

        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ORU"],  # Only accept ORU messages
        )

        message = "MSH|^~\\&|APP|FAC|||20230101||ADT^A01|MSG001|P|2.5.1"
        parser = HL7Parser()
        parsed = parser.parse(message)

        item = await connector._process_message(parsed)

        assert item is None  # ADT should be filtered out


# =============================================================================
# Message Type Tests
# =============================================================================


class TestHL7MessageTypes:
    """Tests for HL7 message type enums."""

    def test_message_type_values(self):
        """Message type enum has expected values."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7MessageType

        assert HL7MessageType.ADT.value == "ADT"
        assert HL7MessageType.ADT_A01.value == "ADT^A01"
        assert HL7MessageType.ORU_R01.value == "ORU^R01"
        assert HL7MessageType.ORM_O01.value == "ORM^O01"
        assert HL7MessageType.SIU_S12.value == "SIU^S12"

    def test_segment_type_values(self):
        """Segment type enum has expected values."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7SegmentType

        assert HL7SegmentType.MSH.value == "MSH"
        assert HL7SegmentType.PID.value == "PID"
        assert HL7SegmentType.PV1.value == "PV1"
        assert HL7SegmentType.OBX.value == "OBX"
        assert HL7SegmentType.ORC.value == "ORC"
        assert HL7SegmentType.SCH.value == "SCH"


# =============================================================================
# Integration Tests
# =============================================================================


class TestHL7Integration:
    """Integration tests for complete HL7 workflows."""

    def test_parse_complete_adt_a01(self):
        """Parse complete ADT^A01 admit message."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|ADT|MERCY|RECV|HIS|20230615143022||ADT^A01|20230615143022001|P|2.5.1
EVN|A01|20230615143022
PID|1||MRN001^^^MERCY^MR||SMITH^JANE^A||19850315|F||C|123 MAIN ST^^BOSTON^MA^02101||5551234567||EN|M|CHR|ACCT001||SSN001
PV1|1|I|ICU^01^BED1|||||||CARDIO|||||||||VN001|||||||||||||||||||||||20230615140000"""

        parser = HL7Parser()
        result = parser.parse(message)

        # Verify MSH
        assert result.msh.sending_application == "ADT"
        assert result.msh.sending_facility == "MERCY"
        assert result.msh.message_type == "ADT^A01"

        # Verify PID
        assert result.pid.patient_identifier_list == ["MRN001"]
        assert result.pid.patient_name == "SMITH^JANE^A"
        assert result.pid.administrative_sex == "F"

        # Verify PV1
        assert result.pv1.patient_class == "I"
        assert result.pv1.assigned_patient_location == "ICU^01^BED1"

    def test_parse_complete_oru_r01(self):
        """Parse complete ORU^R01 lab result message."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser

        message = """MSH|^~\\&|LAB|HOSP|||20230615150000||ORU^R01|LAB123|P|2.5.1
PID|1||PAT001||DOE^JOHN||19600101|M
OBR|1|ORD001|FILL001|CBC^COMPLETE BLOOD COUNT^L|||20230615143000
OBX|1|NM|WBC^WHITE BLOOD CELLS^L||7.5|10^9/L|4.5-11.0||||F
OBX|2|NM|RBC^RED BLOOD CELLS^L||4.8|10^12/L|4.5-5.5||||F
OBX|3|NM|HGB^HEMOGLOBIN^L||14.2|g/dL|13.0-17.0||||F"""

        parser = HL7Parser()
        result = parser.parse(message)

        assert result.message_type == "ORU^R01"
        assert result.obr.universal_service_id == "CBC^COMPLETE BLOOD COUNT^L"
        assert len(result.obx_list) == 3
        assert result.obx_list[0].observation_value == "7.5"
        assert result.obx_list[2].observation_identifier == "HGB^HEMOGLOBIN^L"

    def test_full_redaction_workflow(self):
        """Test full PHI redaction workflow."""
        from aragora.connectors.enterprise.healthcare.hl7v2 import HL7Parser, HL7PHIRedactor

        # PID fields: 1=SetID, 3=MRN, 5=Name, 7=DOB, 8=Sex, 11=Address, 13=Phone, 19=SSN
        # 5 empty fields between phone(13) and SSN(19): 14,15,16,17,18
        message = """MSH|^~\\&|ADT|HOSPITAL|||20230615||ADT^A01|MSG001|P|2.5.1
PID|1||MRN123||JONES^MARY^B||19750420|F|||456 ELM ST^^CHICAGO^IL^60601||3125551234|||||999-88-7777
NTE|1||Patient prefers callback at 312-555-9999"""

        parser = HL7Parser()
        parsed = parser.parse(message)

        # Original should contain PHI
        assert "JONES^MARY^B" in message
        assert "999-88-7777" in message
        assert "312-555-9999" in message

        redactor = HL7PHIRedactor()
        result = redactor.redact(parsed)

        # Redacted should not contain PHI
        assert "JONES^MARY^B" not in result.redacted_message
        assert "999-88-7777" not in result.redacted_message
        assert "312-555-9999" not in result.redacted_message

        # Should have redaction markers
        assert "[REDACTED-" in result.redacted_message
        assert result.redactions_count >= 3
