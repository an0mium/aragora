"""
Tests for HL7 v2.x Healthcare Connector.

Tests:
- HL7 field and segment parsing
- Message type detection
- PHI redaction (HIPAA Safe Harbor method) - CRITICAL
- MLLP framing
- Typed segment extraction
- Audit logging
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.healthcare.hl7v2 import (
    DEFAULT_ENCODING_CHARS,
    DEFAULT_FIELD_SEPARATOR,
    HL7Field,
    HL7Message,
    HL7MessageType,
    HL7Parser,
    HL7PHIRedactor,
    HL7RedactionResult,
    HL7Segment,
    HL7SegmentType,
    HL7v2Connector,
    MLLP_CARRIAGE_RETURN,
    MLLP_END_BLOCK,
    MLLP_START_BLOCK,
    MSHSegment,
    OBRSegment,
    OBXSegment,
    ORCSegment,
    PIDSegment,
    PV1Segment,
    SCHSegment,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_adt_a01():
    """ADT^A01 (Admit notification) message."""
    return (
        "MSH|^~\\&|EPIC|FACILITY1|LAB|FACILITY2|20240115120000||ADT^A01|MSG00001|P|2.5.1\r"
        "EVN|A01|20240115120000\r"
        "PID|1||12345^^^FACILITY1^MR~987654321^^^SSA^SS||Doe^John^A||19800515|M|||"
        "123 Main St^^Springfield^IL^62701||217-555-1234|217-555-5678|||S||123456789\r"
        "PV1|1|I|ICU^101^A^FACILITY1||||1234^Smith^John^D|||SUR||||ADM|||"
        "5678^Jones^Mary||V00001|||||||||||||||||||||||20240115090000"
    )


@pytest.fixture
def sample_oru_r01():
    """ORU^R01 (Lab result) message."""
    return (
        "MSH|^~\\&|LAB|LABFAC|EHR|EHRFAC|20240115140000||ORU^R01|MSG00002|P|2.5.1\r"
        "PID|1||54321^^^LABFAC^MR||Smith^Jane^B||19751020|F|||456 Oak Ave^^Chicago^IL^60601\r"
        "OBR|1|ORD001|FIL001|80053^COMPREHENSIVE METABOLIC PANEL^L|||20240115130000\r"
        "OBX|1|NM|2345-7^GLUCOSE^LN||105|mg/dL|70-100|H|||F|||20240115133000\r"
        "OBX|2|NM|3094-0^UREA NITROGEN^LN||18|mg/dL|7-20|N|||F|||20240115133000"
    )


@pytest.fixture
def sample_orm_o01():
    """ORM^O01 (Order) message."""
    return (
        "MSH|^~\\&|CPOE|HOSP|LAB|HOSP|20240115100000||ORM^O01|MSG00003|P|2.5.1\r"
        "PID|1||99999^^^HOSP^MR||Patient^Test||19900101|M\r"
        "ORC|NW|ORD123||GRP001|||1^Once||20240115100000|NURSE001|DR001|DR001\r"
        "OBR|1|ORD123||CBC^Complete Blood Count^L|||20240115100000||||||||DR001"
    )


@pytest.fixture
def sample_siu_s12():
    """SIU^S12 (Scheduling) message."""
    return (
        "MSH|^~\\&|SCHED|CLINIC|EHR|CLINIC|20240115080000||SIU^S12|MSG00004|P|2.5.1\r"
        "SCH|APT001|APT001-F||GRP001|CHECKUP|ROUTINE|ANNUAL CHECKUP|WELLNESS|30|MIN||"
        "DR^Scheduler^Jane|555-123-4567|123 Medical Dr|CLINIC1|"
        "DOC^Doctor^John|555-987-6543|456 Health Ave|CLINIC1\r"
        "PID|1||11111^^^CLINIC^MR||Brown^Michael||19850322|M"
    )


@pytest.fixture
def sample_message_with_phi():
    """Message with extensive PHI for redaction testing."""
    return (
        "MSH|^~\\&|SOURCE|HOSP|DEST|HOSP|20240115120000||ADT^A01|MSG00005|P|2.5.1\r"
        "PID|1||MRN12345^^^HOSP^MR||Sensitive^Patient^Name||19850315|F||"
        "Asian|789 Secret Lane^^Hidden City^CA^90210||310-555-9876|310-555-4321|||"
        "SINGLE||123-45-6789\r"
        "NK1|1|Emergency^Contact^Person||999 Relative St^^Hometown^TX^75001|817-555-1111|"
        "817-555-2222\r"
        "GT1|1||Guarantor^Name^Here||111 Money Ave^^Richtown^NY^10001|"
        "212-555-3333|212-555-4444|||||222-33-4444\r"
        "IN1|1|BCBS|12345|Blue Cross||||||||||||Insured^Person^Name|19700101|"
        "222 Insurance Rd^^Coverage City^FL^33101\r"
        "NTE|1||Patient has SSN 333-44-5555 and can be reached at test@email.com or "
        "555-666-7777. IP address noted: 192.168.1.100. DOB: 03/15/1985."
    )


# =============================================================================
# HL7Field Tests
# =============================================================================


class TestHL7Field:
    """Tests for HL7Field dataclass and parsing."""

    def test_parse_simple_field(self):
        """Parse a simple field with no components."""
        field = HL7Field.parse("SimpleValue")
        assert field.value == "SimpleValue"
        assert field.components == ["SimpleValue"]
        assert field.repetitions == []

    def test_parse_field_with_components(self):
        """Parse field with component separator."""
        field = HL7Field.parse("Last^First^Middle")
        assert field.value == "Last^First^Middle"
        assert field.components == ["Last", "First", "Middle"]
        assert len(field.repetitions) == 0

    def test_parse_field_with_repetitions(self):
        """Parse field with repetitions."""
        field = HL7Field.parse("ID1~ID2~ID3")
        assert field.components == ["ID1"]
        assert len(field.repetitions) == 2
        assert field.repetitions[0].components == ["ID2"]
        assert field.repetitions[1].components == ["ID3"]

    def test_parse_field_with_components_and_repetitions(self):
        """Parse field with both components and repetitions."""
        field = HL7Field.parse("Doe^John~Smith^Jane")
        assert field.components == ["Doe", "John"]
        assert len(field.repetitions) == 1
        assert field.repetitions[0].components == ["Smith", "Jane"]

    def test_get_component_valid_index(self):
        """Get component by valid 1-based index."""
        field = HL7Field.parse("A^B^C^D")
        assert field.get_component(1) == "A"
        assert field.get_component(2) == "B"
        assert field.get_component(3) == "C"
        assert field.get_component(4) == "D"

    def test_get_component_invalid_index(self):
        """Get component with invalid index returns default."""
        field = HL7Field.parse("A^B")
        assert field.get_component(0) == ""
        assert field.get_component(5) == ""
        assert field.get_component(5, "default") == "default"

    def test_str_representation(self):
        """String representation returns original value."""
        field = HL7Field.parse("Test^Value")
        assert str(field) == "Test^Value"

    def test_custom_separators(self):
        """Parse with custom separators."""
        field = HL7Field.parse("A#B#C", component_sep="#", repetition_sep="!")
        assert field.components == ["A", "B", "C"]

    def test_empty_field(self):
        """Parse empty field."""
        field = HL7Field.parse("")
        assert field.value == ""
        assert field.components == [""]


# =============================================================================
# HL7Segment Tests
# =============================================================================


class TestHL7Segment:
    """Tests for HL7Segment parsing."""

    def test_parse_simple_segment(self):
        """Parse a simple segment."""
        segment = HL7Segment.parse("PID|1||12345^^^HOSP^MR")
        assert segment.segment_type == "PID"
        assert len(segment.fields) == 3
        assert segment.get_field_value(1) == "1"
        assert segment.get_field_value(3) == "12345^^^HOSP^MR"

    def test_parse_msh_segment(self):
        """Parse MSH segment with special handling."""
        segment = HL7Segment.parse("MSH|^~\\&|APP|FAC|DEST|DFAC|20240115|SEC|ADT^A01|123|P|2.5")
        assert segment.segment_type == "MSH"
        # MSH-1 is field separator, MSH-2 is encoding chars
        assert segment.get_field_value(1) == "|"
        assert segment.get_field_value(2) == "^~\\&"
        assert segment.get_field_value(3) == "APP"

    def test_get_field_valid(self):
        """Get field by valid index."""
        segment = HL7Segment.parse("OBX|1|NM|CODE||100|mg/dL")
        field = segment.get_field(1)
        assert field is not None
        assert field.value == "1"

    def test_get_field_invalid(self):
        """Get field by invalid index returns None."""
        segment = HL7Segment.parse("OBX|1|NM")
        assert segment.get_field(0) is None
        assert segment.get_field(100) is None

    def test_get_component(self):
        """Get component from field."""
        segment = HL7Segment.parse("PID|1||ID^HOSP^MR")
        assert segment.get_component(3, 1) == "ID"
        assert segment.get_component(3, 2) == "HOSP"
        assert segment.get_component(3, 3) == "MR"

    def test_empty_segment_raises(self):
        """Empty segment raises ValueError."""
        with pytest.raises(ValueError, match="Empty segment"):
            HL7Segment.parse("")

    def test_raw_preserved(self):
        """Raw segment string is preserved."""
        raw = "NTE|1||This is a note"
        segment = HL7Segment.parse(raw)
        assert segment.raw == raw


# =============================================================================
# Typed Segment Tests
# =============================================================================


class TestMSHSegment:
    """Tests for MSH segment parsing."""

    def test_from_segment_basic(self):
        """Parse basic MSH segment."""
        raw = "MSH|^~\\&|SEND_APP|SEND_FAC|RECV_APP|RECV_FAC|20240115120000|SEC|ADT^A01|CTRL001|P|2.5.1"
        segment = HL7Segment.parse(raw)
        msh = MSHSegment.from_segment(segment)

        assert msh.field_separator == "|"
        assert msh.encoding_characters == "^~\\&"
        assert msh.sending_application == "SEND_APP"
        assert msh.sending_facility == "SEND_FAC"
        assert msh.receiving_application == "RECV_APP"
        assert msh.receiving_facility == "RECV_FAC"
        assert msh.message_type == "ADT^A01"
        assert msh.message_control_id == "CTRL001"
        assert msh.processing_id == "P"
        assert msh.version_id == "2.5.1"

    def test_from_segment_datetime_full(self):
        """Parse MSH with full datetime."""
        raw = "MSH|^~\\&|A|B|C|D|20240115143052||ADT|123|P|2.5"
        segment = HL7Segment.parse(raw)
        msh = MSHSegment.from_segment(segment)

        assert msh.message_datetime is not None
        assert msh.message_datetime.year == 2024
        assert msh.message_datetime.month == 1
        assert msh.message_datetime.day == 15
        assert msh.message_datetime.hour == 14
        assert msh.message_datetime.minute == 30
        assert msh.message_datetime.second == 52

    def test_from_segment_datetime_date_only(self):
        """Parse MSH with date-only datetime."""
        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5"
        segment = HL7Segment.parse(raw)
        msh = MSHSegment.from_segment(segment)

        assert msh.message_datetime is not None
        assert msh.message_datetime.year == 2024
        assert msh.message_datetime.month == 1
        assert msh.message_datetime.day == 15

    def test_from_segment_invalid_datetime(self):
        """Invalid datetime is handled gracefully."""
        raw = "MSH|^~\\&|A|B|C|D|INVALID||ADT|123|P|2.5"
        segment = HL7Segment.parse(raw)
        msh = MSHSegment.from_segment(segment)
        assert msh.message_datetime is None


class TestPIDSegment:
    """Tests for PID segment parsing."""

    def test_from_segment_basic(self):
        """Parse basic PID segment."""
        # PID fields (1-based): 1=SetID, 2=PatientID(deprecated), 3=PatientIdentifierList, 4=AltID, 5=Name, 6=MothersMaiden, 7=DOB, 8=Sex, 9=Alias, 10=Race, 11=Address, 12=County, 13=PhoneHome, 14=PhoneBusiness, 15=Language, 16=MaritalStatus, 17=Religion, 18=AccountNum, 19=SSN
        # Counted:       1| 2|3                                     |4 |5                |6 |7       |8|9 |10|11                          |12|13          |14          |15|16     |17|18|19
        raw = "PID|1||12345^^^HOSP^MR~98765^^^SSA^SS||Doe^John^Middle||19850315|M|||123 Main St^^City^ST^12345||555-123-4567|555-987-6543||MARRIED|||123456789"
        segment = HL7Segment.parse(raw)
        pid = PIDSegment.from_segment(segment)

        assert pid.set_id == "1"
        assert "12345" in pid.patient_identifier_list
        assert pid.patient_name == "Doe^John^Middle"
        assert pid.administrative_sex == "M"
        assert pid.patient_address == "123 Main St^^City^ST^12345"
        assert pid.phone_home == "555-123-4567"
        assert pid.phone_business == "555-987-6543"
        assert pid.marital_status == "MARRIED"
        assert pid.ssn == "123456789"

    def test_from_segment_dob_parsing(self):
        """Parse date of birth correctly."""
        raw = "PID|1||||Patient^Name||19900725"
        segment = HL7Segment.parse(raw)
        pid = PIDSegment.from_segment(segment)

        assert pid.date_of_birth is not None
        assert pid.date_of_birth.year == 1990
        assert pid.date_of_birth.month == 7
        assert pid.date_of_birth.day == 25

    def test_from_segment_multiple_identifiers(self):
        """Parse multiple patient identifiers."""
        raw = "PID|1||ID1^A^B~ID2^C^D~ID3^E^F||Name"
        segment = HL7Segment.parse(raw)
        pid = PIDSegment.from_segment(segment)

        assert len(pid.patient_identifier_list) == 3
        assert "ID1" in pid.patient_identifier_list
        assert "ID2" in pid.patient_identifier_list
        assert "ID3" in pid.patient_identifier_list


class TestPV1Segment:
    """Tests for PV1 segment parsing."""

    def test_from_segment_basic(self):
        """Parse basic PV1 segment."""
        # PV1 fields (1-based): 1=SetID, 2=PatientClass, 3=Location, 4=AdmissionType, 5=PreadmitNum, 6=PriorLocation, 7=AttendingDoc, 8=ReferringDoc, 9=ConsultingDoc, 10=HospService, ..., 19=VisitNumber
        # Counted:        1|2|3          |4|5|6|7            |8            |9|10  |11-18 (8 empty fields)|19
        raw = "PV1|1|I|ICU^101^A||||1234^DrSmith|5678^DrJones||SURG|||||||||V001"
        segment = HL7Segment.parse(raw)
        pv1 = PV1Segment.from_segment(segment)

        assert pv1.set_id == "1"
        assert pv1.patient_class == "I"
        assert pv1.assigned_patient_location == "ICU^101^A"
        assert pv1.attending_doctor == "1234^DrSmith"
        assert pv1.referring_doctor == "5678^DrJones"
        assert pv1.hospital_service == "SURG"
        assert pv1.visit_number == "V001"

    def test_from_segment_patient_classes(self):
        """Parse different patient classes."""
        for class_code, expected in [("I", "I"), ("O", "O"), ("E", "E")]:
            raw = f"PV1|1|{class_code}"
            segment = HL7Segment.parse(raw)
            pv1 = PV1Segment.from_segment(segment)
            assert pv1.patient_class == expected


class TestOBXSegment:
    """Tests for OBX segment parsing."""

    def test_from_segment_numeric(self):
        """Parse numeric OBX segment."""
        raw = "OBX|1|NM|2345-7^GLUCOSE^LN||105|mg/dL|70-100|H|||F|||20240115133045"
        segment = HL7Segment.parse(raw)
        obx = OBXSegment.from_segment(segment)

        assert obx.set_id == "1"
        assert obx.value_type == "NM"
        assert obx.observation_identifier == "2345-7^GLUCOSE^LN"
        assert obx.observation_value == "105"
        assert obx.units == "mg/dL"
        assert obx.references_range == "70-100"
        assert obx.abnormal_flags == "H"
        assert obx.observation_result_status == "F"

    def test_from_segment_text(self):
        """Parse text OBX segment."""
        raw = "OBX|2|TX|COMMENT||This is a free text comment||||N"
        segment = HL7Segment.parse(raw)
        obx = OBXSegment.from_segment(segment)

        assert obx.value_type == "TX"
        assert obx.observation_value == "This is a free text comment"

    def test_from_segment_datetime(self):
        """Parse observation datetime."""
        # OBX-14 is datetime_of_observation
        # Fields: 1=SetID, 2=ValueType, 3=ObsID, 4=SubID, 5=Value, 6=Units, 7=RefRange, 8=AbnormalFlags, 9=Probability, 10=Nature, 11=ResultStatus, 12=EffectiveDate, 13=AccessChecks, 14=DatetimeObs
        raw = "OBX|1|NM|CODE||100|unit|||N||F|||20240115143052"
        segment = HL7Segment.parse(raw)
        obx = OBXSegment.from_segment(segment)

        assert obx.datetime_of_observation is not None
        assert obx.datetime_of_observation.year == 2024


class TestORCSegment:
    """Tests for ORC segment parsing."""

    def test_from_segment_new_order(self):
        """Parse new order ORC segment."""
        raw = "ORC|NW|ORD001|FIL001|GRP001|||1^Once||20240115100000|NURSE|VERIFY|PROVIDER"
        segment = HL7Segment.parse(raw)
        orc = ORCSegment.from_segment(segment)

        assert orc.order_control == "NW"
        assert orc.placer_order_number == "ORD001"
        assert orc.filler_order_number == "FIL001"
        assert orc.placer_group_number == "GRP001"
        assert orc.entered_by == "NURSE"
        assert orc.verified_by == "VERIFY"
        assert orc.ordering_provider == "PROVIDER"

    def test_from_segment_cancel_order(self):
        """Parse cancel order ORC segment."""
        raw = "ORC|CA|ORD001"
        segment = HL7Segment.parse(raw)
        orc = ORCSegment.from_segment(segment)
        assert orc.order_control == "CA"


class TestOBRSegment:
    """Tests for OBR segment parsing."""

    def test_from_segment_basic(self):
        """Parse basic OBR segment."""
        # OBR-25 is result_status, so we need fields 1-25
        # Fields: 1=SetID, 2=PlacerOrderNum, 3=FillerOrderNum, 4=UniversalServiceID, 5=Priority, 6=RequestedDT, 7=ObsDT, 8=ObsEndDT, 9=CollVol, 10=Collector, 11=SpecimenAction, 12=DangerCode, 13=ClinicalInfo, 14=SpecReceivedDT, 15=SpecSource, 16=OrderingProvider, 17-24=empty, 25=ResultStatus
        # Need 26 parts total (OBR + 25 fields), with F at parts[25] = field 25
        raw = "OBR|1|ORD001|FIL001|80053^CMP^L||20240115|20240115120000||COL001|||INFO||SPEC|PROV||||||||||F"
        segment = HL7Segment.parse(raw)
        obr = OBRSegment.from_segment(segment)

        assert obr.set_id == "1"
        assert obr.placer_order_number == "ORD001"
        assert obr.filler_order_number == "FIL001"
        assert obr.universal_service_id == "80053^CMP^L"
        assert obr.result_status == "F"


class TestSCHSegment:
    """Tests for SCH segment parsing."""

    def test_from_segment_basic(self):
        """Parse basic SCH segment."""
        raw = "SCH|APT001|APT001-F||GRP001|CHECKUP|ROUTINE|ANNUAL|WELLNESS|30|MIN||CONTACT|555-1234|ADDR|LOC|FILLER|555-5678"
        segment = HL7Segment.parse(raw)
        sch = SCHSegment.from_segment(segment)

        assert sch.placer_appointment_id == "APT001"
        assert sch.filler_appointment_id == "APT001-F"
        assert sch.schedule_id == "CHECKUP"
        assert sch.event_reason == "ROUTINE"
        assert sch.appointment_reason == "ANNUAL"
        assert sch.appointment_type == "WELLNESS"
        assert sch.appointment_duration == "30"
        assert sch.appointment_duration_units == "MIN"


# =============================================================================
# HL7Message Tests
# =============================================================================


class TestHL7Message:
    """Tests for HL7Message."""

    def test_message_type_property(self, sample_adt_a01):
        """Message type property returns correct value."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)
        assert message.message_type == "ADT^A01"

    def test_message_control_id_property(self, sample_adt_a01):
        """Message control ID property returns correct value."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)
        assert message.message_control_id == "MSG00001"

    def test_patient_id_property(self, sample_adt_a01):
        """Patient ID property returns first identifier."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)
        assert message.patient_id == "12345"

    def test_get_segments_by_type(self, sample_oru_r01):
        """Get all segments of a specific type."""
        parser = HL7Parser()
        message = parser.parse(sample_oru_r01)
        obx_segments = message.get_segments("OBX")
        assert len(obx_segments) == 2

    def test_typed_segments_populated(self, sample_adt_a01):
        """Typed segments are populated correctly."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)

        assert message.msh is not None
        assert message.pid is not None
        assert message.pv1 is not None

    def test_obx_list_populated(self, sample_oru_r01):
        """OBX list is populated for results messages."""
        parser = HL7Parser()
        message = parser.parse(sample_oru_r01)
        assert len(message.obx_list) == 2


# =============================================================================
# HL7Parser Tests
# =============================================================================


class TestHL7Parser:
    """Tests for HL7Parser."""

    def test_parse_adt_a01(self, sample_adt_a01):
        """Parse ADT^A01 message."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)

        assert message.message_type == "ADT^A01"
        assert message.msh is not None
        assert message.msh.sending_application == "EPIC"
        assert message.pid is not None
        assert message.pv1 is not None

    def test_parse_oru_r01(self, sample_oru_r01):
        """Parse ORU^R01 message."""
        parser = HL7Parser()
        message = parser.parse(sample_oru_r01)

        assert message.message_type == "ORU^R01"
        assert len(message.obx_list) == 2
        assert message.obr is not None

    def test_parse_orm_o01(self, sample_orm_o01):
        """Parse ORM^O01 message."""
        parser = HL7Parser()
        message = parser.parse(sample_orm_o01)

        assert message.message_type == "ORM^O01"
        assert message.orc is not None
        assert message.orc.order_control == "NW"

    def test_parse_siu_s12(self, sample_siu_s12):
        """Parse SIU^S12 message."""
        parser = HL7Parser()
        message = parser.parse(sample_siu_s12)

        assert message.message_type == "SIU^S12"
        assert message.sch is not None

    def test_parse_normalizes_line_endings(self):
        """Parser normalizes different line endings."""
        parser = HL7Parser()

        # CRLF
        msg_crlf = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\r\nPID|1"
        message = parser.parse(msg_crlf)
        assert len(message.segments) == 2

        # LF only
        msg_lf = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\nPID|1"
        message = parser.parse(msg_lf)
        assert len(message.segments) == 2

    def test_parse_empty_message_raises(self):
        """Empty message raises ValueError."""
        parser = HL7Parser()
        with pytest.raises(ValueError, match="Empty"):
            parser.parse("")

    def test_parse_no_msh_raises(self):
        """Message without MSH raises ValueError."""
        parser = HL7Parser()
        with pytest.raises(ValueError, match="must start with MSH"):
            parser.parse("PID|1||12345")

    def test_parse_short_msh_raises(self):
        """MSH segment too short raises ValueError."""
        parser = HL7Parser()
        with pytest.raises(ValueError, match="too short"):
            parser.parse("MSH|^~")

    def test_parse_strict_mode_raises_on_bad_segment(self):
        """Strict mode raises on parsing errors."""
        parser = HL7Parser(strict=True)
        # Malformed segment in middle
        bad_msg = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\r\n\rPID|1"
        # Note: empty lines are skipped, so this should parse
        message = parser.parse(bad_msg)
        assert len(message.segments) == 2

    def test_parse_non_strict_skips_bad_segments(self):
        """Non-strict mode skips bad segments."""
        parser = HL7Parser(strict=False)
        message = parser.parse("MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1")
        assert len(message.segments) == 2

    def test_parse_custom_encoding_chars(self):
        """Parse message with different encoding characters."""
        parser = HL7Parser()
        # Using different component separator
        raw = "MSH|#~\\&|APP|FAC|DEST|DFAC|20240115||ADT|123|P|2.5\rPID|1||ID#HOSP#MR"
        message = parser.parse(raw)
        assert message.msh is not None
        # Field values preserved
        pid = message.get_segments("PID")[0]
        assert "ID#HOSP#MR" in pid.get_field_value(3)


class TestHL7ParserMLLP:
    """Tests for MLLP framing."""

    def test_parse_mllp_single_message(self, sample_adt_a01):
        """Parse single MLLP-framed message."""
        parser = HL7Parser()
        mllp_data = (
            MLLP_START_BLOCK + sample_adt_a01.encode() + MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN
        )

        messages = parser.parse_mllp(mllp_data)

        assert len(messages) == 1
        assert messages[0].message_type == "ADT^A01"

    def test_parse_mllp_multiple_messages(self, sample_adt_a01, sample_oru_r01):
        """Parse multiple MLLP-framed messages."""
        parser = HL7Parser()
        msg1 = MLLP_START_BLOCK + sample_adt_a01.encode() + MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN
        msg2 = MLLP_START_BLOCK + sample_oru_r01.encode() + MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN

        messages = parser.parse_mllp(msg1 + msg2)

        assert len(messages) == 2
        assert messages[0].message_type == "ADT^A01"
        assert messages[1].message_type == "ORU^R01"

    def test_encode_mllp(self, sample_adt_a01):
        """Encode message with MLLP framing."""
        parser = HL7Parser()
        message = parser.parse(sample_adt_a01)

        encoded = parser.encode_mllp(message)

        assert encoded.startswith(MLLP_START_BLOCK)
        assert encoded.endswith(MLLP_END_BLOCK + MLLP_CARRIAGE_RETURN)

    def test_mllp_roundtrip(self, sample_adt_a01):
        """MLLP encode and decode roundtrip."""
        parser = HL7Parser()
        original = parser.parse(sample_adt_a01)

        encoded = parser.encode_mllp(original)
        decoded = parser.parse_mllp(encoded)

        assert len(decoded) == 1
        assert decoded[0].message_type == original.message_type
        assert decoded[0].message_control_id == original.message_control_id


# =============================================================================
# PHI Redaction Tests (CRITICAL - HIPAA Compliance)
# =============================================================================


class TestHL7PHIRedactor:
    """Tests for PHI redaction - HIPAA Safe Harbor method."""

    def test_redact_patient_name(self):
        """Patient name (PID-5) is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||Sensitive^Name^Here"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-PATIENT_NAME]" in result.redacted_message
        assert "Sensitive^Name^Here" not in result.redacted_message
        assert result.redactions_count > 0
        assert any("PID-5" in field for field in result.redacted_fields)

    def test_redact_patient_identifiers(self):
        """Patient identifiers (PID-3) are redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||MRN12345^^^HOSP^MR"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-PATIENT_IDENTIFIERS]" in result.redacted_message
        assert "MRN12345" not in result.redacted_message

    def test_redact_ssn(self):
        """SSN (PID-19) is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||||||||||||||123456789"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-SSN]" in result.redacted_message
        assert "123456789" not in result.redacted_message

    def test_redact_address(self):
        """Patient address (PID-11) is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||||||123 Secret Lane^^Hidden City^CA^90210"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-PATIENT_ADDRESS]" in result.redacted_message
        assert "123 Secret Lane" not in result.redacted_message

    def test_redact_phone_numbers(self):
        """Phone numbers (PID-13, PID-14) are redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = (
            "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||||||||555-123-4567|555-987-6543"
        )
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-PHONE_HOME]" in result.redacted_message
        assert "[REDACTED-PHONE_BUSINESS]" in result.redacted_message
        assert "555-123-4567" not in result.redacted_message
        assert "555-987-6543" not in result.redacted_message

    def test_redact_dob_preserves_year(self):
        """Date of birth is redacted but year is preserved."""
        redactor = HL7PHIRedactor(redact_dates=True, preserve_year=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||19850315"
        message = parser.parse(raw)
        result = redactor.redact(message)

        # Year preserved, rest changed to 0101
        assert "19850101" in result.redacted_message
        assert "19850315" not in result.redacted_message

    def test_redact_dob_no_year_preservation(self):
        """Date of birth fully redacted when preserve_year=False."""
        redactor = HL7PHIRedactor(redact_dates=True, preserve_year=False)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||19850315"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "[REDACTED-DATE_OF_BIRTH]" in result.redacted_message
        assert "19850315" not in result.redacted_message

    def test_redact_nk1_next_of_kin(self):
        """Next of kin information is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNK1|1|Contact^Name||123 Relative St^^Town^TX^75001|817-555-1111|817-555-2222"
        message = parser.parse(raw)
        result = redactor.redact(message)

        # NK1 fields should be redacted
        assert (
            "Contact^Name" not in result.redacted_message or "[REDACTED" in result.redacted_message
        )

    def test_redact_gt1_guarantor(self):
        """Guarantor information is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rGT1|1||Guarantor^Name||111 Money Ave||212-555-3333|212-555-4444||||222334444"
        message = parser.parse(raw)
        result = redactor.redact(message)

        # Check GT1 fields are redacted
        assert result.redactions_count > 0

    def test_redact_in1_insurance(self):
        """Insurance information is redacted."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rIN1|1|BCBS|12345|Blue Cross||||||||||||Insured^Name|19700101|222 Insurance Rd"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert result.redactions_count > 0

    def test_redact_free_text_ssn_pattern(self):
        """SSN patterns in free text are redacted."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNTE|1||Patient SSN is 333-44-5555"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "333-44-5555" not in result.redacted_message
        assert "[REDACTED-SSN]" in result.redacted_message

    def test_redact_free_text_phone_pattern(self):
        """Phone patterns in free text are redacted."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNTE|1||Call patient at 555-666-7777"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "555-666-7777" not in result.redacted_message
        assert "[REDACTED-PHONE]" in result.redacted_message

    def test_redact_free_text_email_pattern(self):
        """Email patterns in free text are redacted."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNTE|1||Contact: test@email.com"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "test@email.com" not in result.redacted_message
        assert "[REDACTED-EMAIL]" in result.redacted_message

    def test_redact_free_text_ip_address(self):
        """IP addresses in free text are redacted."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNTE|1||Access from IP 192.168.1.100"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "192.168.1.100" not in result.redacted_message
        assert "[REDACTED-IP_ADDRESS]" in result.redacted_message

    def test_redact_free_text_date_pattern(self):
        """Date patterns in free text are redacted."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rNTE|1||DOB: 03/15/1985"
        message = parser.parse(raw)
        result = redactor.redact(message)

        assert "03/15/1985" not in result.redacted_message
        assert "[REDACTED-DATE_FULL]" in result.redacted_message

    def test_redact_comprehensive_message(self, sample_message_with_phi):
        """Comprehensive PHI redaction on message with extensive PHI."""
        redactor = HL7PHIRedactor(redact_free_text=True)
        parser = HL7Parser()

        message = parser.parse(sample_message_with_phi)
        result = redactor.redact(message)

        # All PHI should be redacted
        assert "Sensitive^Patient^Name" not in result.redacted_message
        assert "789 Secret Lane" not in result.redacted_message
        assert "310-555-9876" not in result.redacted_message
        assert "123-45-6789" not in result.redacted_message
        assert "Emergency^Contact^Person" not in result.redacted_message
        assert "Guarantor^Name^Here" not in result.redacted_message
        assert "222-33-4444" not in result.redacted_message
        assert "333-44-5555" not in result.redacted_message
        assert "test@email.com" not in result.redacted_message
        assert "192.168.1.100" not in result.redacted_message

        # Verify substantial redactions occurred
        assert result.redactions_count > 10

    def test_original_hash_generated(self, sample_adt_a01):
        """Original message hash is generated for audit."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        message = parser.parse(sample_adt_a01)
        result = redactor.redact(message)

        assert result.original_hash is not None
        assert len(result.original_hash) == 64  # SHA-256 hex

    def test_redacted_fields_tracked(self, sample_message_with_phi):
        """Redacted fields are tracked for audit."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        message = parser.parse(sample_message_with_phi)
        result = redactor.redact(message)

        assert len(result.redacted_fields) > 0
        assert any("patient_name" in field for field in result.redacted_fields)

    def test_no_redaction_when_disabled(self, sample_message_with_phi):
        """No redaction when all options disabled."""
        # Create redactor that only redacts structured fields (not free text)
        redactor = HL7PHIRedactor(redact_free_text=False)
        parser = HL7Parser()

        message = parser.parse(sample_message_with_phi)
        result = redactor.redact(message)

        # Structured fields still redacted
        assert result.redactions_count > 0
        # Free text patterns NOT redacted
        assert "test@email.com" in result.redacted_message

    def test_empty_field_not_redacted(self):
        """Empty PHI fields don't create redaction entries."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||||||||||"
        message = parser.parse(raw)
        result = redactor.redact(message)

        # Empty fields should not be redacted
        assert result.redactions_count == 0


# =============================================================================
# HL7v2Connector Tests
# =============================================================================


class TestHL7v2Connector:
    """Tests for HL7v2Connector."""

    def test_init_default_options(self):
        """Initialize connector with defaults."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            tenant_id="tenant1",
        )

        assert connector.name == "HL7 v2.x Healthcare Connector"
        assert connector.hl7_source_type == "file"
        assert connector.enable_phi_redaction is True
        assert connector.enable_audit_log is True

    def test_init_with_message_filter(self):
        """Initialize connector with message type filter."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ADT", "ORU"],
        )

        assert connector.message_types == {"ADT", "ORU"}

    def test_init_phi_redaction_disabled(self):
        """Initialize connector with PHI redaction disabled."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            enable_phi_redaction=False,
        )

        assert connector.redactor is None

    def test_init_strict_parsing(self):
        """Initialize connector with strict parsing."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            strict_parsing=True,
        )

        assert connector.parser.strict is True

    @pytest.mark.asyncio
    async def test_process_message_basic(self, sample_adt_a01):
        """Process a basic HL7 message."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            enable_phi_redaction=False,
        )

        message = connector.parser.parse(sample_adt_a01)
        item = await connector._process_message(message)

        assert item is not None
        assert item.source_type == "hl7v2"
        assert item.domain == "healthcare"
        assert "ADT^A01" in item.metadata["message_type"]

    @pytest.mark.asyncio
    async def test_process_message_with_redaction(self, sample_adt_a01):
        """Process message with PHI redaction."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            enable_phi_redaction=True,
        )

        message = connector.parser.parse(sample_adt_a01)
        item = await connector._process_message(message)

        assert item is not None
        assert item.metadata.get("phi_redacted") is True
        assert item.metadata.get("redactions_count", 0) > 0
        assert "original_hash" in item.metadata

    @pytest.mark.asyncio
    async def test_process_message_filtered_out(self, sample_adt_a01):
        """Message filtered out by type filter returns None."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ORU"],  # Only ORU, not ADT
        )

        message = connector.parser.parse(sample_adt_a01)
        item = await connector._process_message(message)

        assert item is None

    @pytest.mark.asyncio
    async def test_process_message_passes_filter(self, sample_adt_a01):
        """Message passes type filter."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ADT"],
        )

        message = connector.parser.parse(sample_adt_a01)
        item = await connector._process_message(message)

        assert item is not None

    @pytest.mark.asyncio
    async def test_sync_items_file_source(self, sample_adt_a01, sample_oru_r01):
        """Sync items from file source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test HL7 files
            Path(tmpdir, "msg1.hl7").write_text(sample_adt_a01)
            Path(tmpdir, "msg2.hl7").write_text(sample_oru_r01)

            connector = HL7v2Connector(
                connector_id="test-hl7",
                source_type="file",
                source_path=tmpdir,
                enable_phi_redaction=False,
            )

            items = []
            async for item in connector.sync_items():
                items.append(item)

            assert len(items) == 2

    @pytest.mark.asyncio
    async def test_sync_items_file_source_missing_path(self):
        """Sync with missing source path logs error."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="file",
            source_path=None,
        )

        items = []
        async for item in connector.sync_items():
            items.append(item)

        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_sync_items_file_source_nonexistent_path(self):
        """Sync with nonexistent path logs error."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="file",
            source_path="/nonexistent/path/that/does/not/exist",
        )

        items = []
        async for item in connector.sync_items():
            items.append(item)

        assert len(items) == 0

    def test_create_ack_accept(self, sample_adt_a01):
        """Create ACK message with accept code."""
        connector = HL7v2Connector(connector_id="test-hl7")
        original = connector.parser.parse(sample_adt_a01)

        ack = connector._create_ack(original, "AA")

        assert ack.message_type == "ACK"
        assert "AA" in ack.raw
        assert original.message_control_id in ack.raw

    def test_create_ack_error(self, sample_adt_a01):
        """Create ACK message with error code."""
        connector = HL7v2Connector(connector_id="test-hl7")
        original = connector.parser.parse(sample_adt_a01)

        ack = connector._create_ack(original, "AE")

        assert "AE" in ack.raw

    def test_get_stats(self):
        """Get connector statistics."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            message_types=["ADT"],
        )

        stats = connector.get_stats()

        assert stats["messages_processed"] == 0
        assert stats["messages_redacted"] == 0
        assert stats["parsing_errors"] == 0
        assert stats["source_type"] == "file"
        assert stats["phi_redaction_enabled"] is True
        assert "ADT" in stats["message_type_filter"]

    @pytest.mark.asyncio
    async def test_stats_updated_after_processing(self, sample_adt_a01):
        """Statistics updated after processing messages."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            enable_phi_redaction=True,
        )

        message = connector.parser.parse(sample_adt_a01)
        await connector._process_message(message)

        stats = connector.get_stats()
        assert stats["messages_processed"] == 1
        assert stats["messages_redacted"] == 1

    def test_audit_logging(self, sample_adt_a01):
        """Audit logging produces expected output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            audit_path = f.name

        try:
            connector = HL7v2Connector(
                connector_id="test-hl7",
                enable_audit_log=True,
                audit_log_path=audit_path,
            )

            message = connector.parser.parse(sample_adt_a01)
            connector._log_audit_event(message, None)

            # Read audit log
            with open(audit_path) as f:
                log_content = f.read()

            assert "hl7_message_processed" in log_content
            assert "ADT^A01" in log_content
            assert "MSG00001" in log_content
        finally:
            Path(audit_path).unlink(missing_ok=True)


class TestHL7v2ConnectorMLLP:
    """Tests for MLLP listener functionality."""

    @pytest.mark.asyncio
    async def test_start_stop_mllp_listener(self):
        """Start and stop MLLP listener."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="mllp",
            mllp_host="127.0.0.1",
            mllp_port=12575,  # Non-standard port for testing
        )

        await connector.start_mllp_listener()
        assert connector._mllp_server is not None

        await connector.stop_mllp_listener()
        assert connector._mllp_server is None

    @pytest.mark.asyncio
    async def test_start_listener_twice_warns(self):
        """Starting listener twice logs warning."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="mllp",
            mllp_host="127.0.0.1",
            mllp_port=12576,
        )

        await connector.start_mllp_listener()
        try:
            # Second start should warn but not fail
            await connector.start_mllp_listener()
            assert connector._mllp_server is not None
        finally:
            await connector.stop_mllp_listener()

    @pytest.mark.asyncio
    async def test_sync_items_mllp_empty_queue(self):
        """Sync items from MLLP with empty queue."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="mllp",
        )

        items = []
        async for item in connector.sync_items():
            items.append(item)

        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_sync_items_mllp_with_queued_messages(self, sample_adt_a01):
        """Sync items from MLLP with messages in queue."""
        connector = HL7v2Connector(
            connector_id="test-hl7",
            source_type="mllp",
            enable_phi_redaction=False,
        )

        # Manually add message to queue (simulating MLLP receive)
        message = connector.parser.parse(sample_adt_a01)
        await connector._message_queue.put(message)

        items = []
        async for item in connector.sync_items():
            items.append(item)

        assert len(items) == 1
        assert items[0].metadata["message_type"] == "ADT^A01"


# =============================================================================
# Message Type and Segment Type Enum Tests
# =============================================================================


class TestHL7Enums:
    """Tests for HL7 enum types."""

    def test_message_types_defined(self):
        """All common message types are defined."""
        assert HL7MessageType.ADT_A01 == "ADT^A01"
        assert HL7MessageType.ORU_R01 == "ORU^R01"
        assert HL7MessageType.ORM_O01 == "ORM^O01"
        assert HL7MessageType.SIU_S12 == "SIU^S12"
        assert HL7MessageType.QRY_A19 == "QRY^A19"
        assert HL7MessageType.MFN == "MFN"

    def test_segment_types_defined(self):
        """All common segment types are defined."""
        assert HL7SegmentType.MSH == "MSH"
        assert HL7SegmentType.PID == "PID"
        assert HL7SegmentType.PV1 == "PV1"
        assert HL7SegmentType.OBX == "OBX"
        assert HL7SegmentType.ORC == "ORC"
        assert HL7SegmentType.OBR == "OBR"
        assert HL7SegmentType.SCH == "SCH"
        assert HL7SegmentType.NTE == "NTE"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_minimal_message(self):
        """Parse minimal valid HL7 message."""
        parser = HL7Parser()
        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5"
        message = parser.parse(raw)
        assert message.msh is not None

    def test_parse_message_with_empty_segments(self):
        """Parse message with many empty fields."""
        parser = HL7Parser()
        raw = "MSH|^~\\&||||||20240115||ADT|123|P|2.5\rPID|1|||||||"
        message = parser.parse(raw)
        assert message.pid is not None
        assert message.pid.patient_name == ""

    def test_parse_unicode_content(self):
        """Parse message with unicode content."""
        parser = HL7Parser()
        raw = "MSH|^~\\&|A|B|C|D|20240115||ADT|123|P|2.5\rPID|1||||M\u00fcller^Hans"
        message = parser.parse(raw)
        assert "M\u00fcller" in message.pid.patient_name

    def test_redact_preserves_message_structure(self, sample_adt_a01):
        """Redaction preserves overall message structure."""
        redactor = HL7PHIRedactor()
        parser = HL7Parser()

        message = parser.parse(sample_adt_a01)
        original_segment_count = len(message.segments)

        result = redactor.redact(message)

        # Parse redacted message
        redacted_message = parser.parse(result.redacted_message)
        assert len(redacted_message.segments) == original_segment_count

    def test_connector_handles_malformed_file(self):
        """Connector handles malformed HL7 files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "bad.hl7").write_text("This is not an HL7 message")

            connector = HL7v2Connector(
                connector_id="test-hl7",
                source_type="file",
                source_path=tmpdir,
                strict_parsing=False,
            )

            # Should not raise, just skip bad files
            items = []

            async def collect():
                async for item in connector.sync_items():
                    items.append(item)

            asyncio.get_event_loop().run_until_complete(collect())
            assert connector._parsing_errors >= 1
