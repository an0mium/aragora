"""
Tests for Moltbot Device Capabilities component.

Tests device capability models, profiles, and matching.
"""

import pytest

from aragora.extensions.moltbot import (
    ActuatorCapability,
    AudioCapability,
    CapabilityCategory,
    CapabilityMatcher,
    CapabilityRequirement,
    ComputeCapability,
    DeviceCapabilities,
    DisplayCapability,
    InputCapability,
    NetworkCapability,
    SensorCapability,
    VideoCapability,
    # Capability profiles
    edge_compute_capabilities,
    iot_hub_capabilities,
    mobile_app_capabilities,
    smart_display_capabilities,
    smart_speaker_capabilities,
)


class TestDeviceCapabilities:
    """Tests for DeviceCapabilities."""

    def test_create_device_capabilities(self):
        """Test creating device capabilities."""
        caps = DeviceCapabilities(
            device_id="test-device",
            device_type="smart_speaker",
            manufacturer="Aragora",
            model="Echo 1",
            audio=[AudioCapability.MICROPHONE, AudioCapability.SPEAKER],
            network=[NetworkCapability.WIFI, NetworkCapability.BLUETOOTH],
        )

        assert caps.device_id == "test-device"
        assert caps.device_type == "smart_speaker"
        assert len(caps.audio) == 2
        assert len(caps.network) == 2

    def test_has_capability(self):
        """Test has_capability method."""
        caps = DeviceCapabilities(
            device_id="test",
            device_type="speaker",
            audio=[AudioCapability.MICROPHONE, AudioCapability.SPEAKER],
        )

        assert caps.has_capability(CapabilityCategory.AUDIO, "microphone") is True
        assert caps.has_capability(CapabilityCategory.AUDIO, "speaker") is True
        assert caps.has_capability(CapabilityCategory.AUDIO, "wake_word") is False
        assert caps.has_capability(CapabilityCategory.VIDEO, "camera") is False

    def test_has_any_capability(self):
        """Test has_any_capability method."""
        caps = DeviceCapabilities(
            device_id="test",
            device_type="hub",
            network=[NetworkCapability.WIFI, NetworkCapability.ETHERNET],
        )

        assert caps.has_any_capability(CapabilityCategory.NETWORK, ["wifi", "cellular"]) is True
        assert caps.has_any_capability(CapabilityCategory.NETWORK, ["zigbee", "zwave"]) is False

    def test_has_all_capabilities(self):
        """Test has_all_capabilities method."""
        caps = DeviceCapabilities(
            device_id="test",
            device_type="display",
            display=[
                DisplayCapability.SCREEN,
                DisplayCapability.TOUCHSCREEN,
                DisplayCapability.HD_DISPLAY,
            ],
        )

        assert (
            caps.has_all_capabilities(CapabilityCategory.DISPLAY, ["screen", "touchscreen"]) is True
        )
        assert (
            caps.has_all_capabilities(CapabilityCategory.DISPLAY, ["screen", "4k_display"]) is False
        )

    def test_capability_score(self):
        """Test capability score calculation."""
        # Device with many capabilities should score higher
        rich_caps = DeviceCapabilities(
            device_id="rich",
            device_type="tablet",
            audio=[AudioCapability.MICROPHONE, AudioCapability.SPEAKER],
            video=[VideoCapability.CAMERA, VideoCapability.HD_VIDEO],
            display=[DisplayCapability.SCREEN, DisplayCapability.TOUCHSCREEN],
            input=[InputCapability.TOUCH, InputCapability.VOICE],
            network=[NetworkCapability.WIFI, NetworkCapability.BLUETOOTH],
            compute=[ComputeCapability.HIGH_PERFORMANCE, ComputeCapability.GPU],
        )

        minimal_caps = DeviceCapabilities(
            device_id="minimal",
            device_type="sensor",
            network=[NetworkCapability.BLE],
            compute=[ComputeCapability.LOW_POWER],
        )

        assert rich_caps.get_capability_score() > minimal_caps.get_capability_score()

    def test_to_dict(self):
        """Test converting to dictionary."""
        caps = DeviceCapabilities(
            device_id="test-device",
            device_type="speaker",
            manufacturer="Test Corp",
            audio=[AudioCapability.MICROPHONE],
            network=[NetworkCapability.WIFI],
            supports_wake_word=True,
            memory_mb=512,
        )

        result = caps.to_dict()

        assert result["device_id"] == "test-device"
        assert result["device_type"] == "speaker"
        assert result["manufacturer"] == "Test Corp"
        assert "microphone" in result["audio"]
        assert "wifi" in result["network"]
        assert result["supports_wake_word"] is True
        assert result["memory_mb"] == 512
        assert "capability_score" in result


class TestCapabilityProfiles:
    """Tests for pre-defined capability profiles."""

    def test_smart_speaker_profile(self):
        """Test smart speaker capability profile."""
        caps = smart_speaker_capabilities("speaker-1")

        assert caps.device_type == "smart_speaker"
        assert AudioCapability.MICROPHONE in caps.audio
        assert AudioCapability.SPEAKER in caps.audio
        assert AudioCapability.WAKE_WORD in caps.audio
        assert caps.supports_wake_word is True
        assert caps.supports_streaming is True

    def test_smart_display_profile(self):
        """Test smart display capability profile."""
        caps = smart_display_capabilities("display-1")

        assert caps.device_type == "smart_display"
        assert AudioCapability.MICROPHONE in caps.audio
        assert VideoCapability.CAMERA in caps.video
        assert DisplayCapability.TOUCHSCREEN in caps.display
        assert caps.supports_multimodal is True

    def test_mobile_app_profile(self):
        """Test mobile app capability profile."""
        caps = mobile_app_capabilities("mobile-1")

        assert caps.device_type == "mobile_app"
        assert VideoCapability.FOUR_K in caps.video
        assert SensorCapability.GPS in caps.sensors
        assert ComputeCapability.NPU in caps.compute
        assert NetworkCapability.CELLULAR in caps.network

    def test_iot_hub_profile(self):
        """Test IoT hub capability profile."""
        caps = iot_hub_capabilities("hub-1")

        assert caps.device_type == "iot_hub"
        assert NetworkCapability.ZIGBEE in caps.network
        assert NetworkCapability.ZWAVE in caps.network
        assert NetworkCapability.MATTER in caps.network
        assert caps.supports_offline is True
        assert caps.max_concurrent_streams == 10

    def test_edge_compute_profile(self):
        """Test edge compute capability profile."""
        caps = edge_compute_capabilities("edge-1")

        assert caps.device_type == "edge_compute"
        assert ComputeCapability.HIGH_PERFORMANCE in caps.compute
        assert ComputeCapability.GPU in caps.compute
        assert ComputeCapability.EDGE_AI in caps.compute
        assert caps.supports_offline is True
        assert caps.max_concurrent_streams == 20


class TestCapabilityMatcher:
    """Tests for capability matching."""

    def test_match_device_all_required(self):
        """Test matching with all requirements met."""
        device = smart_speaker_capabilities("speaker-1")
        requirements = [
            CapabilityRequirement(CapabilityCategory.AUDIO, "microphone"),
            CapabilityRequirement(CapabilityCategory.AUDIO, "speaker"),
        ]

        meets, score = CapabilityMatcher.match_device(device, requirements)
        assert meets is True
        assert score == 100.0

    def test_match_device_missing_required(self):
        """Test matching with missing required capability."""
        device = smart_speaker_capabilities("speaker-1")
        requirements = [
            CapabilityRequirement(CapabilityCategory.AUDIO, "microphone"),
            CapabilityRequirement(CapabilityCategory.VIDEO, "camera"),  # Missing
        ]

        meets, score = CapabilityMatcher.match_device(device, requirements)
        assert meets is False
        assert score == 50.0  # Only half met

    def test_match_device_optional_capability(self):
        """Test matching with optional capabilities."""
        device = smart_speaker_capabilities("speaker-1")
        requirements = [
            CapabilityRequirement(CapabilityCategory.AUDIO, "microphone", required=True),
            CapabilityRequirement(CapabilityCategory.VIDEO, "camera", required=False),
        ]

        meets, score = CapabilityMatcher.match_device(device, requirements)
        assert meets is True  # Required is met
        assert 50 < score < 100  # Partial score for missing optional

    def test_find_best_match(self):
        """Test finding best matching device."""
        devices = [
            smart_speaker_capabilities("speaker-1"),
            smart_display_capabilities("display-1"),
            iot_hub_capabilities("hub-1"),
        ]

        # Requirements that favor display
        requirements = [
            CapabilityRequirement(CapabilityCategory.AUDIO, "microphone"),
            CapabilityRequirement(CapabilityCategory.VIDEO, "camera"),
            CapabilityRequirement(CapabilityCategory.DISPLAY, "touchscreen"),
        ]

        best = CapabilityMatcher.find_best_match(devices, requirements)
        assert best is not None
        assert best.device_type == "smart_display"

    def test_find_best_match_no_match(self):
        """Test finding best match when no device meets requirements."""
        devices = [
            smart_speaker_capabilities("speaker-1"),
            iot_hub_capabilities("hub-1"),
        ]

        # Requirements that no device can meet
        requirements = [
            CapabilityRequirement(CapabilityCategory.SENSORS, "gps"),
            CapabilityRequirement(CapabilityCategory.NETWORK, "cellular"),
        ]

        best = CapabilityMatcher.find_best_match(devices, requirements)
        assert best is None

    def test_rank_devices(self):
        """Test ranking devices by match score."""
        devices = [
            smart_speaker_capabilities("speaker-1"),
            smart_display_capabilities("display-1"),
            mobile_app_capabilities("mobile-1"),
        ]

        requirements = [
            CapabilityRequirement(CapabilityCategory.AUDIO, "microphone"),
            CapabilityRequirement(CapabilityCategory.VIDEO, "camera", required=False),
            CapabilityRequirement(CapabilityCategory.DISPLAY, "touchscreen", required=False),
        ]

        ranked = CapabilityMatcher.rank_devices(devices, requirements)

        assert len(ranked) == 3
        # All meet required (microphone), but display and mobile have more optionals
        assert ranked[0][2] is True  # First device meets required
        assert ranked[0][1] > ranked[-1][1]  # First has higher score


class TestCapabilityCategories:
    """Tests for capability category enums."""

    def test_audio_capabilities(self):
        """Test audio capability enum."""
        assert AudioCapability.MICROPHONE.value == "microphone"
        assert AudioCapability.WAKE_WORD.value == "wake_word"
        assert len(AudioCapability) == 9

    def test_video_capabilities(self):
        """Test video capability enum."""
        assert VideoCapability.CAMERA.value == "camera"
        assert VideoCapability.FOUR_K.value == "4k"
        assert len(VideoCapability) == 10

    def test_network_capabilities(self):
        """Test network capability enum."""
        assert NetworkCapability.WIFI.value == "wifi"
        assert NetworkCapability.MATTER.value == "matter"
        assert NetworkCapability.FIVE_G.value == "5g"
        assert len(NetworkCapability) == 12

    def test_sensor_capabilities(self):
        """Test sensor capability enum."""
        assert SensorCapability.GPS.value == "gps"
        assert SensorCapability.AIR_QUALITY.value == "air_quality"
        assert len(SensorCapability) == 14

    def test_actuator_capabilities(self):
        """Test actuator capability enum."""
        assert ActuatorCapability.MOTOR.value == "motor"
        assert ActuatorCapability.RGB_LIGHT.value == "rgb_light"
        assert len(ActuatorCapability) == 8

    def test_compute_capabilities(self):
        """Test compute capability enum."""
        assert ComputeCapability.GPU.value == "gpu"
        assert ComputeCapability.NPU.value == "npu"
        assert ComputeCapability.EDGE_AI.value == "edge_ai"
        assert len(ComputeCapability) == 7
