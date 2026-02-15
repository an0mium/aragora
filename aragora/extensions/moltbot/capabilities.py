"""
Device Capabilities - Hardware and Software Capability Models.

Provides a comprehensive model for device capabilities, feature detection,
and capability-based routing for heterogeneous device networks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityCategory(Enum):
    """Categories of device capabilities."""

    AUDIO = "audio"
    VIDEO = "video"
    DISPLAY = "display"
    INPUT = "input"
    NETWORK = "network"
    STORAGE = "storage"
    COMPUTE = "compute"
    SENSORS = "sensors"
    ACTUATORS = "actuators"
    COMMUNICATION = "communication"


class AudioCapability(Enum):
    """Audio-related capabilities."""

    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    MULTI_CHANNEL = "multi_channel"
    NOISE_CANCELLATION = "noise_cancellation"
    ECHO_CANCELLATION = "echo_cancellation"
    VOICE_ACTIVITY_DETECTION = "vad"
    WAKE_WORD = "wake_word"
    SPATIAL_AUDIO = "spatial_audio"
    HIGH_FIDELITY = "high_fidelity"


class VideoCapability(Enum):
    """Video-related capabilities."""

    CAMERA = "camera"
    MULTI_CAMERA = "multi_camera"
    HD_VIDEO = "hd_video"
    FOUR_K = "4k"
    NIGHT_VISION = "night_vision"
    WIDE_ANGLE = "wide_angle"
    PTZ = "ptz"  # Pan-Tilt-Zoom
    DEPTH_SENSING = "depth_sensing"
    FACE_DETECTION = "face_detection"
    OBJECT_DETECTION = "object_detection"


class DisplayCapability(Enum):
    """Display-related capabilities."""

    SCREEN = "screen"
    TOUCHSCREEN = "touchscreen"
    HD_DISPLAY = "hd_display"
    FOUR_K_DISPLAY = "4k_display"
    HDR = "hdr"
    E_INK = "e_ink"
    LED_INDICATOR = "led_indicator"
    MULTI_DISPLAY = "multi_display"


class InputCapability(Enum):
    """Input-related capabilities."""

    TOUCH = "touch"
    MULTI_TOUCH = "multi_touch"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    VOICE = "voice"
    GESTURE = "gesture"
    REMOTE = "remote"
    BUTTON = "button"
    DIAL = "dial"


class NetworkCapability(Enum):
    """Network-related capabilities."""

    WIFI = "wifi"
    WIFI_6 = "wifi_6"
    ETHERNET = "ethernet"
    BLUETOOTH = "bluetooth"
    BLE = "ble"
    ZIGBEE = "zigbee"
    ZWAVE = "zwave"
    THREAD = "thread"
    MATTER = "matter"
    CELLULAR = "cellular"
    FIVE_G = "5g"
    NFC = "nfc"


class SensorCapability(Enum):
    """Sensor-related capabilities."""

    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    PROXIMITY = "proximity"
    LIGHT = "light"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    AIR_QUALITY = "air_quality"
    CO2 = "co2"
    SMOKE = "smoke"
    WATER_LEAK = "water_leak"


class ActuatorCapability(Enum):
    """Actuator-related capabilities."""

    MOTOR = "motor"
    SERVO = "servo"
    RELAY = "relay"
    VALVE = "valve"
    LOCK = "lock"
    HVAC = "hvac"
    DIMMER = "dimmer"
    RGB_LIGHT = "rgb_light"


class ComputeCapability(Enum):
    """Compute-related capabilities."""

    LOW_POWER = "low_power"
    STANDARD = "standard"
    HIGH_PERFORMANCE = "high_performance"
    GPU = "gpu"
    NPU = "npu"  # Neural Processing Unit
    TPU = "tpu"  # Tensor Processing Unit
    EDGE_AI = "edge_ai"


@dataclass
class CapabilitySpec:
    """Specification for a single capability."""

    capability: str
    category: CapabilityCategory
    version: str | None = None
    quality: str | None = None  # e.g., "high", "medium", "low"
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceCapabilities:
    """Complete capability profile for a device."""

    device_id: str
    device_type: str  # e.g., "smart_speaker", "tablet", "hub"
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""

    # Capability lists
    audio: list[AudioCapability] = field(default_factory=list)
    video: list[VideoCapability] = field(default_factory=list)
    display: list[DisplayCapability] = field(default_factory=list)
    input: list[InputCapability] = field(default_factory=list)
    network: list[NetworkCapability] = field(default_factory=list)
    sensors: list[SensorCapability] = field(default_factory=list)
    actuators: list[ActuatorCapability] = field(default_factory=list)
    compute: list[ComputeCapability] = field(default_factory=list)

    # Detailed specs
    specs: list[CapabilitySpec] = field(default_factory=list)

    # Resource limits
    memory_mb: int | None = None
    storage_mb: int | None = None
    battery_capacity_mah: int | None = None
    max_concurrent_streams: int = 1

    # Feature flags
    supports_wake_word: bool = False
    supports_offline: bool = False
    supports_streaming: bool = False
    supports_multimodal: bool = False

    # Timestamps
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    metadata: dict[str, Any] = field(default_factory=dict)

    def has_capability(
        self,
        category: CapabilityCategory,
        capability: str,
    ) -> bool:
        """Check if device has a specific capability."""
        cap_list = self._get_capability_list(category)
        return any(c.value == capability for c in cap_list)

    def has_any_capability(
        self,
        category: CapabilityCategory,
        capabilities: list[str],
    ) -> bool:
        """Check if device has any of the specified capabilities."""
        return any(self.has_capability(category, c) for c in capabilities)

    def has_all_capabilities(
        self,
        category: CapabilityCategory,
        capabilities: list[str],
    ) -> bool:
        """Check if device has all specified capabilities."""
        return all(self.has_capability(category, c) for c in capabilities)

    def _get_capability_list(self, category: CapabilityCategory) -> list[Any]:
        """Get the capability list for a category."""
        mapping: dict[CapabilityCategory, list[Any]] = {
            CapabilityCategory.AUDIO: self.audio,
            CapabilityCategory.VIDEO: self.video,
            CapabilityCategory.DISPLAY: self.display,
            CapabilityCategory.INPUT: self.input,
            CapabilityCategory.NETWORK: self.network,
            CapabilityCategory.SENSORS: self.sensors,
            CapabilityCategory.ACTUATORS: self.actuators,
            CapabilityCategory.COMPUTE: self.compute,
        }
        return mapping.get(category, [])

    def get_capability_score(self) -> float:
        """Calculate overall capability score (0-100)."""
        total = 0
        max_score = 0

        # Weight different categories
        weights = {
            CapabilityCategory.AUDIO: 15,
            CapabilityCategory.VIDEO: 15,
            CapabilityCategory.DISPLAY: 15,
            CapabilityCategory.INPUT: 10,
            CapabilityCategory.NETWORK: 15,
            CapabilityCategory.SENSORS: 10,
            CapabilityCategory.ACTUATORS: 5,
            CapabilityCategory.COMPUTE: 15,
        }

        # Audio scoring
        if self.audio:
            total += min(len(self.audio) * 3, weights[CapabilityCategory.AUDIO])
        max_score += weights[CapabilityCategory.AUDIO]

        # Video scoring
        if self.video:
            total += min(len(self.video) * 3, weights[CapabilityCategory.VIDEO])
        max_score += weights[CapabilityCategory.VIDEO]

        # Display scoring
        if self.display:
            total += min(len(self.display) * 5, weights[CapabilityCategory.DISPLAY])
        max_score += weights[CapabilityCategory.DISPLAY]

        # Input scoring
        if self.input:
            total += min(len(self.input) * 2, weights[CapabilityCategory.INPUT])
        max_score += weights[CapabilityCategory.INPUT]

        # Network scoring
        if self.network:
            total += min(len(self.network) * 3, weights[CapabilityCategory.NETWORK])
        max_score += weights[CapabilityCategory.NETWORK]

        # Sensors scoring
        if self.sensors:
            total += min(len(self.sensors) * 2, weights[CapabilityCategory.SENSORS])
        max_score += weights[CapabilityCategory.SENSORS]

        # Actuators scoring
        if self.actuators:
            total += min(len(self.actuators) * 2, weights[CapabilityCategory.ACTUATORS])
        max_score += weights[CapabilityCategory.ACTUATORS]

        # Compute scoring
        if self.compute:
            total += min(len(self.compute) * 5, weights[CapabilityCategory.COMPUTE])
        max_score += weights[CapabilityCategory.COMPUTE]

        return (total / max_score * 100) if max_score > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "audio": [c.value for c in self.audio],
            "video": [c.value for c in self.video],
            "display": [c.value for c in self.display],
            "input": [c.value for c in self.input],
            "network": [c.value for c in self.network],
            "sensors": [c.value for c in self.sensors],
            "actuators": [c.value for c in self.actuators],
            "compute": [c.value for c in self.compute],
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "supports_wake_word": self.supports_wake_word,
            "supports_offline": self.supports_offline,
            "supports_streaming": self.supports_streaming,
            "supports_multimodal": self.supports_multimodal,
            "capability_score": self.get_capability_score(),
            "registered_at": self.registered_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


# ============================================================================
# Capability Profiles - Pre-defined device capability templates
# ============================================================================


def smart_speaker_capabilities(device_id: str) -> DeviceCapabilities:
    """Create capabilities for a smart speaker device."""
    return DeviceCapabilities(
        device_id=device_id,
        device_type="smart_speaker",
        audio=[
            AudioCapability.MICROPHONE,
            AudioCapability.SPEAKER,
            AudioCapability.NOISE_CANCELLATION,
            AudioCapability.ECHO_CANCELLATION,
            AudioCapability.VOICE_ACTIVITY_DETECTION,
            AudioCapability.WAKE_WORD,
        ],
        input=[InputCapability.VOICE, InputCapability.BUTTON],
        network=[
            NetworkCapability.WIFI,
            NetworkCapability.BLUETOOTH,
        ],
        compute=[ComputeCapability.LOW_POWER],
        supports_wake_word=True,
        supports_streaming=True,
        memory_mb=512,
        storage_mb=4096,
    )


def smart_display_capabilities(device_id: str) -> DeviceCapabilities:
    """Create capabilities for a smart display device."""
    return DeviceCapabilities(
        device_id=device_id,
        device_type="smart_display",
        audio=[
            AudioCapability.MICROPHONE,
            AudioCapability.SPEAKER,
            AudioCapability.NOISE_CANCELLATION,
            AudioCapability.WAKE_WORD,
        ],
        video=[
            VideoCapability.CAMERA,
            VideoCapability.HD_VIDEO,
        ],
        display=[
            DisplayCapability.SCREEN,
            DisplayCapability.TOUCHSCREEN,
            DisplayCapability.HD_DISPLAY,
        ],
        input=[
            InputCapability.VOICE,
            InputCapability.TOUCH,
            InputCapability.GESTURE,
        ],
        network=[
            NetworkCapability.WIFI,
            NetworkCapability.BLUETOOTH,
        ],
        compute=[ComputeCapability.STANDARD],
        supports_wake_word=True,
        supports_streaming=True,
        supports_multimodal=True,
        memory_mb=2048,
        storage_mb=16384,
    )


def mobile_app_capabilities(device_id: str) -> DeviceCapabilities:
    """Create capabilities for a mobile app."""
    return DeviceCapabilities(
        device_id=device_id,
        device_type="mobile_app",
        audio=[
            AudioCapability.MICROPHONE,
            AudioCapability.SPEAKER,
            AudioCapability.NOISE_CANCELLATION,
        ],
        video=[
            VideoCapability.CAMERA,
            VideoCapability.MULTI_CAMERA,
            VideoCapability.HD_VIDEO,
            VideoCapability.FOUR_K,
        ],
        display=[
            DisplayCapability.SCREEN,
            DisplayCapability.TOUCHSCREEN,
            DisplayCapability.HD_DISPLAY,
        ],
        input=[
            InputCapability.TOUCH,
            InputCapability.MULTI_TOUCH,
            InputCapability.VOICE,
            InputCapability.GESTURE,
        ],
        network=[
            NetworkCapability.WIFI,
            NetworkCapability.CELLULAR,
            NetworkCapability.BLUETOOTH,
            NetworkCapability.NFC,
        ],
        sensors=[
            SensorCapability.ACCELEROMETER,
            SensorCapability.GYROSCOPE,
            SensorCapability.GPS,
            SensorCapability.PROXIMITY,
            SensorCapability.LIGHT,
        ],
        compute=[
            ComputeCapability.HIGH_PERFORMANCE,
            ComputeCapability.GPU,
            ComputeCapability.NPU,
        ],
        supports_wake_word=False,  # Usually requires app open
        supports_streaming=True,
        supports_multimodal=True,
        memory_mb=8192,
        storage_mb=131072,
    )


def iot_hub_capabilities(device_id: str) -> DeviceCapabilities:
    """Create capabilities for an IoT hub device."""
    return DeviceCapabilities(
        device_id=device_id,
        device_type="iot_hub",
        network=[
            NetworkCapability.WIFI,
            NetworkCapability.ETHERNET,
            NetworkCapability.BLUETOOTH,
            NetworkCapability.BLE,
            NetworkCapability.ZIGBEE,
            NetworkCapability.ZWAVE,
            NetworkCapability.THREAD,
            NetworkCapability.MATTER,
        ],
        compute=[ComputeCapability.STANDARD],
        supports_offline=True,
        max_concurrent_streams=10,
        memory_mb=1024,
        storage_mb=8192,
    )


def edge_compute_capabilities(device_id: str) -> DeviceCapabilities:
    """Create capabilities for an edge compute device."""
    return DeviceCapabilities(
        device_id=device_id,
        device_type="edge_compute",
        network=[
            NetworkCapability.WIFI,
            NetworkCapability.ETHERNET,
        ],
        compute=[
            ComputeCapability.HIGH_PERFORMANCE,
            ComputeCapability.GPU,
            ComputeCapability.EDGE_AI,
        ],
        supports_offline=True,
        supports_streaming=True,
        max_concurrent_streams=20,
        memory_mb=16384,
        storage_mb=524288,
    )


# ============================================================================
# Capability Matcher - Match capabilities to requirements
# ============================================================================


@dataclass
class CapabilityRequirement:
    """A required capability for a task."""

    category: CapabilityCategory
    capability: str
    required: bool = True  # False means preferred but not required
    min_quality: str | None = None


class CapabilityMatcher:
    """Match devices to capability requirements."""

    @staticmethod
    def match_device(
        device: DeviceCapabilities,
        requirements: list[CapabilityRequirement],
    ) -> tuple[bool, float]:
        """
        Check if a device meets capability requirements.

        Args:
            device: Device capabilities
            requirements: Required capabilities

        Returns:
            Tuple of (meets_required, match_score)
        """
        required_met = True
        total_score = 0.0
        max_score = 0.0

        for req in requirements:
            has_cap = device.has_capability(req.category, req.capability)

            if req.required:
                max_score += 1.0
                if has_cap:
                    total_score += 1.0
                else:
                    required_met = False
            else:
                max_score += 0.5
                if has_cap:
                    total_score += 0.5

        match_score = (total_score / max_score * 100) if max_score > 0 else 100

        return required_met, match_score

    @staticmethod
    def find_best_match(
        devices: list[DeviceCapabilities],
        requirements: list[CapabilityRequirement],
    ) -> DeviceCapabilities | None:
        """
        Find the best matching device for requirements.

        Args:
            devices: Available devices
            requirements: Required capabilities

        Returns:
            Best matching device or None
        """
        candidates = []

        for device in devices:
            meets_required, score = CapabilityMatcher.match_device(device, requirements)
            if meets_required:
                candidates.append((device, score))

        if not candidates:
            return None

        # Sort by score and capability score as tiebreaker
        candidates.sort(
            key=lambda x: (x[1], x[0].get_capability_score()),
            reverse=True,
        )

        return candidates[0][0]

    @staticmethod
    def rank_devices(
        devices: list[DeviceCapabilities],
        requirements: list[CapabilityRequirement],
    ) -> list[tuple[DeviceCapabilities, float, bool]]:
        """
        Rank devices by how well they match requirements.

        Args:
            devices: Available devices
            requirements: Required capabilities

        Returns:
            List of (device, score, meets_required) sorted by score
        """
        results = []

        for device in devices:
            meets_required, score = CapabilityMatcher.match_device(device, requirements)
            results.append((device, score, meets_required))

        results.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return results
