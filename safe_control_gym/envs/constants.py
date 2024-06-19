from dataclasses import dataclass
import math

RAD_TO_DEG = 180 / math.pi


@dataclass
class SimConstants: ...


@dataclass
class DroneConstants:
    firmware_freq: int = 500  # Firmware frequency in Hz
    supply_voltage: float = 3.0  # Power supply voltage
    thrust_curve_a: float = -0.0006239  # Thrust curve parameters for brushed motors
    thrust_curve_b: float = 0.088  # Thrust curve parameters for brushed motors
    tumble_threshold: float = -0.5  # Vertical acceleration threshold for tumbling detection
    tumble_duration: int = 30  # Number of consecutive steps before tumbling is detected
    # TODO: acc and gyro were swapped in original implementation. Possible bug?
    acc_lpf_cutoff: int = 80  # Low-pass filter cutoff freq
    gyro_lpf_cutoff: int = 30  # Low-pass filter cutoff freq
