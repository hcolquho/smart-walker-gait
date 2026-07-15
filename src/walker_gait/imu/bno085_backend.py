"""
BNO085 IMU backend over I2C (e.g. Raspberry Pi / Jetson host, or USB-serial
breakout depending on your exact board). Fully buildable now, independent of
the camera -- develop and test this in parallel.

Install:
    pip install adafruit-circuitpython-bno08x adafruit-blinka

Wiring: BNO085 SCL/SDA -> host I2C pins, plus 3.3V/GND. Confirm I2C address
(default 0x4A or 0x4B depending on board) with `i2cdetect -y 1` on Linux.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from walker_gait.core.types import ImuSample
from walker_gait.imu.base import ImuSource, IMU_REGISTRY


@IMU_REGISTRY.register("bno085")
class Bno085ImuSource(ImuSource):
    def __init__(self, i2c_address: int = 0x4A):
        import board
        import busio
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import (
            BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_ROTATION_VECTOR,
        )
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.sensor = BNO08X_I2C(i2c, address=i2c_address)
        self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

    def get_reading(self) -> Optional[ImuSample]:
        ax, ay, az = self.sensor.acceleration
        gx, gy, gz = self.sensor.gyro
        qi, qj, qk, qreal = self.sensor.quaternion  # note Adafruit's (i,j,k,real) order
        return ImuSample(
            accel=np.array([ax, ay, az], dtype=np.float32),
            gyro=np.array([gx, gy, gz], dtype=np.float32),
            quat=np.array([qreal, qi, qj, qk], dtype=np.float32),  # normalize to (w,x,y,z)
            timestamp=time.time(),
        )
