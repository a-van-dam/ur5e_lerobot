#!/usr/bin/env python3
"""
OnRobot 2FG Gripper Modbus TCP Controller
==========================================
Clean, non-blocking Modbus TCP driver for OnRobot 2FG series (2FG7 / 2FG14)
connected via the OnRobot Compute Box.

Network & Modbus Specification:
  - Default Compute Box IP: 192.168.1.1:502
  - Modbus Slave / Device ID: 65 (0x41)
  - Holding Registers (Feedback @ 256):
      256: Status Word (Bit 0: Busy, Bit 1: Grip Detected)
      257: Current External Width (0.1 mm units)
      258: Current Internal Width (0.1 mm units)
      259: Min External Width (0.1 mm units)
      260: Max External Width (0.1 mm units)
      263: Measured Force (N)
  - Holding Registers (Command @ 0):
      0: Target Width in 0.1 mm (uint16)
      1: Target Force in N (uint16)
      2: Speed Percentage 1..100 (uint16)
      3: Command (1 = Grip External, 2 = Grip Internal, 3 = Stop)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    from pymodbus.client import ModbusTcpClient
    PYMODBUS_AVAILABLE = True
except ImportError:
    ModbusTcpClient = None
    PYMODBUS_AVAILABLE = False

logger = logging.getLogger("OnRobot2FG")


class OnRobot2FGModbus:
    """
    Thread-safe Modbus TCP driver for OnRobot 2FG7 / 2FG14 grippers.
    Supports both non-blocking commands (for 35 Hz teleop loops) and synchronous waits.
    """

    def __init__(
        self,
        host: str = "192.168.1.1",
        port: int = 502,
        device_id: int = 65,
        timeout: float = 1.0,
        mock: bool = False,
    ):
        self.host = host
        self.port = port
        self.device_id = device_id
        self.timeout = timeout
        self.mock = mock

        self.client: Optional[ModbusTcpClient] = None
        self._is_connected = False
        self._lock = threading.Lock()

        # Gripper operational limits (auto-populated on connect)
        self.min_width_mm: float = 55.0
        self.max_width_mm: float = 105.0
        self.default_force_n: float = 30.0
        self.default_speed_pct: int = 80

        # Latest cache (initialized to wide-open default 105.0 mm)
        self._cached_width_mm: float = 105.0
        self._cached_status: int = 0
        self._last_cmd_time: float = 0.0
        self._last_cmd_target: float = 105.0

        # Background telemetry poller so 30 Hz recording / policy rollouts never block on Modbus I/O
        self._poll_stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def _start_poller(self, hz: float = 25.0) -> None:
        """Starts a lightweight daemon thread refreshing cached gripper width & status."""
        if self.mock or not self._is_connected:
            return
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._poll_stop_event.clear()
        period = 1.0 / max(1.0, float(hz))

        def _poll_loop():
            while not self._poll_stop_event.is_set() and self._is_connected:
                # Yield immediately after a write command so grip() gets priority on Modbus TCP
                if time.time() - self._last_cmd_time >= 0.08:
                    with self._lock:
                        try:
                            if self._is_connected and self.client is not None:
                                res = self._read_holding(address=256, count=2)
                                if not res.isError() and len(res.registers) >= 2:
                                    self._cached_status = res.registers[0]
                                    self._cached_width_mm = res.registers[1] / 10.0
                        except Exception:
                            pass
                self._poll_stop_event.wait(period)

        self._poll_thread = threading.Thread(target=_poll_loop, name="OnRobot2FGPoller", daemon=True)
        self._poll_thread.start()

    def _stop_poller(self) -> None:
        self._poll_stop_event.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=0.5)
        self._poll_thread = None

    def connect(self) -> bool:
        """Establishes connection to the OnRobot Compute Box and reads stroke limits."""
        if self.mock:
            self._is_connected = True
            return True

        if not PYMODBUS_AVAILABLE or ModbusTcpClient is None:
            logger.warning("pymodbus is not installed; OnRobot gripper cannot connect.")
            return False

        with self._lock:
            try:
                self.client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout)
                if not self.client.connect():
                    self._is_connected = False
                    return False

                self._is_connected = True
            except Exception as e:
                logger.error(f"Failed to connect to Compute Box at {self.host}:{self.port}: {e}")
                self._is_connected = False
                return False

        # Read actual limits from hardware
        info = self.detect_gripper()
        if info is not None:
            self.min_width_mm = info.get("min_external_mm", 55.0)
            self.max_width_mm = info.get("max_external_mm", 105.0)
            self._cached_width_mm = info.get("external_width_mm", 105.0)
            self._last_cmd_target = self._cached_width_mm
            logger.info(
                f"OnRobot 2FG connected: Width={self._cached_width_mm:.1f}mm "
                f"Limits=[{self.min_width_mm:.1f}, {self.max_width_mm:.1f}]mm"
            )

        if self._is_connected:
            self._start_poller(hz=25.0)

        return self._is_connected

    def disconnect(self) -> None:
        """Closes Modbus socket cleanly."""
        self._stop_poller()
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _read_holding(self, address: int, count: int = 1):
        if not self._is_connected or self.client is None:
            raise ConnectionError("Modbus client is not connected.")
        try:
            return self.client.read_holding_registers(address=address, count=count, device_id=self.device_id)
        except TypeError:
            return self.client.read_holding_registers(address=address, count=count, slave=self.device_id)

    def _write_holding(self, address: int, values: list[int]):
        if not self._is_connected or self.client is None:
            raise ConnectionError("Modbus client is not connected.")
        try:
            return self.client.write_registers(address=address, values=values, device_id=self.device_id)
        except TypeError:
            return self.client.write_registers(address=address, values=values, slave=self.device_id)

    def detect_gripper(self) -> Optional[Dict[str, Any]]:
        """Scans Device ID 65 and returns telemetry dictionary if present."""
        if self.mock:
            return {
                "detected": True,
                "device_id": self.device_id,
                "status_word": 0,
                "busy": False,
                "grip_detected": False,
                "external_width_mm": self._cached_width_mm,
                "internal_width_mm": self._cached_width_mm + 12.6,
                "min_external_mm": self.min_width_mm,
                "max_external_mm": self.max_width_mm,
                "measured_force_n": 5,
            }

        with self._lock:
            try:
                res = self._read_holding(address=256, count=8)
                if res.isError():
                    return None
                regs = res.registers
                status_word = regs[0]
                ext_width = regs[1] / 10.0
                int_width = regs[2] / 10.0
                min_ext = regs[3] / 10.0
                max_ext = regs[4] / 10.0
                force_n = regs[7]

                self._cached_width_mm = ext_width
                self._cached_status = status_word

                return {
                    "detected": True,
                    "device_id": self.device_id,
                    "status_word": status_word,
                    "busy": bool(status_word & 0x01),
                    "grip_detected": bool(status_word & 0x02),
                    "external_width_mm": ext_width,
                    "internal_width_mm": int_width,
                    "min_external_mm": min_ext,
                    "max_external_mm": max_ext,
                    "measured_force_n": force_n,
                }
            except Exception as e:
                logger.debug(f"Error reading gripper telemetry: {e}")
                return None

    def get_width(self) -> float:
        """Reads current external aperture width in mm."""
        if self.mock or not self._is_connected:
            return self._cached_width_mm

        with self._lock:
            try:
                res = self._read_holding(address=257, count=1)
                if not res.isError():
                    self._cached_width_mm = res.registers[0] / 10.0
            except Exception:
                pass
        return self._cached_width_mm

    def get_cached_width(self) -> float:
        """Returns latest measured gripper aperture width in mm without blocking on Modbus I/O."""
        return float(self._cached_width_mm)

    def get_target_width(self) -> float:
        """Returns latest commanded target gripper width in mm."""
        return float(self._last_cmd_target)

    def is_closed(self, threshold_mm: Optional[float] = None) -> bool:
        """Returns True if the gripper's commanded target width is below the open/close threshold."""
        thresh = (0.5 * (self.min_width_mm + self.max_width_mm)) if threshold_mm is None else float(threshold_mm)
        return float(self._last_cmd_target) < thresh

    def get_status(self) -> int:
        """Reads status bitmask."""
        if self.mock or not self._is_connected:
            return self._cached_status

        with self._lock:
            try:
                res = self._read_holding(address=256, count=1)
                if not res.isError():
                    self._cached_status = res.registers[0]
            except Exception:
                pass
        return self._cached_status

    def is_busy(self) -> bool:
        return bool(self.get_status() & 0x01)

    def is_grip_detected(self) -> bool:
        return bool(self.get_status() & 0x02)

    def grip(
        self,
        width_mm: float,
        force_n: Optional[float] = None,
        speed_percent: Optional[int] = None,
        internal: bool = False,
        wait: bool = False,
        timeout: float = 3.0,
    ) -> float:
        """
        Commands the gripper to the target width:
          - width_mm: Desired opening width in mm (clamped to physical limits)
          - force_n: Gripping force ceiling in N (default: self.default_force_n)
          - speed_percent: Speed 1..100% (default: self.default_speed_pct)
          - wait: If True, blocks until motion finishes or timeout
        """
        # Clamp target to physical limits
        target_clamped = max(self.min_width_mm, min(self.max_width_mm, float(width_mm)))
        force = self.default_force_n if force_n is None else float(force_n)
        speed = self.default_speed_pct if speed_percent is None else int(speed_percent)

        self._last_cmd_target = target_clamped
        self._last_cmd_time = time.time()

        if self.mock or not self._is_connected:
            self._cached_width_mm = target_clamped
            return self._cached_width_mm

        raw_width = max(0, min(65535, int(round(target_clamped * 10.0))))
        raw_force = max(0, min(65535, int(round(force))))
        raw_speed = max(1, min(100, int(speed)))
        cmd = 2 if internal else 1

        values = [raw_width, raw_force, raw_speed, cmd]

        with self._lock:
            try:
                self._write_holding(address=0, values=values)
            except Exception as e:
                logger.error(f"Failed to dispatch grip command {values}: {e}")
                return self._cached_width_mm

        if wait:
            time.sleep(0.12)  # Allow hardware to raise busy bit
            t0 = time.time()
            while time.time() - t0 < timeout:
                if not self.is_busy():
                    break
                time.sleep(0.05)

        return self.get_width()

    def jog_delta(
        self,
        delta_mm: float,
        force_n: Optional[float] = None,
        speed_percent: Optional[int] = None,
    ) -> float:
        """
        Incrementally adjusts current target width by delta_mm.
        Ideal for continuous button hold teleop (e.g., A/B buttons).
        """
        new_target = self._last_cmd_target + delta_mm
        return self.grip(
            width_mm=new_target,
            force_n=force_n,
            speed_percent=speed_percent,
            wait=False,
        )

    def close(self, force_n: Optional[float] = None, wait: bool = False) -> float:
        """Close to minimum external width."""
        return self.grip(self.min_width_mm, force_n=force_n, wait=wait)

    def open(self, force_n: Optional[float] = None, wait: bool = False) -> float:
        """Open to maximum external width."""
        return self.grip(self.max_width_mm, force_n=force_n, wait=wait)

    def stop(self) -> None:
        """Sends stop command (Register 3 = 3)."""
        if self.mock:
            return
        with self._lock:
            try:
                self._write_holding(address=0, values=[0, 0, 0, 3])
            except Exception as e:
                logger.error(f"Failed to send stop command: {e}")


# Global singleton instance for shared application access
_GLOBAL_GRIPPER: Optional[OnRobot2FGModbus] = None
_GLOBAL_LOCK = threading.Lock()


def get_shared_gripper(host: str = "192.168.1.1", auto_connect: bool = True) -> OnRobot2FGModbus:
    """Returns or creates the shared OnRobot gripper singleton."""
    global _GLOBAL_GRIPPER
    with _GLOBAL_LOCK:
        if _GLOBAL_GRIPPER is None:
            _GLOBAL_GRIPPER = OnRobot2FGModbus(host=host)
            if auto_connect:
                _GLOBAL_GRIPPER.connect()
        return _GLOBAL_GRIPPER

