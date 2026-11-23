"""
CPU energy measurement (spec §2, §10).

Primary: RAPL "CPU Package" power sensor read via LibreHardwareMonitorLib.dll
over pythonnet (.NET interop) — must be confirmed working on this machine in
the pilot (spec §19). Only instantaneous power (watts) is exposed, not a
cumulative energy counter, so it's sampled on an interval and
trapezoidal-integrated.

Fallback: a constant-power estimate (config: energy.cpu.estimated_tdp_watts)
multiplied by elapsed time, used only if RAPL access can't be established at
startup. Every reading is tagged with the method that produced it
("rapl_measured" or "estimated_constant_power") so the two are never blended
silently. In estimated mode there is no sensor, so no raw samples exist and
stop() returns an empty `samples` list.

`.method` is readable right after construction, before any phase runs.
"""

import time
import threading


class CPUEnergyMonitor:
    def __init__(self, dll_path: str, sample_interval_ms: int = 100,
                 estimated_tdp_watts: float = 55.0, idle_watts: float = 0.0):
        self._sample_interval_s = sample_interval_ms / 1000.0
        self._estimated_tdp_watts = estimated_tdp_watts
        self._idle_watts = idle_watts
        self._computer = None
        self._cpu_hw = None
        self._power_sensor = None
        self.method = self._try_init_rapl(dll_path)

        self._phase = None
        self._start_time = None
        self._samples = []
        self._stop_event = None
        self._sampler_thread = None

    def _try_init_rapl(self, dll_path: str) -> str:
        """Attempts to load LibreHardwareMonitorLib and locate the CPU Package
        power sensor. Returns "rapl_measured" on success, else
        "estimated_constant_power" (and prints why, so the fallback is
        visible in the run log, not just in the recorded data)."""
        try:
            from pythonnet import load
            load("coreclr")

            import clr  # pythonnet
            clr.AddReference(dll_path)
            from LibreHardwareMonitor.Hardware import Computer, HardwareType, SensorType

            computer = Computer()
            computer.IsCpuEnabled = True
            computer.Open()

            cpu_hw = next((hw for hw in computer.Hardware if hw.HardwareType == HardwareType.Cpu), None)
            if cpu_hw is None:
                print("[energy_monitor_cpu] No CPU hardware object found — "
                      "falling back to estimated_constant_power.")
                return "estimated_constant_power"

            cpu_hw.Update()
            package_sensor = next(
                (s for s in cpu_hw.Sensors if s.SensorType == SensorType.Power and "Package" in str(s.Name)),
                None,
            )
            if package_sensor is None or package_sensor.Value is None:
                print("[energy_monitor_cpu] CPU Package power sensor not found or unreadable — "
                      "falling back to estimated_constant_power. Confirm RAPL access in the "
                      "pilot (spec §19); this may need a BIOS/driver change.")
                return "estimated_constant_power"

            self._computer = computer
            self._cpu_hw = cpu_hw
            self._power_sensor = package_sensor
            return "rapl_measured"

        except Exception as exc:  # broad on purpose: any interop failure -> documented fallback
            print(f"[energy_monitor_cpu] LibreHardwareMonitor unavailable ({exc}) — "
                  f"falling back to estimated_constant_power.")
            return "estimated_constant_power"

    def read_current_watts(self) -> float:
        self._cpu_hw.Update()
        value = self._power_sensor.Value
        return float(value) if value is not None else 0.0

    def _read_sample(self) -> dict:
        return {"perf_t": time.perf_counter(), "wall_t": time.time(), "watts": self.read_current_watts()}

    def _sample_loop(self):
        while not self._stop_event.is_set():
            self._samples.append(self._read_sample())
            self._stop_event.wait(self._sample_interval_s)

    def start(self, phase: str = "unspecified"):
        self._phase = phase
        self._start_time = time.perf_counter()
        if self.method == "rapl_measured":
            self._samples = [self._read_sample()]
            self._stop_event = threading.Event()
            self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler_thread.start()
        else:
            self._samples = []

    def stop(self) -> dict:
        """Returns {"cpu_kwh", "method", "phase", "samples"}."""
        elapsed_s = time.perf_counter() - self._start_time

        if self.method == "rapl_measured":
            self._stop_event.set()
            self._sampler_thread.join(timeout=self._sample_interval_s * 2)
            self._samples.append(self._read_sample())
            raw_kwh = _trapezoidal_energy_kwh(self._samples)
            idle_kwh = self._idle_watts * (elapsed_s / 3600.0) / 1000.0  # W*h -> Wh -> kWh
            cpu_kwh = max(raw_kwh - idle_kwh, 0.0)
        else:
            # estimated_constant_power: no sensor, so no idle baseline to subtract either.
            cpu_kwh = self._estimated_tdp_watts * (elapsed_s / 3600.0) / 1000.0  # W*h -> Wh -> kWh

        return {"cpu_kwh": cpu_kwh, "method": self.method, "phase": self._phase, "samples": self._samples}

    def close(self):
        if self._computer is not None:
            self._computer.Close()

    def set_idle_watts(self, idle_watts: float):
        """Lets main.py configure the idle baseline (spec §10) after
        measuring it with this same instance, instead of constructing a
        second CPUEnergyMonitor (which would re-open LibreHardwareMonitor
        sensors unnecessarily)."""
        self._idle_watts = idle_watts


def _trapezoidal_energy_kwh(samples) -> float:
    if len(samples) < 2:
        return 0.0
    energy_wh = 0.0
    for s0, s1 in zip(samples, samples[1:]):
        dt_h = (s1["perf_t"] - s0["perf_t"]) / 3600.0
        energy_wh += (s0["watts"] + s1["watts"]) / 2.0 * dt_h
    return energy_wh / 1000.0


def measure_idle_cpu_watts(monitor: CPUEnergyMonitor, duration_s: int = 60,
                            sample_interval_s: float = 1.0) -> dict:
    """One-time idle baseline (spec §10). Returns {"mean_watts", "samples"};
    in estimated mode there is no sensor to baseline against, so this
    returns 0.0 watts and no samples."""
    if monitor.method != "rapl_measured":
        return {"mean_watts": 0.0, "samples": []}
    samples = []
    elapsed = 0.0
    while elapsed < duration_s:
        samples.append({
            "perf_t": time.perf_counter(),
            "wall_t": time.time(),
            "watts": monitor.read_current_watts(),
        })
        time.sleep(sample_interval_s)
        elapsed += sample_interval_s
    mean_watts = sum(s["watts"] for s in samples) / len(samples) if samples else 0.0
    return {"mean_watts": mean_watts, "samples": samples}
