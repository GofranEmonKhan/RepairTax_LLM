"""
GPU energy measurement via NVML (spec §2, §10).

Authoritative energy figure: nvmlDeviceGetTotalEnergyConsumption() (an
onboard cumulative joule counter) whenever the card/driver expose it;
otherwise nvmlDeviceGetPowerUsage() samples trapezoidal-integrated.
Regardless of which method is authoritative, a background sampler ALWAYS
runs during every phase and its raw readings are returned from stop(), so
they can be written to sensor_samples.csv — every reported figure stays
independently re-derivable from raw readings.

Create ONE GPUEnergyMonitor per run and call start(phase)/stop() around
each measured phase (see orchestrator.py). `.method` is readable right
after construction (before any phase runs), so it can go into
experiment_metadata.json.
"""

import time
import threading

try:
    import pynvml
except ImportError:  # pragma: no cover - only when the dependency is missing
    pynvml = None


class GPUEnergyMonitor:
    def __init__(self, gpu_index: int = 0, sample_interval_ms: int = 100, idle_watts: float = 0.0):
        if pynvml is None:
            raise RuntimeError("pynvml (nvidia-ml-py) is not installed — GPU energy cannot be measured.")
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        driver = pynvml.nvmlSystemGetDriverVersion()
        self.driver_version = driver.decode() if isinstance(driver, bytes) else driver
        self._sample_interval_s = sample_interval_ms / 1000.0
        self._idle_watts = idle_watts
        self.method = "nvml_energy_counter" if self._counter_supported() else "nvml_power_sampled"

        self._phase = None
        self._start_time = None
        self._start_energy_mj = None
        self._samples = []
        self._stop_event = None
        self._sampler_thread = None

    def _counter_supported(self) -> bool:
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            return True
        except pynvml.NVMLError:
            return False

    def _read_sample(self) -> dict:
        return {
            "perf_t": time.perf_counter(),
            "wall_t": time.time(),
            "watts": pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
        }

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self._samples.append(self._read_sample())
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self._sample_interval_s)

    def start(self, phase: str = "unspecified"):
        self._phase = phase
        self._start_time = time.perf_counter()
        self._samples = [self._read_sample()]
        if self.method == "nvml_energy_counter":
            self._start_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self._stop_event = threading.Event()
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()

    def stop(self) -> dict:
        """Returns {"gpu_kwh", "method", "phase", "samples"}. `samples` is the
        raw [{"perf_t","wall_t","watts"}, ...] list for this phase. Idle
        baseline watts (set at construction / via set_idle_watts) are
        subtracted from the authoritative figure (spec §10)."""
        elapsed_s = time.perf_counter() - self._start_time
        self._stop_event.set()
        self._sampler_thread.join(timeout=self._sample_interval_s * 2)
        try:
            self._samples.append(self._read_sample())
        except pynvml.NVMLError:
            pass

        if self.method == "nvml_energy_counter":
            end_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            raw_kwh = (end_energy_mj - self._start_energy_mj) / 1000.0 / 3.6e6  # mJ -> J -> kWh
        else:
            raw_kwh = _trapezoidal_energy_kwh(self._samples)

        idle_kwh = self._idle_watts * (elapsed_s / 3600.0) / 1000.0  # W*h -> Wh -> kWh
        return {
            "gpu_kwh": max(raw_kwh - idle_kwh, 0.0),
            "method": self.method,
            "phase": self._phase,
            "samples": self._samples,
        }

    def set_idle_watts(self, idle_watts: float):
        """Lets main.py configure the idle baseline (spec §10) after
        measuring it, without constructing a second monitor instance."""
        self._idle_watts = idle_watts


def _trapezoidal_energy_kwh(samples) -> float:
    """samples: [{"perf_t", "watts", ...}, ...] -> energy in kWh."""
    if len(samples) < 2:
        return 0.0
    energy_wh = 0.0
    for s0, s1 in zip(samples, samples[1:]):
        dt_h = (s1["perf_t"] - s0["perf_t"]) / 3600.0
        energy_wh += (s0["watts"] + s1["watts"]) / 2.0 * dt_h
    return energy_wh / 1000.0


def measure_idle_gpu_watts(gpu_index: int = 0, duration_s: int = 60, sample_interval_s: float = 1.0) -> dict:
    """One-time idle baseline (spec §10) — run at experiment startup with
    nothing else using the GPU. Returns {"mean_watts", "samples"} so the
    raw readings can be logged under phase "idle_baseline"."""
    if pynvml is None:
        raise RuntimeError("pynvml (nvidia-ml-py) is not installed.")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    elapsed = 0.0
    while elapsed < duration_s:
        samples.append({
            "perf_t": time.perf_counter(),
            "wall_t": time.time(),
            "watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        })
        time.sleep(sample_interval_s)
        elapsed += sample_interval_s
    mean_watts = sum(s["watts"] for s in samples) / len(samples) if samples else 0.0
    return {"mean_watts": mean_watts, "samples": samples}
