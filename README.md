# green-llm-code

Implementation of `llm-code-correction-energy-carbon-spec.md`. Measures the
energy and carbon cost of iterative LLM-based C++17 code generation and
correction, running locally on the Lenovo LOQ 15IRX10 (RTX 5050 8GB, i7-13650HX,
16GB RAM) under Windows 11.

## 1. Before you run anything

Things in `config.yaml` that need your attention â€” the pipeline refuses to start
(`config_loader.py`) until the carbon one is set:

1. **`carbon.intensity_g_per_kwh` and `carbon.intensity_source`** â€” pick and
   cite one documented source for the Bangladesh grid (spec Â§11). Two
   candidates were found while drafting the spec: Ember/Our World in Data
   2024 (~562 gCOâ‚‚/kWh) and an aggregated 2025 estimate (~696 gCOâ‚‚/kWh) â€”
   they disagree, so don't guess; pick one and record it.
2. **`energy.cpu.librehardwaremonitor_dll_path`** â€” where you put
   `LibreHardwareMonitorLib.dll` (step 4 below).
3. **`hardware.windows_power_mode`, `hardware.lenovo_thermal_mode`,
   `hardware.battery_conservation`** â€” the pipeline can't detect these; write
   what they are set to before each run (they're copied into the run's metadata).
4. **`model.name` / `model.quantization`** and **`pilot.problem_count`** â€”
   both are defaults from spec Â§18, not confirmed decisions. Change them if
   you want something other than Qwen2.5-7B-Instruct (4-bit NF4) / a 5-problem pilot.

## 2. Install

```bash
# 1. Install a CUDA-enabled torch build FIRST â€” a plain `pip install torch`
#    gives you a CPU-only build. Pick the right command for your driver at
#    https://pytorch.org/get-started/locally/

# 2. Then everything else:
pip install -r requirements.txt
```

## 3. Docker Desktop

Install Docker Desktop and make sure it's running before you start the
pipeline â€” `src/main.py` builds the sandbox image (`docker/Dockerfile`)
automatically on first run if it isn't already present.

## 4. CPU energy measurement (RAPL via LibreHardwareMonitor)

1. Download `LibreHardwareMonitorLib.dll` from the
   [LibreHardwareMonitor releases page](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases).
2. Put it somewhere stable, e.g. `C:\LibreHardwareMonitor\LibreHardwareMonitorLib.dll`.
3. Set `energy.cpu.librehardwaremonitor_dll_path` in `config.yaml` to that path.
4. **This is not guaranteed to work on this CPU/OS combination** â€” reading
   RAPL on Windows depends on BIOS/driver permissions (spec Â§2, Â§19). If it
   can't be loaded or the CPU Package power sensor can't be read, the
   pipeline automatically falls back to `estimated_constant_power`
   (`energy.cpu.estimated_tdp_watts` Ã— elapsed time) and prints why. Check
   the console output and `results/experiment_metadata.json` after your
   pilot run to see which one actually happened â€” don't assume.

## 5. GPU: force the discrete card

In Windows Graphics Settings, set your Python interpreter (or the terminal
you run it from) to "High performance NVIDIA" â€” otherwise Optimus can route
the workload to the Intel iGPU while NVML measures an idle RTX 5050 (spec Â§2).

## 6. Add your problems

Drop your problem JSON files into `data/problems/` â€” one file per problem,
matching the schema in spec Â§3 (see `data/problems/HR_M_001.json`, already
included, for the exact shape). Filenames don't matter; problems are loaded
alphabetically by filename for a deterministic run order.

## 7. Run

```bash
# From the project root (relative paths in config.yaml depend on this):

# Pilot first â€” confirms sensors, VRAM, RAM, carbon source before the full run (spec Â§19)
python -m src.main --pilot

# Full run over everything in data/problems/
python -m src.main

# Or an arbitrary subset
python -m src.main --limit 10

# Name a run yourself (must not already exist)
python -m src.main --pilot --experiment-id EXP_PILOT_A
```

Every run gets its own experiment ID â€” `EXP_001`, `EXP_002`, ... assigned
automatically. The pipeline **refuses** to write into an ID that already exists,
so a pilot and a later full run can never overwrite or mix into each other.
Don't delete pilot output: if you change the measurement configuration, that's a
new experiment ID by design.

## 8. Where results go

```
experiments/EXP_001/<problem_id>/attempt_00N/{solution.cpp, prompt.txt, attempt.json}
results/EXP_001/
    experiment_metadata.json      hardware, model, seed, GPU/CPU energy method, idle baseline, carbon factor
    raw_attempts.jsonl            one record per attempt (append-only)
    problem_results.jsonl         one cumulative record per problem (append-only)
    sensor_samples.csv            raw power readings per attempt/phase/sensor (audit trail)
    analysis/attempt_level.csv
    analysis/problem_level.csv
    analysis/summary_by_difficulty.csv
```

`sensor_samples.csv` holds every raw watt reading behind every energy figure
(phases: `idle_baseline`, `generation`, `correction_generation`, `compilation`,
`execution`; GPU and CPU are separate rows). In estimated-CPU mode there is no
sensor, so no CPU rows exist by design.

The three analysis CSVs are regenerated automatically at the end of every run â€”
including when you stop it early with Ctrl+C.

## 9. Before trusting a full 50â€“60-problem run

Run the pilot checklist in spec Â§19 first. After the pilot, open
`results/EXP_001/experiment_metadata.json` and check `measurement.gpu_energy_method`
(`nvml_energy_counter` = the onboard counter works; `nvml_power_sampled` = fallback)
and `measurement.cpu_energy_method` (`rapl_measured` vs `estimated_constant_power`)
â€” both are also printed to the console at startup. Then check peak VRAM/RAM under
load and idle-baseline stability across a couple of runs.

`model.seed` in `config.yaml` (default 42) is applied once at startup so a run is
reproducible; set it to `null` for unseeded sampling.

