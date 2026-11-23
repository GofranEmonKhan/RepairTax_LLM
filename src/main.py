"""
Entry point (spec §14, §15, §19).

Usage (run from the project root, so relative paths in config.yaml resolve):
    python -m src.main                       # full run over every problem in data/problems/
    python -m src.main --pilot               # only the first `pilot.problem_count` problems
    python -m src.main --limit 10            # only the first 10 problems (overrides --pilot)
    python -m src.main --experiment-id NAME  # name the run yourself (must not already exist)

Each model gets its own directory, and every run gets a sequential run_id:
    experiments/<model_name>/run_001/
    experiments/<model_name>/run_002/

and similarly:
    results/<model_name>/run_001/
    results/<model_name>/run_002/

This keeps repeated runs of the same model together while keeping different
models completely separated.
"""

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time

import docker

from src.config_loader import load_config
from src.prompt_builder import PromptBuilder
from src.llm_client import LLMClient
from src.energy_monitor_gpu import GPUEnergyMonitor, measure_idle_gpu_watts
from src.energy_monitor_cpu import CPUEnergyMonitor, measure_idle_cpu_watts
from src.docker_sandbox import Sandbox, ensure_image_built
from src.orchestrator import run_problem
from src.result_logger import save_experiment_metadata, append_sensor_samples, write_analysis_csvs

_RUN_ID_RE = re.compile(r"^run_(\d+)$")


def load_problem_files(problems_dir: str) -> list:
    """Loads every *.json in problems_dir as (path, problem) pairs, sorted by
    filename for a deterministic run order (dataset schema, spec §3)."""
    pairs = []
    for path in sorted(glob.glob(os.path.join(problems_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            pairs.append((path, json.load(f)))
    return pairs


def sha256_of_file(path: str) -> str:
    """Plain SHA-256 of one file's bytes — reproducible with standard tools
    (e.g. `certutil -hashfile config.yaml SHA256` on Windows)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_slug(model_name: str) -> str:
    """Convert a Hugging Face model name into a filesystem-safe model directory name.

    Example:
        Qwen/Qwen2.5-7B-Instruct
        -> Qwen2.5-7B-Instruct
    """
    slug = model_name.split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug


def next_run_id(model_name: str, *base_dirs: str) -> str:
    """Return the next free run_NNN for a specific model.

    The same run number is kept synchronized across experiments/ and results/.
    """
    slug = model_slug(model_name)

    used = [0]

    for base in base_dirs:
        model_dir = os.path.join(base, slug)

        if os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                m = _RUN_ID_RE.match(name)
                if m:
                    used.append(int(m.group(1)))

    return f"run_{max(used) + 1:03d}"


def main():
    parser = argparse.ArgumentParser(description="LLM code-correction energy/carbon experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true", help="Run only config.pilot.problem_count problems")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N problems")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Name this run (default: next free run_NNN for the selected model). "
             "Must not already exist."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    problems_dir = config["paths"]["problems_dir"]
    experiments_base = config["paths"]["experiments_dir"]
    results_base = config["paths"]["results_dir"]

    model_name = config["model"]["name"]
    model_dir_name = model_slug(model_name)

    run_id = args.experiment_id or next_run_id(
        model_name,
        experiments_base,
        results_base,
    )

    experiments_dir = os.path.join(
        experiments_base,
        model_dir_name,
        run_id,
    )

    results_dir = os.path.join(
        results_base,
        model_dir_name,
        run_id,
    )

    experiment_id = f"{model_dir_name}/{run_id}"

    if os.path.exists(experiments_dir) or os.path.exists(results_dir):
        print(f"Experiment '{experiment_id}' already exists — refusing to write into it. "
              f"Pick another --experiment-id or omit it to auto-generate one.")
        sys.exit(1)
    os.makedirs(experiments_dir, exist_ok=False)
    os.makedirs(results_dir, exist_ok=False)

    problem_files = load_problem_files(problems_dir)
    if not problem_files:
        print(f"No problem JSON files found in {problems_dir}/ — nothing to run.")
        sys.exit(1)

    if args.limit is not None:
        problem_files = problem_files[: args.limit]
        experiment_type = "limited"
    elif args.pilot:
        problem_files = problem_files[: config["pilot"]["problem_count"]]
        experiment_type = "pilot"
    else:
        experiment_type = "full"
    problems = [p for _, p in problem_files]

    print(f"Experiment {experiment_id} ({experiment_type}): {len(problems)} problem(s) from {problems_dir}/")

    # --- One-time setup: energy monitors, idle baseline, model, sandbox image ---
    gpu_monitor = GPUEnergyMonitor(
        gpu_index=config["hardware"]["gpu_index"],
        sample_interval_ms=config["energy"]["gpu"]["sample_interval_ms"],
    )
    cpu_monitor = CPUEnergyMonitor(
        dll_path=config["energy"]["cpu"]["librehardwaremonitor_dll_path"],
        sample_interval_ms=config["energy"]["cpu"]["sample_interval_ms"],
        estimated_tdp_watts=config["energy"]["cpu"]["estimated_tdp_watts"],
    )

    idle_gpu_watts = 0.0
    idle_cpu_watts = 0.0
    if config["idle_baseline"]["measure_on_startup"]:
        window = config["idle_baseline"]["window_seconds"]
        print(f"Measuring idle baseline over {window}s (spec §10) — leave the machine idle ...")
        idle_gpu = measure_idle_gpu_watts(config["hardware"]["gpu_index"], duration_s=window)
        idle_cpu = measure_idle_cpu_watts(cpu_monitor, duration_s=window)
        idle_gpu_watts, idle_cpu_watts = idle_gpu["mean_watts"], idle_cpu["mean_watts"]
        gpu_monitor.set_idle_watts(idle_gpu_watts)
        cpu_monitor.set_idle_watts(idle_cpu_watts)
        append_sensor_samples(results_dir, experiment_id, "idle_baseline", "idle_baseline", "gpu", idle_gpu["samples"])
        append_sensor_samples(results_dir, experiment_id, "idle_baseline", "idle_baseline", "cpu", idle_cpu["samples"])

    print(f"Idle baseline: GPU {idle_gpu_watts:.2f} W, CPU {idle_cpu_watts:.2f} W")
    print(f"GPU energy method: {gpu_monitor.method} | CPU energy method: {cpu_monitor.method}")

    print(f"Loading model {config['model']['name']} ({config['model']['quantization']}) ...")
    llm = LLMClient(
        model_name=config["model"]["name"], device_map=config["model"]["device_map"],
        temperature=config["model"]["temperature"], max_new_tokens=config["model"]["max_new_tokens"],
        seed=config["model"].get("seed"),
    )

    prompt_builder = PromptBuilder(config["paths"]["prompts_dir"])

    docker_client = docker.from_env()
    ensure_image_built(docker_client, config["sandbox"]["docker_image"], "docker")

    # Software/reproducibility fingerprint. The sandbox probe below also confirms the
    # image can start a container and that g++ works, before any model time is spent.
    with Sandbox(docker_client, config["sandbox"]["docker_image"],
                 memory_limit=config["sandbox"]["memory_limit"],
                 nano_cpus=config["sandbox"]["nano_cpus"]) as probe:
        version_lines = probe.run("g++ --version", config["sandbox"]["compile_timeout_seconds"])["stdout"].strip().splitlines()
    compiler_version = version_lines[0] if version_lines else None

    prompt_paths = sorted(glob.glob(os.path.join(config["paths"]["prompts_dir"], "*.txt")))
    shutil.copyfile(args.config, os.path.join(results_dir, "config_snapshot.yaml"))

    save_experiment_metadata(results_dir, {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": dict(config["hardware"]),
        "software": {
            "gpu_driver_version": gpu_monitor.driver_version,
            "docker_version": docker_client.version().get("Version"),
            "compiler_version": compiler_version,
            "python_version": platform.python_version(),
            "packages": {n: package_version(n) for n in
                         ("torch", "transformers", "bitsandbytes", "accelerate", "nvidia-ml-py", "pythonnet", "docker")},
        },
        "model": {
            "name": config["model"]["name"], "quantization": config["model"]["quantization"],
            "temperature": config["model"]["temperature"],
            "max_new_tokens": config["model"]["max_new_tokens"],
            "seed": config["model"].get("seed"),
        },
        "measurement": {
            "gpu_energy_method": gpu_monitor.method,
            "gpu_sampling_interval_ms": config["energy"]["gpu"]["sample_interval_ms"],
            "cpu_energy_method": cpu_monitor.method,
            "cpu_sampling_interval_ms": config["energy"]["cpu"]["sample_interval_ms"],
            "idle_baseline": {"gpu_watts": idle_gpu_watts, "cpu_watts": idle_cpu_watts},
            "carbon_intensity_g_per_kwh": config["carbon"]["intensity_g_per_kwh"],
            "carbon_intensity_source": config["carbon"]["intensity_source"],
        },
        "experiment_settings": {
            "max_attempts": config["experiment"]["max_attempts"],
            "held_out_test_set": config["experiment"]["held_out_test_set"],
            "failing_test_selection": config["failing_test_selection"],
            "test_timeout_seconds": config["sandbox"]["test_timeout_seconds"],
            "problem_count": len(problems),
        },
        # Per-file SHA-256, so any file can be re-verified with a standard tool and a
        # change between two runs pinpoints exactly which prompt/problem file differed.
        "reproducibility": {
            "config_sha256": sha256_of_file(args.config),
            "prompts_sha256": {os.path.basename(pp): sha256_of_file(pp) for pp in prompt_paths},
            "dataset_sha256": {os.path.basename(path): sha256_of_file(path) for path, _ in problem_files},
        },
    })

    # --- Run every problem, one warm sandbox container per problem (spec §12) ---
    try:
        for i, problem in enumerate(problems, start=1):
            print(f"[{i}/{len(problems)}] {problem['problem_id']} — {problem.get('title', '')}")
            with Sandbox(docker_client, config["sandbox"]["docker_image"],
                         memory_limit=config["sandbox"]["memory_limit"],
                         nano_cpus=config["sandbox"]["nano_cpus"]) as sandbox:
                try:
                    result = run_problem(problem, llm, prompt_builder, sandbox, gpu_monitor, cpu_monitor,
                                          config, experiments_dir, results_dir, experiment_id)
                    status = "PASSED" if result["result"]["successful"] else "FAILED"
                    print(f"    -> {status} (attempts: {result['result']['total_attempts']}, "
                          f"energy: {result['cumulative']['total_energy_kwh']:.6f} kWh)")
                except Exception as exc:
                    print(f"    -> ERROR while running problem {problem['problem_id']}: {exc}")
                    continue
    finally:
        # Runs even on Ctrl+C, so whatever finished is still turned into analysis tables.
        write_analysis_csvs(results_dir)
        cpu_monitor.close()

    print(f"\nDone. Results in {results_dir}/ (analysis tables in {results_dir}/analysis/), "
          f"per-attempt files in {experiments_dir}/")


if __name__ == "__main__":
    main()