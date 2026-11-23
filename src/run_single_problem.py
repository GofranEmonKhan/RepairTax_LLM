"""
Run exactly one problem without modifying src.main.

Usage:
    python -m src.run_single_problem HR_M_001
"""

import glob
import hashlib
import importlib.metadata
import json
import os
import platform
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
from src.result_logger import (
    save_experiment_metadata,
    append_sensor_samples,
    write_analysis_csvs,
)


def load_problem(problem_id, problems_dir):
    """Load exactly one problem by problem_id."""
    for path in sorted(glob.glob(os.path.join(problems_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            problem = json.load(f)

        if problem.get("problem_id") == problem_id:
            return path, problem

    return None, None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def model_slug(model_name):
    return model_name.split("/")[-1]


def next_run_id(model_name, *base_dirs):
    """Find next free run_NNN for the selected model."""
    slug = model_slug(model_name)
    used = [0]

    for base in base_dirs:
        model_dir = os.path.join(base, slug)

        if os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                if name.startswith("run_"):
                    try:
                        used.append(int(name.split("_")[1]))
                    except (ValueError, IndexError):
                        pass

    return f"run_{max(used) + 1:03d}"


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("    python -m src.run_single_problem HR_M_001")
        sys.exit(1)

    problem_id = sys.argv[1]

    config = load_config("config.yaml")

    problems_dir = config["paths"]["problems_dir"]
    experiments_base = config["paths"]["experiments_dir"]
    results_base = config["paths"]["results_dir"]

    model_name = config["model"]["name"]
    model_dir_name = model_slug(model_name)

    run_id = next_run_id(
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

    problem_path, problem = load_problem(problem_id, problems_dir)

    if problem is None:
        print(f"Problem '{problem_id}' was not found in {problems_dir}/")
        sys.exit(1)

    os.makedirs(experiments_dir, exist_ok=False)
    os.makedirs(results_dir, exist_ok=False)

    print()
    print("=" * 70)
    print(f"Single-problem experiment")
    print(f"Model:       {model_name}")
    print(f"Problem:     {problem_id}")
    print(f"Title:       {problem.get('title', '')}")
    print(f"Run:         {run_id}")
    print(f"Experiment:  {experiment_id}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # Energy monitors
    # ------------------------------------------------------------

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

        print(
            f"Measuring idle baseline over {window}s "
            f"(leave the machine idle) ..."
        )

        idle_gpu = measure_idle_gpu_watts(
            config["hardware"]["gpu_index"],
            duration_s=window,
        )

        idle_cpu = measure_idle_cpu_watts(
            cpu_monitor,
            duration_s=window,
        )

        idle_gpu_watts = idle_gpu["mean_watts"]
        idle_cpu_watts = idle_cpu["mean_watts"]

        gpu_monitor.set_idle_watts(idle_gpu_watts)
        cpu_monitor.set_idle_watts(idle_cpu_watts)

        append_sensor_samples(
            results_dir,
            experiment_id,
            "idle_baseline",
            "idle_baseline",
            "gpu",
            idle_gpu["samples"],
        )

        append_sensor_samples(
            results_dir,
            experiment_id,
            "idle_baseline",
            "idle_baseline",
            "cpu",
            idle_cpu["samples"],
        )

    print(
        f"Idle baseline: GPU {idle_gpu_watts:.2f} W, "
        f"CPU {idle_cpu_watts:.2f} W"
    )

    print(
        f"GPU energy method: {gpu_monitor.method} | "
        f"CPU energy method: {cpu_monitor.method}"
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print(
        f"Loading model {config['model']['name']} "
        f"({config['model']['quantization']}) ..."
    )

    llm = LLMClient(
        model_name=config["model"]["name"],
        device_map=config["model"]["device_map"],
        temperature=config["model"]["temperature"],
        max_new_tokens=config["model"]["max_new_tokens"],
        seed=config["model"].get("seed"),
    )

    prompt_builder = PromptBuilder(
        config["paths"]["prompts_dir"]
    )

    # ------------------------------------------------------------
    # Docker
    # ------------------------------------------------------------

    docker_client = docker.from_env()

    ensure_image_built(
        docker_client,
        config["sandbox"]["docker_image"],
        "docker",
    )

    # ------------------------------------------------------------
    # Compiler version
    # ------------------------------------------------------------

    with Sandbox(
        docker_client,
        config["sandbox"]["docker_image"],
        memory_limit=config["sandbox"]["memory_limit"],
        nano_cpus=config["sandbox"]["nano_cpus"],
    ) as probe:

        version_lines = probe.run(
            "g++ --version",
            config["sandbox"]["compile_timeout_seconds"],
        )["stdout"].strip().splitlines()

    compiler_version = version_lines[0] if version_lines else None

    # ------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------

    prompt_paths = sorted(
        glob.glob(
            os.path.join(
                config["paths"]["prompts_dir"],
                "*.txt",
            )
        )
    )

    shutil.copyfile(
        "config.yaml",
        os.path.join(
            results_dir,
            "config_snapshot.yaml",
        ),
    )

    save_experiment_metadata(
        results_dir,
        {
            "experiment_id": experiment_id,
            "experiment_type": "single_problem_debug",
            "run_started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

            "target_problem": {
                "problem_id": problem_id,
                "source_file": problem_path,
            },

            "hardware": dict(config["hardware"]),

            "software": {
                "gpu_driver_version": gpu_monitor.driver_version,
                "docker_version": docker_client.version().get("Version"),
                "compiler_version": compiler_version,
                "python_version": platform.python_version(),
                "packages": {
                    n: package_version(n)
                    for n in (
                        "torch",
                        "transformers",
                        "bitsandbytes",
                        "accelerate",
                        "nvidia-ml-py",
                        "pythonnet",
                        "docker",
                    )
                },
            },

            "model": {
                "name": config["model"]["name"],
                "quantization": config["model"]["quantization"],
                "temperature": config["model"]["temperature"],
                "max_new_tokens": config["model"]["max_new_tokens"],
                "seed": config["model"].get("seed"),
            },

            "measurement": {
                "gpu_energy_method": gpu_monitor.method,
                "gpu_sampling_interval_ms":
                    config["energy"]["gpu"]["sample_interval_ms"],

                "cpu_energy_method": cpu_monitor.method,
                "cpu_sampling_interval_ms":
                    config["energy"]["cpu"]["sample_interval_ms"],

                "idle_baseline": {
                    "gpu_watts": idle_gpu_watts,
                    "cpu_watts": idle_cpu_watts,
                },

                "carbon_intensity_g_per_kwh":
                    config["carbon"]["intensity_g_per_kwh"],

                "carbon_intensity_source":
                    config["carbon"]["intensity_source"],
            },

            "experiment_settings": {
                "max_attempts":
                    config["experiment"]["max_attempts"],

                "held_out_test_set":
                    config["experiment"]["held_out_test_set"],

                "failing_test_selection":
                    config["failing_test_selection"],

                "test_timeout_seconds":
                    config["sandbox"]["test_timeout_seconds"],

                "problem_count": 1,
            },

            "reproducibility": {
                "config_sha256":
                    sha256_of_file("config.yaml"),

                "prompts_sha256": {
                    os.path.basename(pp):
                        sha256_of_file(pp)
                    for pp in prompt_paths
                },

                "dataset_sha256": {
                    os.path.basename(problem_path):
                        sha256_of_file(problem_path)
                },
            },
        },
    )

    # ------------------------------------------------------------
    # Run exactly one problem
    # ------------------------------------------------------------

    try:
        print(
            f"[1/1] {problem['problem_id']} — "
            f"{problem.get('title', '')}"
        )

        with Sandbox(
            docker_client,
            config["sandbox"]["docker_image"],
            memory_limit=config["sandbox"]["memory_limit"],
            nano_cpus=config["sandbox"]["nano_cpus"],
        ) as sandbox:

            result = run_problem(
                problem,
                llm,
                prompt_builder,
                sandbox,
                gpu_monitor,
                cpu_monitor,
                config,
                experiments_dir,
                results_dir,
                experiment_id,
            )

            status = (
                "PASSED"
                if result["result"]["successful"]
                else "FAILED"
            )

            print(
                f"    -> {status} "
                f"(attempts: {result['result']['total_attempts']}, "
                f"energy: "
                f"{result['cumulative']['total_energy_kwh']:.6f} kWh)"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as exc:
        print(
            f"    -> ERROR while running problem "
            f"{problem_id}: {exc}"
        )

    finally:
        write_analysis_csvs(results_dir)
        cpu_monitor.close()

    print()
    print("=" * 70)
    print("Single-problem run finished.")
    print(f"Experiment files: {experiments_dir}")
    print(f"Result files:     {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()