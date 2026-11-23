"""
Persists per-attempt files and the run-level results files (spec §8, §9, §15).

Layout (both base dirs are experiment-scoped by main.py, so a pilot run and a
later full run never share — or silently corrupt — the same files):

  experiments/<experiment_id>/<problem_id>/attempt_00N/{solution.cpp, prompt.txt, attempt.json}
  results/<experiment_id>/raw_attempts.jsonl
  results/<experiment_id>/problem_results.jsonl
  results/<experiment_id>/experiment_metadata.json
  results/<experiment_id>/sensor_samples.csv
  results/<experiment_id>/analysis/{attempt_level,problem_level,summary_by_difficulty}.csv

raw_attempts.jsonl, problem_results.jsonl and sensor_samples.csv are
append-only (one record/row at a time) rather than load-modify-rewrite, so a
crash mid-write can't corrupt already-written records.
"""

import csv
import datetime
import json
import os
import statistics


def save_attempt_files(experiments_dir: str, problem_id: str, attempt_number: int,
                        raw_response: str, source_code: str, prompt_text: str, attempt_record: dict):
    attempt_dir = os.path.join(experiments_dir, problem_id, f"attempt_{attempt_number:03d}")
    os.makedirs(attempt_dir, exist_ok=True)

    with open(os.path.join(attempt_dir, "solution.cpp"), "w", encoding="utf-8") as f:
        f.write(source_code)
    with open(os.path.join(attempt_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)
    with open(os.path.join(attempt_dir, "attempt.json"), "w", encoding="utf-8") as f:
        json.dump({**attempt_record, "raw_response": raw_response}, f, indent=2)


def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_attempt_to_raw_log(results_dir: str, attempt_record: dict):
    _append_jsonl(os.path.join(results_dir, "raw_attempts.jsonl"), attempt_record)


def save_problem_result(results_dir: str, problem_result: dict):
    _append_jsonl(os.path.join(results_dir, "problem_results.jsonl"), problem_result)


def save_experiment_metadata(results_dir: str, metadata: dict):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def append_sensor_samples(results_dir: str, experiment_id: str, attempt_id: str, phase: str,
                           sensor: str, samples: list):
    """Appends raw readings to sensor_samples.csv — one row per reading, one
    sensor per row (gpu and cpu are sampled independently, so they are not
    merged into one row). The energy figure for any attempt/phase/sensor can
    be re-derived from these rows by trapezoidal integration.
    `attempt_id` is "idle_baseline" for the startup baseline measurement."""
    path = os.path.join(results_dir, "sensor_samples.csv")
    os.makedirs(results_dir, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment_id", "attempt_id", "phase", "sensor", "timestamp", "power_w"])
        for s in samples:
            ts = datetime.datetime.fromtimestamp(s["wall_t"]).isoformat(timespec="milliseconds")
            writer.writerow([experiment_id, attempt_id, phase, sensor, ts, f"{s['watts']:.4f}"])


def write_analysis_csvs(results_dir: str):
    """Reads raw_attempts.jsonl and problem_results.jsonl and writes the three
    analysis tables under results/<experiment_id>/analysis/."""
    attempts = _read_jsonl(os.path.join(results_dir, "raw_attempts.jsonl"))
    problems = _read_jsonl(os.path.join(results_dir, "problem_results.jsonl"))

    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    _write_attempt_level_csv(analysis_dir, attempts)
    _write_problem_level_csv(analysis_dir, problems)
    _write_summary_by_difficulty_csv(analysis_dir, problems)


def _write_attempt_level_csv(analysis_dir: str, attempts: list):
    fieldnames = [
        "problem_id", "attempt_id", "attempt_number", "attempt_type", "difficulty",
        "generation_duration_s", "prompt_tokens", "completion_tokens",
        "gpu_energy_kwh", "cpu_energy_kwh", "cpu_energy_method", "generation_carbon_gco2e",
        "compile_status", "test_status", "failure_type",
    ]
    with open(os.path.join(analysis_dir, "attempt_level.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in attempts:
            g, c, e = a["generation"], a["compilation"], a["execution"]
            if e is None:
                test_status = "not_run_compile_failed"
            else:
                test_status = "passed" if a["failure_type"] is None else a["failure_type"]
            writer.writerow({
                "problem_id": a["problem_id"], "attempt_id": a["attempt_id"],
                "attempt_number": a["attempt_number"], "attempt_type": a["attempt_type"],
                "difficulty": a["difficulty"],
                "generation_duration_s": g["generation_time_seconds"],
                "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["generated_tokens"],
                "gpu_energy_kwh": g["energy"]["gpu_kwh"], "cpu_energy_kwh": g["energy"]["cpu_kwh"],
                "cpu_energy_method": g["energy"]["cpu_energy_method"],
                "generation_carbon_gco2e": g["carbon_gco2e"],
                "compile_status": "success" if c["success"] else "failed",
                "test_status": test_status,
                "failure_type": a["failure_type"],
            })


def _write_problem_level_csv(analysis_dir: str, problems: list):
    fieldnames = [
        "problem_id", "difficulty", "total_attempts", "total_correction_attempts", "solved",
        "successful_attempt_number",
        "cumulative_generation_energy_kwh", "cumulative_compilation_energy_kwh",
        "cumulative_execution_energy_kwh", "total_energy_kwh", "cumulative_carbon_gco2e",
        "successful_attempt_energy_kwh", "successful_attempt_carbon_gco2e",
        "correction_energy_overhead_pct",
    ]
    with open(os.path.join(analysis_dir, "problem_level.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in problems:
            merged = p.get("successful_attempt_merged") or {}
            writer.writerow({
                "problem_id": p["problem_id"], "difficulty": p.get("difficulty"),
                "total_attempts": p["result"]["total_attempts"],
                "total_correction_attempts": p["result"]["total_correction_attempts"],
                "solved": p["result"]["successful"],
                "successful_attempt_number": p["result"]["successful_attempt"],
                "cumulative_generation_energy_kwh": p["cumulative"]["generation_energy_kwh"],
                "cumulative_compilation_energy_kwh": p["cumulative"]["compilation_energy_kwh"],
                "cumulative_execution_energy_kwh": p["cumulative"]["execution_energy_kwh"],
                "total_energy_kwh": p["cumulative"]["total_energy_kwh"],
                "cumulative_carbon_gco2e": p["cumulative"]["total_carbon_gco2e"],
                "successful_attempt_energy_kwh": merged.get("energy_kwh"),
                "successful_attempt_carbon_gco2e": merged.get("carbon_gco2e"),
                "correction_energy_overhead_pct": p["correction_energy_overhead_pct"],
            })


def _write_summary_by_difficulty_csv(analysis_dir: str, problems: list):
    fieldnames = [
        "difficulty", "problem_count", "pass_rate", "mean_attempts",
        "mean_generation_energy_kwh", "mean_execution_energy_kwh",
        "mean_cumulative_energy_kwh", "median_cumulative_energy_kwh",
        "stddev_cumulative_energy_kwh", "mean_carbon_gco2e",
    ]
    groups = {}
    for p in problems:
        groups.setdefault(p.get("difficulty"), []).append(p)

    with open(os.path.join(analysis_dir, "summary_by_difficulty.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty, group in groups.items():
            energies = [g["cumulative"]["total_energy_kwh"] for g in group]
            writer.writerow({
                "difficulty": difficulty,
                "problem_count": len(group),
                "pass_rate": sum(1 for g in group if g["result"]["successful"]) / len(group),
                "mean_attempts": statistics.mean(g["result"]["total_attempts"] for g in group),
                "mean_generation_energy_kwh": statistics.mean(g["cumulative"]["generation_energy_kwh"] for g in group),
                "mean_execution_energy_kwh": statistics.mean(g["cumulative"]["execution_energy_kwh"] for g in group),
                "mean_cumulative_energy_kwh": statistics.mean(energies),
                "median_cumulative_energy_kwh": statistics.median(energies),
                # stdev needs >= 2 data points; blank (not 0) when there's only one.
                "stddev_cumulative_energy_kwh": statistics.stdev(energies) if len(energies) >= 2 else None,
                "mean_carbon_gco2e": statistics.mean(g["cumulative"]["total_carbon_gco2e"] for g in group),
            })
