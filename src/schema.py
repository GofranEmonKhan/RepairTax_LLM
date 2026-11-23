"""
Builds the JSON records defined in the spec (§7 attempt-level, §8 problem-level,
§9 experiment metadata). Kept as plain dict builders (not dataclasses) so every
record is directly json.dump-able with no extra serialization step.

Rule from the spec, enforced here: a phase that did not run is `None` (-> JSON
null), never `0`. `0` is only ever written when a sensor actually measured zero.

Folded-in feedback (see spec changelog): every attempt now carries
`experiment_id`, `problem_id`, `attempt_id`, and `difficulty` directly, so
downstream analysis files (raw_attempts.jsonl, attempt_level.csv, ...) never
have to be joined back against the problem file just to get these. Problem
results now also break `cumulative` energy/carbon down by phase
(generation/compilation/execution), not just a single combined total.
"""

from typing import Optional


def build_generation_block(
    prompt_tokens: int,
    generated_tokens: int,
    generation_time_seconds: float,
    gpu_kwh: float,
    cpu_kwh: Optional[float],
    cpu_energy_method: str,
    carbon_gco2e: Optional[float],
    feedback_source: Optional[dict] = None,
) -> dict:
    """generation block — always populated, every attempt (spec §7.A)."""
    total_kwh = gpu_kwh + (cpu_kwh or 0.0)
    block = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": prompt_tokens + generated_tokens,
        "generation_time_seconds": generation_time_seconds,
        "energy": {
            "gpu_kwh": gpu_kwh,
            "cpu_kwh": cpu_kwh,
            "cpu_energy_method": cpu_energy_method,
            "total_kwh": total_kwh,
        },
        "carbon_gco2e": carbon_gco2e,
    }
    if feedback_source is not None:
        block["feedback_source"] = feedback_source
    return block


def build_compilation_block(
    success: bool,
    exit_code: Optional[int],
    compiler_stderr: str,
    compilation_time_seconds: float,
    cpu_kwh: Optional[float],
    carbon_gco2e: Optional[float],
) -> dict:
    """compilation block — always attempted, every attempt (spec §7.B)."""
    return {
        "success": success,
        "exit_code": exit_code,
        "compiler_stderr": compiler_stderr,
        "compilation_time_seconds": compilation_time_seconds,
        "energy": {"total_kwh": cpu_kwh},
        "carbon_gco2e": carbon_gco2e,
    }


def build_execution_block(
    tests_total: int,
    tests_passed: int,
    tests_failed: int,
    per_test_results: list,
    execution_time_seconds: float,
    cpu_kwh: Optional[float],
    carbon_gco2e: Optional[float],
) -> dict:
    """execution block — populated only if compilation succeeded (spec §7.B).
    Callers pass None for this whole block when compilation failed, so it
    serializes as JSON null rather than a block full of zeros."""
    return {
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "per_test_results": per_test_results,
        "execution_time_seconds": execution_time_seconds,
        "energy": {"total_kwh": cpu_kwh},
        "carbon_gco2e": carbon_gco2e,
    }


def build_merged_block(generation: dict, compilation: dict, execution: Optional[dict]) -> dict:
    """merged block — only computed for the attempt that compiles and passes
    all tests. This attempt's own cost only, not cumulative (spec §7.C)."""
    energy = generation["energy"]["total_kwh"]
    carbon = generation["carbon_gco2e"] or 0.0
    if compilation["energy"]["total_kwh"] is not None:
        energy += compilation["energy"]["total_kwh"]
        carbon += compilation["carbon_gco2e"] or 0.0
    if execution is not None and execution["energy"]["total_kwh"] is not None:
        energy += execution["energy"]["total_kwh"]
        carbon += execution["carbon_gco2e"] or 0.0
    return {"energy_kwh": energy, "carbon_gco2e": carbon}


def build_attempt_record(
    experiment_id: str,
    problem_id: str,
    attempt_id: str,
    attempt_number: int,
    attempt_type: str,
    difficulty,
    generation: dict,
    compilation: dict,
    execution: Optional[dict],
    failure_type: Optional[str],
    merged: Optional[dict] = None,
) -> dict:
    """Full attempt record (spec §7)."""
    return {
        "experiment_id": experiment_id,
        "problem_id": problem_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "attempt_type": attempt_type,
        "difficulty": difficulty,
        "generation": generation,
        "compilation": compilation,
        "execution": execution,
        "failure_type": failure_type,
        "merged": merged,
    }


def build_problem_result(
    experiment_id: str,
    problem_id: str,
    difficulty,
    model_name: str,
    model_quantization: str,
    attempts: list,
    successful: bool,
    successful_attempt: Optional[int],
    failure_reason: Optional[str] = None,
) -> dict:
    """Problem-level aggregate (spec §8): cumulative totals across ALL
    attempts, broken down by phase (generation/compilation/execution) as
    well as combined, and kept distinct from the successful attempt's own
    `merged` cost."""
    cum_gen_kwh = cum_comp_kwh = cum_exec_kwh = 0.0
    cum_gen_carbon = cum_comp_carbon = cum_exec_carbon = 0.0

    for a in attempts:
        g = a["generation"]
        cum_gen_kwh += g["energy"]["total_kwh"] or 0.0
        cum_gen_carbon += g["carbon_gco2e"] or 0.0

        c = a["compilation"]
        if c["energy"]["total_kwh"] is not None:
            cum_comp_kwh += c["energy"]["total_kwh"]
            cum_comp_carbon += c["carbon_gco2e"] or 0.0

        e = a["execution"]
        if e is not None and e["energy"]["total_kwh"] is not None:
            cum_exec_kwh += e["energy"]["total_kwh"]
            cum_exec_carbon += e["carbon_gco2e"] or 0.0

    total_energy_kwh = cum_gen_kwh + cum_comp_kwh + cum_exec_kwh
    total_carbon_gco2e = cum_gen_carbon + cum_comp_carbon + cum_exec_carbon

    successful_attempt_merged = None
    correction_energy_overhead_pct = None
    if successful and successful_attempt is not None:
        final = attempts[successful_attempt - 1]
        successful_attempt_merged = final["merged"]
        initial_gen_kwh = attempts[0]["generation"]["energy"]["total_kwh"]
        if initial_gen_kwh:
            correction_energy_overhead_pct = (
                (total_energy_kwh - initial_gen_kwh) / initial_gen_kwh
            ) * 100.0

    return {
        "experiment_id": experiment_id,
        "problem_id": problem_id,
        "difficulty": difficulty,
        "model": {"name": model_name, "quantization": model_quantization},
        "result": {
            "successful": successful,
            "successful_attempt": successful_attempt,
            "total_attempts": len(attempts),
            "total_correction_attempts": len(attempts) - 1,
            "failure_reason": failure_reason,
        },
        "cumulative": {
            "generation_energy_kwh": cum_gen_kwh,
            "compilation_energy_kwh": cum_comp_kwh,
            "execution_energy_kwh": cum_exec_kwh,
            "total_energy_kwh": total_energy_kwh,
            "generation_carbon_gco2e": cum_gen_carbon,
            "compilation_carbon_gco2e": cum_comp_carbon,
            "execution_carbon_gco2e": cum_exec_carbon,
            "total_carbon_gco2e": total_carbon_gco2e,
        },
        "successful_attempt_merged": successful_attempt_merged,
        "correction_energy_overhead_pct": correction_energy_overhead_pct,
    }
