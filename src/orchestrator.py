"""
Runs one problem end-to-end (spec §14): initial generation, then up to
max_attempts total attempts (initial + corrections). Every attempt is
compiled and, if compilation succeeds, tested inside the same warm
per-problem sandbox. Every attempt is logged, including failed ones; the
`merged` cost is only computed for the attempt that ultimately passes.

Every measured phase (generation / correction_generation / compilation /
execution) also writes its raw sensor readings to sensor_samples.csv, so
every reported energy figure can be re-derived from raw data.
"""

import time

from src.code_extraction import extract_code
from src.compiler import compile_cpp
from src.test_runner import run_tests, select_failing_test
from src.carbon_tracker import compute_carbon_gco2e
from src.schema import (
    build_generation_block, build_compilation_block, build_execution_block,
    build_merged_block, build_attempt_record, build_problem_result,
)
from src.result_logger import (
    save_attempt_files, save_attempt_to_raw_log, save_problem_result, append_sensor_samples,
)

_ATTEMPT_TYPE_BY_FAILURE = {
    "compile_error": "compile_correction",
    "wrong_answer": "wrong_answer_correction",
    "runtime_error": "runtime_error_correction",
    "timeout": "timeout_correction",
}


def _attempt_id(problem_id: str, attempt_number: int) -> str:
    return f"{problem_id}_A{attempt_number:02d}"


def _log_samples(results_dir, experiment_id, attempt_id, sensor, energy_result):
    """Writes the raw readings returned by a monitor's stop() to
    sensor_samples.csv. Skipped when there are none (estimated CPU mode)."""
    if energy_result["samples"]:
        append_sensor_samples(results_dir, experiment_id, attempt_id,
                               energy_result["phase"], sensor, energy_result["samples"])


def _generate(llm, gpu_monitor, cpu_monitor, intensity, system_prompt, user_prompt,
              feedback_source, results_dir, experiment_id, attempt_id, phase):
    """One measured generation call -> (raw_text, generation_block, full_prompt_text)."""
    gpu_monitor.start(phase)
    cpu_monitor.start(phase)
    start = time.perf_counter()
    result = llm.generate(system_prompt, user_prompt)
    elapsed = time.perf_counter() - start
    gpu_energy = gpu_monitor.stop()
    cpu_energy = cpu_monitor.stop()
    _log_samples(results_dir, experiment_id, attempt_id, "gpu", gpu_energy)
    _log_samples(results_dir, experiment_id, attempt_id, "cpu", cpu_energy)

    total_kwh = gpu_energy["gpu_kwh"] + cpu_energy["cpu_kwh"]
    carbon = compute_carbon_gco2e(total_kwh, intensity)

    generation_block = build_generation_block(
        prompt_tokens=result["prompt_tokens"], generated_tokens=result["generated_tokens"],
        generation_time_seconds=elapsed, gpu_kwh=gpu_energy["gpu_kwh"], cpu_kwh=cpu_energy["cpu_kwh"],
        cpu_energy_method=cpu_energy["method"], carbon_gco2e=carbon, feedback_source=feedback_source,
    )
    return result["text"], generation_block, f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"


def _compile(sandbox, cpu_monitor, intensity, source_code, timeout_seconds,
             results_dir, experiment_id, attempt_id):
    cpu_monitor.start("compilation")
    result = compile_cpp(sandbox, source_code, timeout_seconds)
    cpu_energy = cpu_monitor.stop()
    _log_samples(results_dir, experiment_id, attempt_id, "cpu", cpu_energy)
    carbon = compute_carbon_gco2e(cpu_energy["cpu_kwh"], intensity)
    block = build_compilation_block(
        success=result["success"], exit_code=result["exit_code"], compiler_stderr=result["stderr"],
        compilation_time_seconds=result["compilation_time_seconds"], cpu_kwh=cpu_energy["cpu_kwh"],
        carbon_gco2e=carbon,
    )
    return result, block


def _test(sandbox, cpu_monitor, intensity, test_cases, output_comparator, timeout_seconds,
          results_dir, experiment_id, attempt_id):
    cpu_monitor.start("execution")
    start = time.perf_counter()
    test_result = run_tests(sandbox, test_cases, output_comparator, timeout_seconds)
    elapsed = time.perf_counter() - start
    cpu_energy = cpu_monitor.stop()
    _log_samples(results_dir, experiment_id, attempt_id, "cpu", cpu_energy)
    carbon = compute_carbon_gco2e(cpu_energy["cpu_kwh"], intensity)
    block = build_execution_block(
        tests_total=test_result["tests_total"], tests_passed=test_result["tests_passed"],
        tests_failed=test_result["tests_failed"], per_test_results=test_result["per_test_results"],
        execution_time_seconds=elapsed, cpu_kwh=cpu_energy["cpu_kwh"], carbon_gco2e=carbon,
    )
    return test_result, block


def _save(experiments_dir, results_dir, problem_id, attempt_number, raw_response, code,
          prompt_text, attempt_record):
    save_attempt_files(experiments_dir, problem_id, attempt_number, raw_response, code,
                        prompt_text, attempt_record)
    save_attempt_to_raw_log(results_dir, attempt_record)


def run_problem(problem: dict, llm, prompt_builder, sandbox, gpu_monitor, cpu_monitor,
                 config: dict, experiments_dir: str, results_dir: str, experiment_id: str) -> dict:
    """`experiments_dir` and `results_dir` are already experiment-scoped
    (experiments/<experiment_id>, results/<experiment_id>) — see main.py."""
    problem_id = problem["problem_id"]
    difficulty = problem.get("difficulty")
    intensity = config["carbon"]["intensity_g_per_kwh"]
    max_attempts = config["experiment"]["max_attempts"]
    compile_timeout = config["sandbox"]["compile_timeout_seconds"]
    test_timeout = config["sandbox"]["test_timeout_seconds"]
    output_comparator = problem["output_comparator"]
    model_name = config["model"]["name"]
    model_quant = config["model"]["quantization"]

    attempts_log = []

    def finish(successful: bool, successful_attempt, failure_reason=None) -> dict:
        result = build_problem_result(experiment_id, problem_id, difficulty, model_name, model_quant,
                                       attempts_log, successful, successful_attempt, failure_reason)
        save_problem_result(results_dir, result)
        return result

    # Initial generation: fixed system prompt + problem-specific user prompt, no test cases (spec §6)
    raw_response, generation_block, prompt_text = _generate(
        llm, gpu_monitor, cpu_monitor, intensity,
        prompt_builder.initial_system, prompt_builder.build_initial_prompt(problem),
        feedback_source=None, results_dir=results_dir, experiment_id=experiment_id,
        attempt_id=_attempt_id(problem_id, 1), phase="generation",
    )
    current_feedback_source = None  # what prompted the code being evaluated this iteration

    for attempt_number in range(1, max_attempts + 1):
        attempt_id = _attempt_id(problem_id, attempt_number)
        code = extract_code(raw_response)
        attempt_type = ("initial" if current_feedback_source is None
                         else _ATTEMPT_TYPE_BY_FAILURE[current_feedback_source["failure_type"]])

        compile_result, compilation_block = _compile(
            sandbox, cpu_monitor, intensity, code, compile_timeout,
            results_dir, experiment_id, attempt_id)

        if not compile_result["success"]:
            attempt_record = build_attempt_record(
                experiment_id, problem_id, attempt_id, attempt_number, attempt_type, difficulty,
                generation_block, compilation_block, execution=None, failure_type="compile_error",
            )
            attempts_log.append(attempt_record)
            _save(experiments_dir, results_dir, problem_id, attempt_number, raw_response, code,
                  prompt_text, attempt_record)

            if attempt_number == max_attempts:
                return finish(False, None, "max_attempts_exceeded")

            next_user_prompt = prompt_builder.build_compile_correction_prompt(
                problem, previous_code=code, compiler_stderr=compile_result["stderr"])
            current_feedback_source = {"from_attempt": attempt_number,
                                        "failure_type": "compile_error", "test_id": None}

        else:
            test_result, execution_block = _test(
                sandbox, cpu_monitor, intensity, problem["test_cases"], output_comparator,
                test_timeout, results_dir, experiment_id, attempt_id)
            attempt_record = build_attempt_record(
                experiment_id, problem_id, attempt_id, attempt_number, attempt_type, difficulty,
                generation_block, compilation_block, execution=execution_block,
                failure_type=test_result["failure_type"],
            )

            if test_result["all_passed"]:
                attempt_record["merged"] = build_merged_block(
                    generation_block, compilation_block, execution_block)
                attempts_log.append(attempt_record)
                _save(experiments_dir, results_dir, problem_id, attempt_number, raw_response, code,
                      prompt_text, attempt_record)
                return finish(True, attempt_number)

            attempts_log.append(attempt_record)
            _save(experiments_dir, results_dir, problem_id, attempt_number, raw_response, code,
                  prompt_text, attempt_record)

            if attempt_number == max_attempts:
                return finish(False, None, "max_attempts_exceeded")

            failing_test = select_failing_test(test_result)
            if test_result["failure_type"] == "wrong_answer":
                next_user_prompt = prompt_builder.build_wrong_answer_prompt(problem, code, failing_test)
            else:
                next_user_prompt = prompt_builder.build_timeout_or_runtime_prompt(
                    problem, code, failing_test, test_result["failure_type"],
                    timeout_seconds=test_timeout, elapsed_seconds=failing_test.get("runtime_seconds"),
                    exit_code=failing_test.get("exit_code"), stderr=failing_test.get("stderr"),
                )
            current_feedback_source = {"from_attempt": attempt_number,
                                        "failure_type": test_result["failure_type"],
                                        "test_id": failing_test["test_id"]}

        # Stateless correction request: full problem + full previous code + latest feedback only (spec §6)
        raw_response, generation_block, prompt_text = _generate(
            llm, gpu_monitor, cpu_monitor, intensity,
            prompt_builder.correction_system, next_user_prompt, feedback_source=current_feedback_source,
            results_dir=results_dir, experiment_id=experiment_id,
            attempt_id=_attempt_id(problem_id, attempt_number + 1), phase="correction_generation",
        )

    # Unreachable in practice: every branch above returns before falling
    # through past the last allowed attempt. Kept as a defensive fallback.
    return finish(False, None, "max_attempts_exceeded")
