"""
Runs a compiled solution against every test case (spec §3, §7.B) using the
problem's declared output_comparator. Local evaluation always runs ALL test
cases; only ONE failing test is later chosen (select_failing_test) to show
the model as correction feedback (spec §6, §18).

Every per_test_results entry now carries `runtime_seconds` (previously only
timeouts did), sourced from the sandbox's per-run wall clock.

Note: per-test `result` values are the same vocabulary as the attempt-level
`failure_type` ("pass" / "wrong_answer" / "runtime_error" / "timeout"), which
is slightly more granular than the illustrative "pass"/"fail" shown in the
spec's worked example — kept this way so select_failing_test() below can
filter directly on the matching failure type.
"""


def _compare_whitespace_normalized(actual: str, expected: str) -> bool:
    return " ".join(actual.split()) == " ".join(expected.split())


def _compare_exact(actual: str, expected: str) -> bool:
    return actual == expected


COMPARATORS = {
    "whitespace_normalized": _compare_whitespace_normalized,
    "exact": _compare_exact,
}


def run_tests(sandbox, test_cases: list, output_comparator: str, timeout_seconds: float) -> dict:
    """Returns {"all_passed", "tests_total", "tests_passed", "tests_failed",
    "failure_type", "per_test_results"}."""
    compare = COMPARATORS[output_comparator]
    per_test_results = []
    passed = 0

    for test in test_cases:
        input_filename = f"input_{test['test_id']}.txt"
        sandbox.write_file(input_filename, test["input"])
        result = sandbox.run(f"./solution < {input_filename}", timeout_seconds)

        if result["timed_out"]:
            per_test_results.append({
                "test_id": test["test_id"], "result": "timeout",
                "input": test["input"], "expected_output": test["expected_output"],
                "actual_output": None, "runtime_seconds": result["elapsed_seconds"],
            })
            continue

        if result["exit_code"] != 0:
            per_test_results.append({
                "test_id": test["test_id"], "result": "runtime_error",
                "input": test["input"], "expected_output": test["expected_output"],
                "actual_output": result["stdout"], "exit_code": result["exit_code"],
                "stderr": result["stderr"], "runtime_seconds": result["elapsed_seconds"],
            })
            continue

        is_pass = compare(result["stdout"], test["expected_output"])
        per_test_results.append({
            "test_id": test["test_id"],
            "result": "pass" if is_pass else "wrong_answer",
            "input": test["input"], "expected_output": test["expected_output"],
            "actual_output": result["stdout"], "runtime_seconds": result["elapsed_seconds"],
        })
        if is_pass:
            passed += 1

    tests_total = len(test_cases)
    tests_failed = tests_total - passed
    all_passed = passed == tests_total

    failure_type = None
    if not all_passed:
        # Priority when several tests fail with DIFFERENT causes in one
        # attempt — not specified beyond "single failing test" in the doc.
        # This ordering (timeout > runtime_error > wrong_answer) is this
        # implementation's choice; change it here if you'd prefer another.
        outcomes = {r["result"] for r in per_test_results}
        if "timeout" in outcomes:
            failure_type = "timeout"
        elif "runtime_error" in outcomes:
            failure_type = "runtime_error"
        else:
            failure_type = "wrong_answer"

    return {
        "all_passed": all_passed,
        "tests_total": tests_total,
        "tests_passed": passed,
        "tests_failed": tests_failed,
        "failure_type": failure_type,
        "per_test_results": per_test_results,
    }


def select_failing_test(test_result: dict) -> dict:
    """Default rule (spec §18, unconfirmed): lowest test_id among tests
    matching the attempt's chosen failure_type."""
    matching = [r for r in test_result["per_test_results"] if r["result"] == test_result["failure_type"]]
    return min(matching, key=lambda r: r["test_id"])
