"""
Builds system/user prompts from the templates in prompts/ (spec §6).
Uses plain string substitution rather than str.format(), so problem text
containing a literal `{` or `}` can never break template rendering.
"""

import os


def _format_constraints(constraints) -> str:
    """Problems store `constraints` as a list of strings (spec §3 sample) —
    render as one bullet per line."""
    if isinstance(constraints, list):
        return "\n".join(f"- {c}" for c in constraints)
    return str(constraints)


def _render(template: str, mapping: dict) -> str:
    text = template
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))
    return text


class PromptBuilder:
    def __init__(self, prompts_dir: str):
        self._dir = prompts_dir
        self.initial_system = self._load("initial_system.txt")
        self.initial_user_template = self._load("initial_user.txt")
        self.correction_system = self._load("correction_system.txt")
        self.compile_correction_template = self._load("compile_correction_user.txt")
        self.wrong_answer_template = self._load("wrong_answer_user.txt")
        self.timeout_or_runtime_template = self._load("timeout_or_runtime_user.txt")

    def _load(self, filename: str) -> str:
        with open(os.path.join(self._dir, filename), "r", encoding="utf-8") as f:
            return f.read()

    def _problem_fields(self, problem: dict) -> dict:
        return {
            "problem_id": problem["problem_id"],
            "difficulty": problem.get("difficulty"),
            "title": problem["title"],
            "description": problem["description"],
            "input_format": problem["input_format"],
            "output_format": problem["output_format"],
            "constraints": _format_constraints(problem["constraints"]),
        }

    def build_initial_prompt(self, problem: dict) -> str:
        return _render(self.initial_user_template, self._problem_fields(problem))

    def build_compile_correction_prompt(self, problem: dict, previous_code: str, compiler_stderr: str) -> str:
        mapping = {**self._problem_fields(problem),
                   "previous_code": previous_code, "compiler_stderr": compiler_stderr}
        return _render(self.compile_correction_template, mapping)

    def build_wrong_answer_prompt(self, problem: dict, previous_code: str, failing_test: dict) -> str:
        mapping = {
            **self._problem_fields(problem),
            "previous_code": previous_code,
            "test_id": failing_test["test_id"],
            "input": failing_test["input"],
            "expected_output": failing_test["expected_output"],
            "actual_output": failing_test["actual_output"],
        }
        return _render(self.wrong_answer_template, mapping)

    def build_timeout_or_runtime_prompt(self, problem: dict, previous_code: str, failing_test: dict,
                                         failure_type: str, timeout_seconds: float = None,
                                         elapsed_seconds: float = None, exit_code: int = None,
                                         stderr: str = None) -> str:
        if failure_type == "timeout":
            label = "TIME_LIMIT_EXCEEDED"
            detail = (
                f"Time limit: {timeout_seconds} seconds\n"
                f"Observed runtime: {elapsed_seconds} seconds\n\n"
                "The program exceeded the allowed time limit."
            )
        else:  # runtime_error
            label = "RUNTIME_ERROR"
            detail = (
                f"Exit code: {exit_code}\n"
                f"Program stderr / crash information:\n{stderr}\n\n"
                "The program compiled but crashed or exited abnormally during execution."
            )

        mapping = {
            **self._problem_fields(problem),
            "previous_code": previous_code,
            "test_id": failing_test["test_id"],
            "failure_type_label": label,
            "failure_detail": detail,
        }
        return _render(self.timeout_or_runtime_template, mapping)
