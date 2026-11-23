"""
Compiles generated C++17 code inside the warm sandbox (spec §7.B, §12).
Always attempted for every generated attempt.
"""


def compile_cpp(sandbox, source_code: str, timeout_seconds: float = 15.0) -> dict:
    """Returns {"success", "exit_code", "stderr", "compilation_time_seconds"}."""
    sandbox.write_file("solution.cpp", source_code)
    result = sandbox.run("g++ -std=c++17 -O2 -o solution solution.cpp", timeout_seconds)
    return {
        "success": result["exit_code"] == 0,
        "exit_code": result["exit_code"],
        "stderr": result["stderr"].strip(),
        "compilation_time_seconds": result["elapsed_seconds"],
    }
