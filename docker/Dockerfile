# Minimal sandbox for compiling/running generated C++17 solutions.
# gcc:13 (Debian-based) ships g++ 13 and GNU coreutils (needed for the
# `timeout` command used to enforce per-test execution limits).
FROM gcc:13

WORKDIR /sandbox

# Kept alive so the orchestrator can `docker exec` into the same warm
# container across every attempt of one problem (spec §12).
CMD ["sleep", "infinity"]
