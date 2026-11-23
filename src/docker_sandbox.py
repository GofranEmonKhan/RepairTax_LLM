"""
Docker-based execution sandbox (spec §12). One container is created per
problem and reused across all of that problem's attempts and test runs
("warm, per-problem" reuse) — a fresh container is created/destroyed for
each new problem, not for each attempt or test case.

Isolation: no network, capped CPU/memory, generated code never touches the
host filesystem or a host shell directly.
"""

import io
import tarfile
import time

import docker


class Sandbox:
    def __init__(self, docker_client: docker.DockerClient, image: str,
                 memory_limit: str = "512m", nano_cpus: int = 1_000_000_000):
        self._client = docker_client
        self._image = image
        self._memory_limit = memory_limit
        self._nano_cpus = nano_cpus
        self._container = None

    def __enter__(self):
        self._container = self._client.containers.run(
            self._image,
            detach=True,
            network_disabled=True,
            mem_limit=self._memory_limit,
            nano_cpus=self._nano_cpus,
            working_dir="/sandbox",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._container is not None:
            self._container.remove(force=True)

    def write_file(self, filename: str, content: str):
        """Writes `content` to /sandbox/<filename> via a tar stream — avoids
        the shell-escaping problems a heredoc would have with arbitrary
        generated code (backticks, quotes, `$`, etc.)."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_buffer.seek(0)
        self._container.put_archive("/sandbox", tar_buffer)

    def run(self, shell_command: str, timeout_seconds: float) -> dict:
        """Runs `shell_command` inside the warm container, killed after
        timeout_seconds by the `timeout` coreutil rather than Python-side
        process management, which is more reliable for killing processes
        inside a container.

        Returns {"exit_code", "stdout", "stderr", "elapsed_seconds", "timed_out"}.
        """
        wrapped = f"timeout -s KILL {timeout_seconds}s {shell_command}"
        start = time.perf_counter()
        result = self._container.exec_run(cmd=["sh", "-c", wrapped], demux=True, workdir="/sandbox")
        elapsed = time.perf_counter() - start
        stdout, stderr = result.output
        exit_code = result.exit_code
        # `timeout -s KILL` returns 137 (128 + SIGKILL) when it had to kill the process.
        timed_out = exit_code == 137
        return {
            "exit_code": exit_code,
            "stdout": (stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (stderr or b"").decode("utf-8", errors="replace"),
            "elapsed_seconds": elapsed,
            "timed_out": timed_out,
        }


def ensure_image_built(docker_client: docker.DockerClient, image: str, dockerfile_dir: str):
    """Builds the sandbox image if it isn't already present."""
    try:
        docker_client.images.get(image)
    except docker.errors.ImageNotFound:
        print(f"[docker_sandbox] Building sandbox image '{image}' from {dockerfile_dir} ...")
        docker_client.images.build(path=dockerfile_dir, tag=image, rm=True)
