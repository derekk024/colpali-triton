from __future__ import annotations

from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = PROJECT_ROOT / "infra" / "aws" / "run_phase11_job.sh"

EXPECTED_TIMEOUTS = {
    "container_build": ("BUILD_TIMEOUT_SECONDS", 1200),
    "hardware_gate": ("HARDWARE_TIMEOUT_SECONDS", 120),
    "host_metadata": ("HOST_METADATA_TIMEOUT_SECONDS", 120),
    "cuda_tests": ("CUDA_TESTS_TIMEOUT_SECONDS", 900),
    "cuda_assertion": ("CUDA_ASSERTION_TIMEOUT_SECONDS", 120),
    "tuning_preflight": ("TUNING_PREFLIGHT_TIMEOUT_SECONDS", 120),
    "tuning": ("TUNING_TIMEOUT_SECONDS", 3600),
    "derive_winner": ("DERIVE_TIMEOUT_SECONDS", 120),
    "benchmark": ("BENCHMARK_TIMEOUT_SECONDS", 1800),
    "result_contract": ("RESULT_CONTRACT_TIMEOUT_SECONDS", 120),
    "profiling_ncu": ("PROFILER_TIMEOUT_SECONDS", 300),
    "profiling_nsys": ("PROFILER_TIMEOUT_SECONDS", 300),
    "profiling_status": ("PROFILE_STATUS_TIMEOUT_SECONDS", 120),
}


def _heredoc(script: str, output_name: str) -> str:
    marker = (
        f'python3 - "${{RESULT_DIRECTORY}}/{output_name}" <<\'PY\'\n'
    )
    _, found, remainder = script.partition(marker)
    assert found
    body, found, _ = remainder.partition("\nPY\n")
    assert found
    return body


def test_phase11_job_shell_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(JOB_PATH)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_timeout_defaults_are_recorded_in_both_json_contracts() -> None:
    script = JOB_PATH.read_text()
    commands = _heredoc(script, "commands.json")
    status = _heredoc(script, "step_status.json")

    for key, (environment_name, seconds) in EXPECTED_TIMEOUTS.items():
        assert f"{environment_name}={seconds}" in script
        pattern = re.compile(
            rf'"{key}": int\(\s*'
            rf'os\.environ\["{environment_name}"\]\s*\)'
        )
        assert pattern.search(commands)
        assert pattern.search(status)

    assert '"timeouts_seconds": timeouts' in commands
    assert '"timeouts_seconds": timeouts' in status
    assert '"profiling_attempts": {' in status


def test_every_paid_step_uses_a_bounded_runner() -> None:
    script = JOB_PATH.read_text()

    assert re.search(
        r"run_logged_timed\s+\\\s+"
        r'"\$\{BUILD_TIMEOUT_SECONDS\}"',
        script,
    )
    for step, timeout_name in (
        ("hardware_gate", "HARDWARE_TIMEOUT_SECONDS"),
        ("cuda_tests", "CUDA_TESTS_TIMEOUT_SECONDS"),
        ("tuning", "TUNING_TIMEOUT_SECONDS"),
        ("benchmark", "BENCHMARK_TIMEOUT_SECONDS"),
    ):
        assert re.search(
            rf'run_docker_logged_timed\s+\\\s+"{step}"\s+\\\s+'
            rf'"\$\{{{timeout_name}\}}"',
            script,
        )
    for step, timeout_name in (
        ("tuning_preflight", "TUNING_PREFLIGHT_TIMEOUT_SECONDS"),
        ("derive_winner", "DERIVE_TIMEOUT_SECONDS"),
    ):
        assert re.search(
            rf'run_docker_capture_timed\s+\\\s+"{step}"\s+\\\s+'
            rf'"\$\{{{timeout_name}\}}"',
            script,
        )
    for timeout_name in (
        "HOST_METADATA_TIMEOUT_SECONDS",
        "CUDA_ASSERTION_TIMEOUT_SECONDS",
        "RESULT_CONTRACT_TIMEOUT_SECONDS",
        "PROFILE_STATUS_TIMEOUT_SECONDS",
    ):
        assert re.search(
            rf'run_capture_timed\s+\\\s+"\$\{{{timeout_name}\}}"',
            script,
        )

    assert script.count(
        "timeout --signal=TERM --kill-after=30s 300s"
    ) == 2


def test_fail_closed_gates_prevent_downstream_paid_work() -> None:
    script = JOB_PATH.read_text()

    assert (
        "if (( CUDA_TESTS_EXIT == 0 && CUDA_ASSERTION_EXIT == 0 )); then"
        in script
    )
    assert "assert_phase11_tuning_preflight.py" in script
    assert (
        "if (( TUNING_PREFLIGHT_COMMAND_EXIT == 0 )); then"
        in script
    )
    assert (
        "&& TUNING_PREFLIGHT_EXIT == 0 )); then\n"
        "    run_docker_logged_timed \\\n"
        '        "tuning"'
    ) in script
    assert (
        "if (( TUNING_EXIT == 0 )); then\n"
        "    run_docker_capture_timed \\\n"
        '        "derive_winner"'
    ) in script

    benchmark_guard_start = script.index(
        "if (( COMMAND_CONTRACT_EXIT == 0 \\\n",
        script.index("DERIVE_WINNER_EXIT=$?"),
    )
    benchmark_guard_end = script.index(
        ")); then",
        benchmark_guard_start,
    )
    benchmark_guard = script[benchmark_guard_start:benchmark_guard_end]
    for gate in (
        "COMMAND_CONTRACT_EXIT",
        "CONTAINER_BUILD_EXIT",
        "HARDWARE_GATE_EXIT",
        "HOST_METADATA_EXIT",
        "CUDA_TESTS_EXIT",
        "CUDA_ASSERTION_EXIT",
        "TUNING_PREFLIGHT_EXIT",
        "TUNING_EXIT",
        "DERIVE_WINNER_EXIT",
    ):
        assert f"{gate} == 0" in benchmark_guard

    contract_exit = script.index("RESULT_CONTRACT_EXIT=$?")
    profiling_start = script.index("NCU_ATTEMPTED=0")
    assert contract_exit < profiling_start
    assert "if (( RESULT_CONTRACT_EXIT == 0 )); then" in script
    assert (
        "PROFILE_SUCCEEDED == 0 && RESULT_CONTRACT_EXIT == 0"
        in script
    )


def test_timed_out_containers_are_force_removed_by_cid() -> None:
    script = JOB_PATH.read_text()

    assert '--cidfile "${CID_FILE}"' in script
    assert '--cidfile "${NCU_CID_FILE}"' in script
    assert '--cidfile "${NSYS_CID_FILE}"' in script
    assert 'docker rm --force "${CONTAINER_ID}"' in script
    assert (
        "EXIT_CODE == 124 || EXIT_CODE == 137 || EXIT_CODE == 143"
        in script
    )
    assert re.search(
        r"cleanup_timed_container\s+\\\s+"
        r'"profiling_ncu" "\$\{NCU_EXIT\}"',
        script,
    )
    assert re.search(
        r"cleanup_timed_container\s+\\\s+"
        r'"profiling_nsys" "\$\{NSYS_EXIT\}"',
        script,
    )
    assert 'rmdir "${CID_DIRECTORY}"' in script
