#!/usr/bin/env python
"""Flower SuperLink-backed FedMed plugin."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, List

from chris_plugin import chris_plugin

__version__ = "0.2.1"

APP_PACKAGE = "fedmed_flower_app"
APP_DIR = Path(resources.files(APP_PACKAGE))
SERVER_SUMMARY_TOKEN = "[fedmed-superlink-app] SUMMARY "
DEFAULT_HOST = "0.0.0.0"
DEFAULT_FLEET_PORT = 9092
DEFAULT_CONTROL_PORT = 9093
DEFAULT_SERVERAPP_PORT = 9091
DEFAULT_STATE_DIR = Path("/tmp/fedmed-flwr")
DEFAULT_SUMMARY = "server_summary.json"
DEFAULT_WEIGHTS = "server_final.ckpt"
DEFAULT_FEDERATION = "fedmed-local"
IMAGE_TAG = f"docker.io/fedmed/pl-superlink:{__version__}"
REPO_URL = "https://github.com/EC528-Fall-2025/FedMed-ChRIS"

Process = subprocess.Popen

CHILDREN: list[Process] = []


@dataclass(frozen=True)
class FlowerRunConfig:
    """Subset of plugin args that map directly to Flower run overrides."""

    rounds: int
    total_clients: int
    local_epochs: int
    learning_rate: float
    data_seed: int
    fraction_evaluate: float

    def as_override_string(self) -> str:
        return (
            f"num-server-rounds={self.rounds} "
            f"total-clients={self.total_clients} "
            f"local-epochs={self.local_epochs} "
            f"learning-rate={self.learning_rate} "
            f"data-seed={self.data_seed} "
            f"fraction-evaluate={self.fraction_evaluate}"
        )


@dataclass(frozen=True)
class SuperLinkAddresses:
    """Collection of network addresses derived from CLI options."""

    fleet: str
    control: str
    serverapp: str


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the FedMed Flower SuperLink inside ChRIS.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address for Flower services")
    parser.add_argument("--fleet-port", type=int, default=DEFAULT_FLEET_PORT, help="Fleet API port")
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT, help="Control API port")
    parser.add_argument("--serverapp-port", type=int, default=DEFAULT_SERVERAPP_PORT, help="ServerAppIo port")
    parser.add_argument("--rounds", type=int, default=1, help="federated rounds to run")
    parser.add_argument("--total-clients", type=int, default=1, help="expected number of SuperNodes")
    parser.add_argument("--local-epochs", type=int, default=10, help="local epochs per round")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="client learning rate")
    parser.add_argument("--data-seed", type=int, default=13, help="seed for synthetic data generation")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="fraction of clients used for evaluation")
    parser.add_argument("--federation-name", default=DEFAULT_FEDERATION, help="Flower federation name from the pyproject")
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY, help="filename to store the training summary")
    parser.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR), help="directory used as FLWR_HOME")
    parser.add_argument("--keep-state", action="store_true", help="keep the Flower state directory after finishing")
    parser.add_argument("--startup-delay", type=float, default=3.0, help="seconds to wait for SuperLink to boot")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"fedmed-pl-superlink {__version__}",
    )
    return parser


parser = build_parser()


def _discover_ipv4_addresses() -> list[str]:
    """Attempt to list IPv4 addresses that clients could need to target."""
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
        ips = sorted({token for token in output.split() if token})
        return ips
    except Exception:
        return []


def _stream_lines(
    stream,  # type: ignore[override]
    prefix: str,
    hook: Callable[[str], None] | None = None,
) -> None:
    """Continuously read a subprocess pipe, echoing lines and invoking hook if set."""
    if stream is None:
        return
    for raw in iter(stream.readline, ""):
        line = raw.rstrip()
        print(f"[{prefix}] {line}", flush=True)
        if hook:
            hook(line)
    stream.close()


def _register_child(proc: Process) -> None:
    """Track a child process so it can be torn down during cleanup."""
    CHILDREN.append(proc)


def _terminate_process(proc: Process, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _cleanup_children() -> None:
    for proc in reversed(CHILDREN):
        try:
            _terminate_process(proc)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[fedmed-pl-superlink] failed to terminate {proc.args}: {exc}", flush=True)
    CHILDREN.clear()


def _stage_flower_app(state_dir: Path) -> tuple[Path, Callable[[], None]]:
    """Copy the Flower App into a temporary project directory for `flwr run`."""
    staging_root = Path(
        tempfile.mkdtemp(prefix="fedmed-flower-app-", dir=str(state_dir))
    )
    project_dir = staging_root / "project"
    package_dir = project_dir / APP_PACKAGE
    shutil.copytree(APP_DIR, package_dir)
    pyproject_src = APP_DIR / "pyproject.toml"
    if pyproject_src.exists():
        shutil.copy(pyproject_src, project_dir / "pyproject.toml")

    def _cleanup() -> None:
        shutil.rmtree(staging_root, ignore_errors=True)

    return project_dir, _cleanup


def _build_run_config(options: Namespace) -> FlowerRunConfig:
    """Extract the Flower run override settings from plugin options."""
    return FlowerRunConfig(
        rounds=options.rounds,
        total_clients=options.total_clients,
        local_epochs=options.local_epochs,
        learning_rate=options.learning_rate,
        data_seed=options.data_seed,
        fraction_evaluate=options.fraction_evaluate,
    )


def _build_addresses(options: Namespace) -> SuperLinkAddresses:
    """Produce the Fleet/Control/ServerApp bind addresses."""
    return SuperLinkAddresses(
        fleet=f"{options.host}:{options.fleet_port}",
        control=f"{options.host}:{options.control_port}",
        serverapp=f"{options.host}:{options.serverapp_port}",
    )


def _prepare_environment(state_dir: str) -> tuple[dict[str, str], Path]:
    """Create the FLWR_HOME directory and return (env, home_path)."""
    env = os.environ.copy()
    flwr_home = Path(state_dir).expanduser()
    flwr_home.mkdir(parents=True, exist_ok=True)
    env["FLWR_HOME"] = str(flwr_home)
    return env, flwr_home


def handle_signals() -> None:
    def _handle(signum, _frame):  # type: ignore[override]
        print(f"\n[fedmed-pl-superlink] received signal {signum}, cleaning up...", flush=True)
        _cleanup_children()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _launch_superlink(addresses: SuperLinkAddresses, env: dict[str, str]) -> Process:
    """Start the long-lived Flower SuperLink services inside this container."""
    cmd: List[str] = [
        "flower-superlink",
        "--insecure",
        f"--fleet-api-address={addresses.fleet}",
        f"--control-api-address={addresses.control}",
        f"--serverappio-api-address={addresses.serverapp}",
    ]
    print(f"[fedmed-pl-superlink] starting SuperLink: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    _register_child(proc)
    prefix = "superlink"
    threading.Thread(target=_stream_lines, args=(proc.stdout, prefix), daemon=True).start()
    threading.Thread(target=_stream_lines, args=(proc.stderr, prefix), daemon=True).start()
    return proc


def _run_federation(
    options: Namespace,
    addresses: SuperLinkAddresses,
    run_cfg: FlowerRunConfig,
    env: dict[str, str],
) -> dict[str, Any]:
    """Bundle the Flower App, run `flwr run`, and return the parsed summary."""
    staged_app_dir, cleanup_app = _stage_flower_app(Path(env["FLWR_HOME"]))
    run_config = run_cfg.as_override_string()
    fed_config = f"address='{addresses.control}' insecure=true"
    cmd: List[str] = [
        "flwr",
        "run",
        str(staged_app_dir),
        options.federation_name,
        "--stream",
        "--run-config",
        run_config,
        "--federation-config",
        fed_config,
    ]

    summary: dict[str, Any] | None = None

    def _capture(line: str) -> None:
        nonlocal summary
        if SERVER_SUMMARY_TOKEN in line:
            payload = line.split(SERVER_SUMMARY_TOKEN, maxsplit=1)[1].strip()
            try:
                summary = json.loads(payload)
            except json.JSONDecodeError as exc:
                print(f"[fedmed-pl-superlink] failed to parse summary: {exc}", flush=True)

    print(f"[fedmed-pl-superlink] launching Flower run: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    _register_child(proc)
    stdout_thread = threading.Thread(
        target=_stream_lines, args=(proc.stdout, "flower-run", _capture), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_stream_lines, args=(proc.stderr, "flower-run"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    exit_code = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    cleanup_app()
    if exit_code != 0:
        raise RuntimeError(f"flwr run exited with {exit_code}")
    if summary is None:
        summary = {
            "run_id": None,
            "rounds": options.rounds,
            "total_clients": options.total_clients,
            "message": "Training completed but summary was not emitted.",
        }
    return summary


@chris_plugin(
    parser=parser,
    title="FedMed Flower SuperLink",
    category="Federated Learning",
    min_memory_limit="500Mi",
    min_cpu_limit="1000m",
)
def _plugin_main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    del inputdir
    handle_signals()

    if getattr(options, "json", False):
        emit_plugin_json()
        return

    if options.total_clients <= 0:
        raise ValueError("total-clients must be >= 1")
    if not APP_DIR.exists():
        raise FileNotFoundError(f"Flower app missing at {APP_DIR}")

    addresses = _build_addresses(options)
    run_cfg = _build_run_config(options)
    env, flwr_home = _prepare_environment(options.state_dir)
    env["FEDMED_OUTPUT_CKPT"] = str((outputdir / DEFAULT_WEIGHTS).resolve())

    fleet_addr = addresses.fleet
    print(
        f"[fedmed-pl-superlink] SuperNodes should target Fleet API at {fleet_addr}",
        flush=True,
    )
    reachable_ips = _discover_ipv4_addresses()
    if reachable_ips:
        print(
            "[fedmed-pl-superlink] reachable IPv4 addresses: "
            + ", ".join(reachable_ips),
            flush=True,
        )
    else:
        print(
            "[fedmed-pl-superlink] unable to auto-detect host IPs; clients must be pointed at the compute-node address manually.",
            flush=True,
        )

    superlink = _launch_superlink(addresses, env)
    time.sleep(max(0, options.startup_delay))
    try:
        summary = _run_federation(options, addresses, run_cfg, env)
    finally:
        _terminate_process(superlink)
        _cleanup_children()

    summary_path = outputdir / options.summary_file
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[fedmed-pl-superlink] wrote summary to {summary_path}", flush=True)

    if not options.keep_state:
        shutil.rmtree(flwr_home, ignore_errors=True)
        print(f"[fedmed-pl-superlink] cleaned {flwr_home}", flush=True)


def main(*args, **kwargs):
    if not args and not kwargs and "--json" in sys.argv:
        emit_plugin_json()
        return
    return _plugin_main(*args, **kwargs)


def emit_plugin_json() -> None:
    from chris_plugin.tool import chris_plugin_info

    argv = [
        "chris_plugin_info",
        "--dock-image",
        IMAGE_TAG,
        "--name",
        "pl-superlink",
        "--public-repo",
        REPO_URL,
    ]
    prev = sys.argv
    try:
        sys.argv = argv
        chris_plugin_info.main()
    finally:
        sys.argv = prev


if __name__ == "__main__":
    main()
