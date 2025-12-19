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
from importlib import resources
from pathlib import Path
from typing import Any, Callable, List

from chris_plugin import chris_plugin

__version__ = "0.1.7"

APP_PACKAGE = "fedmed_flower_app"
APP_DIR = Path(resources.files(APP_PACKAGE))
SERVER_SUMMARY_TOKEN = "[fedmed-superlink-app] SUMMARY "
DEFAULT_HOST = "0.0.0.0"
DEFAULT_FLEET_PORT = 9092
DEFAULT_CONTROL_PORT = 9093
DEFAULT_SERVERAPP_PORT = 9091
DEFAULT_ROUNDS = 3
DEFAULT_TOTAL_CLIENTS = 3
DEFAULT_LOCAL_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.2
DEFAULT_DATA_SEED = 13
DEFAULT_FRACTION_EVAL = 1.0
DEFAULT_STARTUP_DELAY = 3.0
DEFAULT_STATE_DIR = Path("/tmp/fedmed-flwr")
DEFAULT_SUMMARY = "server_summary.json"
DEFAULT_WEIGHTS = "server_final.ckpt"
DEFAULT_FEDERATION = "fedmed-local"
IMAGE_TAG = f"docker.io/fedmed/pl-superlink:{__version__}"
REPO_URL = "https://github.com/EC528-Fall-2025/FedMed-ChRIS"

Process = subprocess.Popen

CHILDREN: list[Process] = []


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the FedMed Flower SuperLink inside ChRIS.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(
        host=DEFAULT_HOST,
        fleet_port=DEFAULT_FLEET_PORT,
        control_port=DEFAULT_CONTROL_PORT,
        serverapp_port=DEFAULT_SERVERAPP_PORT,
        federation_name=DEFAULT_FEDERATION,
        summary_file=DEFAULT_SUMMARY,
        state_dir=str(DEFAULT_STATE_DIR),
        startup_delay=DEFAULT_STARTUP_DELAY,
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="federated rounds to run")
    parser.add_argument(
        "--total-clients", type=int, default=DEFAULT_TOTAL_CLIENTS, help="expected number of SuperNodes"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS, help="local epochs per round"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="client learning rate"
    )
    parser.add_argument(
        "--data-seed", type=int, default=DEFAULT_DATA_SEED, help="seed for synthetic data generation"
    )
    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=DEFAULT_FRACTION_EVAL,
        help="fraction of clients used for evaluation",
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    # NEW optional bastion-related arguments (for reverse tunnelling)
    parser.add_argument("--bastion-host", default=None, help="SSH bastion hostname for reverse tunneling")
    parser.add_argument("--bastion-user", default=None, help="SSH user on bastion")
    parser.add_argument("--bastion-port", type=int, default=22, help="SSH port on bastion")

    parser.add_argument(
        "--bastion-key",
        type=str,
        default="id_ed25519",
        help="path to SSH private key inside container",
    )
    parser.add_argument(
        "--bastion-known-hosts",
        type=str,
        default="known_hosts",
        help="path to known_hosts file; if missing, host key checking is relaxed",
    )

    parser.add_argument(
        "--bastion-fleet-port",
        type=int,
        default=19092,
        help="remote port on bastion forwarding to Fleet API",
    )
    parser.add_argument(
        "--bastion-control-port",
        type=int,
        default=19093,
        help="remote port on bastion forwarding to Control API",
    )
    parser.add_argument(
        "--bastion-serverapp-port",
        type=int,
        default=19091,
        help="remote port on bastion forwarding to ServerAppIo",
    )

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


def _prepare_environment(state_dir: str) -> tuple[dict[str, str], Path]:
    """Create the FLWR_HOME directory and return (env, home_path)."""
    env = os.environ.copy()
    flwr_home = Path(state_dir).expanduser()
    flwr_home.mkdir(parents=True, exist_ok=True)
    print(f"[fedmed-pl-superlink] prepared FLWR_HOME at {flwr_home}", flush=True)
    env["FLWR_HOME"] = str(flwr_home)
    return env, flwr_home

def _resolve_input_file(inputdir: Path, raw_path: str) -> Path | None:
    """
    Resolve a file that is supposed to live under the plugin's input directory.

    Cases:
      - relative path: look in inputdir / raw_path, else search by basename
      - absolute path starting with /incoming: treat it as a hint and search
        under inputdir for the same basename
      - any other absolute path: only use it if it actually exists
    """
    base = inputdir
    p = Path(raw_path)

    # Case 1: relative path like "id_ed25519" or "keys/id_ed25519"
    if not p.is_absolute():
        candidate = base / p
        if candidate.is_file():
            print(f"[fedmed-pl-superlink] using bastion file: {candidate}", flush=True)
            return candidate

        # try by basename anywhere under inputdir
        basename = p.name
        matches = list(base.rglob(basename))
        if matches:
            chosen = matches[0]
            print(
                f"[fedmed-pl-superlink] resolved {raw_path} -> {chosen} (found under {base})",
                flush=True,
            )
            return chosen

        print(
            f"[fedmed-pl-superlink] ERROR: could not find {raw_path} under {base}",
            flush=True,
        )
        return None

    # Case 2: absolute path under /incoming – treat as hint
    if str(p).startswith("/incoming/"):
        basename = p.name
        if base.exists():
            matches = list(base.rglob(basename))
            if matches:
                chosen = matches[0]
                print(
                    f"[fedmed-pl-superlink] resolved {raw_path} -> {chosen} (found under {base})",
                    flush=True,
                )
                return chosen
        print(
            f"[fedmed-pl-superlink] ERROR: {raw_path} not usable and nothing named {basename} under {base}",
            flush=True,
        )
        return None

    # Case 3: other absolute path – only accept if it actually exists
    if p.is_file():
        print(f"[fedmed-pl-superlink] using bastion file (absolute): {p}", flush=True)
        return p

    print(
        f"[fedmed-pl-superlink] ERROR: absolute path {raw_path} does not exist",
        flush=True,
    )
    return None

# Added for reverse tunelling
def _maybe_open_reverse_tunnels(
    options: Namespace,
    inputdir: Path,
    fleet_local: str,
    control_local: str,
    serverapp_local: str,
) -> Process | None:
    """Optionally open reverse SSH tunnels from this container to a bastion.

    Exposes bastion:<bastion_*_port> -> container:<local ports>.
    If bastion_host or bastion_user is unset, this is a no-op.
    """
    bastion_host = options.bastion_host
    bastion_user = options.bastion_user
    if not bastion_host or not bastion_user:
        return None

    # Resolve original key under /share/incoming (read-only bind mount)
    orig_key_path = _resolve_input_file(inputdir, options.bastion_key)
    if orig_key_path is None:
        print("[fedmed-pl-superlink] WARNING: no valid SSH key found; skipping reverse tunnels", flush=True)
        return None

    # 🔑 Copy key to a writable location and fix permissions there
    keys_dir = Path("/tmp/fedmed-ssh")
    keys_dir.mkdir(parents=True, exist_ok=True)

    key_path = keys_dir / orig_key_path.name
    try:
        shutil.copy2(orig_key_path, key_path)
        key_path.chmod(0o600)
        print(
            f"[fedmed-pl-superlink] copied bastion key to {key_path} and set permissions 0600",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[fedmed-pl-superlink] ERROR: failed to copy/chmod bastion key: {exc}",
            flush=True,
        )
        return None

    # Optional: handle known_hosts similarly
    known_hosts_path = None
    if options.bastion_known_hosts:
        orig_known = _resolve_input_file(inputdir, options.bastion_known_hosts)
        if orig_known and orig_known.is_file():
            try:
                known_hosts_path = keys_dir / "known_hosts"
                shutil.copy2(orig_known, known_hosts_path)
                print(
                    f"[fedmed-pl-superlink] copied known_hosts to {known_hosts_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[fedmed-pl-superlink] WARNING: failed to copy known_hosts: {exc}",
                    flush=True,
                )
                known_hosts_path = None

    ssh_opts: list[str] = [
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
    ]

    if known_hosts_path and known_hosts_path.exists():
        ssh_opts.extend([
            "-o", f"UserKnownHostsFile={known_hosts_path}",
            "-o", "StrictHostKeyChecking=yes",
        ])
    else:
        ssh_opts.extend(["-o", "StrictHostKeyChecking=no"])

    cmd: list[str] = [
        "ssh",
        "-vv",
        "-N",
        *ssh_opts,
        "-p", str(options.bastion_port),
        "-i", str(key_path),
        "-R", f"0.0.0.0:{options.bastion_fleet_port}:{fleet_local}",
        "-R", f"0.0.0.0:{options.bastion_control_port}:{control_local}",
        "-R", f"0.0.0.0:{options.bastion_serverapp_port}:{serverapp_local}",
        f"{bastion_user}@{bastion_host}",
    ]

    #_check_superlink_reachable(options.superlink_host, options.bastion_port)

    print(f"[fedmed-pl-superlink] opening reverse tunnels: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _register_child(proc)
    threading.Thread(target=_stream_lines, args=(proc.stdout, "ssh"), daemon=True).start()
    threading.Thread(target=_stream_lines, args=(proc.stderr, "ssh"), daemon=True).start()
    return proc


def handle_signals() -> None:
    def _handle(signum, _frame):  # type: ignore[override]
        print(f"\n[fedmed-pl-superlink] received signal {signum}, cleaning up...", flush=True)
        _cleanup_children()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _launch_superlink(addresses: dict[str, str], env: dict[str, str]) -> Process:
    """Start the long-lived Flower SuperLink services inside this container."""
    cmd: List[str] = [
        "flower-superlink",
        "--insecure",
        f"--fleet-api-address={addresses['fleet']}",
        f"--control-api-address={addresses['control']}",
        f"--serverappio-api-address={addresses['serverapp']}",
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
    control_address: str,
    run_config: str,
    env: dict[str, str],
) -> dict[str, Any]:
    """Bundle the Flower App, run `flwr run`, and return the parsed summary."""
    print("[fedmed-pl-superlink] staging Flower app...", flush=True)
    staged_app_dir, cleanup_app = _stage_flower_app(Path(env["FLWR_HOME"]))
    print(f"[fedmed-pl-superlink] staged app at {staged_app_dir}", flush=True)
    fed_config = f"address='{control_address}' insecure=true"
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
    print("[fedmed-pl-superlink] waiting for flwr run to finish...", flush=True)
    exit_code = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    print(f"[fedmed-pl-superlink] flwr run exited with {exit_code}", flush=True)
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
    min_memory_limit="8Gi",
    min_cpu_limit="1000m",
)
def main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    print(
        "\n==============================="
        "\n=== FedMed Flower SuperLink ===\n"
        "===============================\n",
        flush=True,
    )
    handle_signals()

    # DEBUG: show what pl-dircopy actually put into /incoming
    print(f"[fedmed-pl-superlink] DEBUG: inputdir = {inputdir}", flush=True)
    root = inputdir
    if not root.exists():
        print(f"[fedmed-pl-superlink] DEBUG: {root} does not exist", flush=True)
    else:
        for p in root.rglob("*"):
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            print(f"  {root}/{rel}", flush=True)

    if getattr(options, "json", False):
        emit_plugin_json()
        return

    if options.total_clients <= 0:
        raise ValueError("total-clients must be >= 1")

    addresses = {
        "fleet": f"{options.host}:{options.fleet_port}",
        "control": f"{options.host}:{options.control_port}",
        "serverapp": f"{options.host}:{options.serverapp_port}",
    }
    run_config = (
        f"num-server-rounds={options.rounds} "
        f"total-clients={options.total_clients} "
        f"local-epochs={options.local_epochs} "
        f"learning-rate={options.learning_rate} "
        f"data-seed={options.data_seed} "
        f"fraction-evaluate={options.fraction_evaluate}"
    )
    
    # Get environment and temp flower working directory
    env, flwr_home = _prepare_environment(options.state_dir)

    # Define output path for weights
    env["FEDMED_OUTPUT_CKPT"] = str((outputdir / DEFAULT_WEIGHTS).resolve())

    print(
        f"[fedmed-pl-superlink] SuperNodes should target Fleet API at {addresses["fleet"]}",
        flush=True,
    )
    reachable_ips = _discover_ipv4_addresses()
    if reachable_ips:
        print(
            "[fedmed-pl-superlink] reachable IPv4 addresses: "
            + ", ".join(reachable_ips),
            flush=True,
        )
        # Use the first container IP as the target for SSH -R
        tunnel_ip = reachable_ips[0]
        print(
            f"[fedmed-pl-superlink] using {tunnel_ip} as SSH -R backend target",
            flush=True,
        )
    else:
        print(
            "[fedmed-pl-superlink] unable to auto-detect host IPs; "
            "falling back to 127.0.0.1 for SSH -R backend (may fail on host)",
            flush=True,
        )
        tunnel_ip = "127.0.0.1"

    # Local targets *from the EC2 host's perspective* for SSH -R
    fleet_local     = f"{tunnel_ip}:{options.fleet_port}"
    control_local   = f"{tunnel_ip}:{options.control_port}"
    serverapp_local = f"{tunnel_ip}:{options.serverapp_port}"

    # Open reverse tunnels to bastion (no-op if bastion_* not set)
    _maybe_open_reverse_tunnels(options, inputdir, fleet_local, control_local, serverapp_local)

    superlink = _launch_superlink(addresses, env)
    time.sleep(max(0, options.startup_delay))
    try:
        summary = _run_federation(options, addresses["control"], run_config, env)
    finally:
        _terminate_process(superlink)
        _cleanup_children()

    summary_path = outputdir / options.summary_file
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[fedmed-pl-superlink] wrote summary to {summary_path}", flush=True)

    # Get rid of temp flower working directory
    shutil.rmtree(flwr_home, ignore_errors=True)
    print(f"[fedmed-pl-superlink] cleaned {flwr_home}", flush=True)


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
