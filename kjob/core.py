"""Submit, monitor and clean up Kubeflow PyTorchJobs that mirror the current notebook's environment.

The job copies the notebook pod's image, working directory, security context and the volumes mounted into the
notebook container, so code, virtualenvs and data on those volumes are visible at the same paths. Volumes that a
namespace PodDefault injects are left out (the PodDefault webhook adds them again and rejects duplicates).
ReadWriteOnce claims that are backed by NFS are mounted as the underlying NFS export so jobs can run on other nodes.
"""

from __future__ import annotations

import datetime as _dt
import copy
import json
import os
import re
import shlex
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity
from urllib3.exceptions import HTTPError

from . import cluster
from ._output import DISPLAY_PROGRAM, frame_prefix, pod_output_chunks

GROUP, VERSION, PLURAL = "kubeflow.org", "v1", "pytorchjobs"
MANAGED_BY = "kjob"
SIDECAR_CONTAINERS = {"istio-proxy"}
# Notebook env that must not reach a job: kjob sets these itself (GPU visibility depends on the job's own GPUs).
KJOB_ENV = {"PYTHONUNBUFFERED", "KJOB_NAME", "KJOB_NODE", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
GPU_PRODUCTS = {  # short names for --gpu-type; any other value is used as the raw nvidia.com/gpu.product label
    "a5000": "NVIDIA-RTX-A5000",
    "v100": "Tesla-V100-SXM2-16GB",
    "a30": "NVIDIA-A30",                 # whole GPU
    "a30-shared": "NVIDIA-A30-SHARED",   # MPS slice; still counts as one nvidia.com/gpu in the quota
}
# NFS volumes written by the group / namespace share PodDefaults. A notebook started before its PodDefault changed
# keeps the old ones (e.g. a group the user has since left); the PodDefault adds the current ones to every job.
MANAGED_VOLUME_PREFIXES = ("gs-", "ns-")
# On MPS nodes (product label "...-SHARED", sharing-strategy=mps) the device plugin mounts its own /dev/shm into GPU
# containers; a second /dev/shm volume makes the container fail with "Duplicate mount point: /dev/shm".
SHARING_STRATEGY_LABEL = "nvidia.com/gpu.sharing-strategy"
TERMINAL = ("Succeeded", "Failed", "Deleted")  # Deleted: the job is gone (kcancel elsewhere, TTL)
DEFAULT_CPUS, DEFAULT_MEMORY = 1, "2Gi"
DEFAULT_SHM = None   # /dev/shm limit = --mem: only a cap (its use counts toward memory), big enough for DataLoader workers
STATUS_GRACE_SECONDS = 30  # after the log stream ends, do not wait forever for an operator update
LOG_FRAGMENT_CHARS = 64 * 1024  # progress output may never contain a newline
EVICTION_GRACE_SECONDS = 10  # how long to wait for an Evicted reason after the runtime diagnostic
RECONNECT_SECONDS = 3  # before following again a log stream that ended while the script still runs
# Docker's message when the log stream outlives an evicted container
RUNTIME_GONE = re.compile(r"^rpc error: code = (?:Unknown|NotFound) desc = .*No such container\b")


class LogFragment(str):
    """A partial log record to display without appending a newline."""


def unique_name(stem: str) -> str:
    try:
        base = sanitize_name(stem)[:37].rstrip("-")
    except ValueError:      # e.g. a script name without ASCII letters or digits
        base = "job"
    return f"{base}-{uuid.uuid4().hex[:12]}"


def _matches(selector: dict, labels: dict) -> bool:
    if any(labels.get(k) != v for k, v in selector.get("matchLabels", {}).items()):
        return False
    for e in selector.get("matchExpressions", []):
        key, op, values = e["key"], e["operator"], e.get("values", [])
        if op == "In" and labels.get(key) not in values:
            return False
        if op == "NotIn" and key in labels and labels[key] in values:
            return False
        if op == "Exists" and key not in labels:
            return False
        if op == "DoesNotExist" and key in labels:
            return False
        if op not in {"In", "NotIn", "Exists", "DoesNotExist"}:
            raise ValueError(f"Unsupported PodDefault selector operator: {op}")
    return True


def _load_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def current_namespace() -> str:
    f = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    return f.read_text().strip() if f.exists() else os.environ.get("NB_NAMESPACE", "default")


def current_notebook() -> str:
    """Name of the notebook pod this code runs in (the job label kjob/notebook)."""
    return socket.gethostname()


def sanitize_name(name: str) -> str:
    """Job names: lowercase alphanumerics and '-', starting with a letter (the operator requires a DNS-1035 label),
    at most 50 chars (pods add '-master-0')."""
    n = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not n:
        raise ValueError(f"invalid job name {name!r}")
    if not n[0].isalpha():
        n = "j-" + n
    return n[:50].rstrip("-")


def _nfs_mounts() -> dict[str, tuple[str, str]]:
    """mount point -> (server, export path) for NFS mounts visible in this container."""
    out = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        src, mnt, fstype = [re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), x)
                            for x in line.split()[:3]]
        if fstype.startswith("nfs") and ":" in src:
            server, path = src.rsplit(":", 1)
            out[mnt] = (server, path)
    return out


def main_container(pod: client.V1Pod) -> client.V1Container:
    """The notebook's own container: named by the default-container annotation or the notebook-name label, or the
    only non-sidecar container. Container order is not used, so an extra helper container is never picked."""
    candidates = [c for c in pod.spec.containers if c.name not in SIDECAR_CONTAINERS]
    for want in ((pod.metadata.annotations or {}).get("kubectl.kubernetes.io/default-container"),
                 (pod.metadata.labels or {}).get("notebook-name")):
        hit = [c for c in candidates if want and c.name == want]
        if hit:
            return hit[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot tell which container of {pod.metadata.name} is the notebook "
                     f"({', '.join(c.name for c in candidates)}); set the kubectl.kubernetes.io/default-container annotation.")


# Volume sources a job on another node sees with the same content (emptyDir, hostPath, configMap ... do not count).
SHARED_SOURCES = ("nfs", "persistent_volume_claim", "cephfs", "glusterfs", "csi", "flex_volume", "azure_file", "quobyte")


def _is_shared(v: client.V1Volume) -> bool:
    return any(getattr(v, a, None) is not None for a in SHARED_SOURCES)


def _is_token(v: client.V1Volume) -> bool:
    """The service-account token volume Kubernetes adds (jobs run without API access); other projected volumes stay."""
    if v.name.startswith("kube-api-access"):
        return True
    sources = (v.projected.sources or []) if v.projected else []
    return any(s.service_account_token for s in sources) and all(
        s.service_account_token or s.downward_api or (s.config_map and s.config_map.name == "kube-root-ca.crt")
        for s in sources)


def _env_from_key(e: dict) -> tuple:
    ref = "configMapRef" if e.get("configMapRef") else "secretRef"
    return e.get("prefix"), ref, (e.get(ref) or {}).get("name")


def _toleration_key(t: dict) -> tuple:
    return t.get("key"), t.get("operator") or "Equal", t.get("value"), t.get("effect")


def _own_container_refs(e: client.V1EnvVar, notebook_container: str) -> client.V1EnvVar:
    """A resourceFieldRef naming the notebook container would point at a container the job does not have."""
    ref = e.value_from and e.value_from.resource_field_ref
    if ref is None or ref.container_name != notebook_container:
        return e
    e = copy.deepcopy(e)
    e.value_from.resource_field_ref.container_name = None    # default: the container the variable is in
    return e


def job_shell(name: str, command: str, workdir: str, log: str, run_id: str = "",
              termination_log: str = "/dev/termination-log", live_progress: bool = False) -> str:
    """Container command. Output goes to the pod log and to the log file. The script runs in its own process group
    and gets the SIGTERM sent at the -t limit or by kcancel, so it can save state within the grace period
    (bash as PID 1 would otherwise ignore SIGTERM and everything is killed 30 s later).

    The pod log is plain text unless live_progress: then it is framed (see _output) so that CR progress updates
    reach ksrun/kwait at once, at the cost of a pod log that only kjob reads well."""
    q, ts = shlex.quote, "$(date -u +%FT%TZ)"
    run = f" run {run_id}" if run_id else ""      # on every [kjob] line: runs sharing a log file stay apart
    result = '{"kjob":1,"run_id":' + json.dumps(run_id) + ',"log_saved":%s,"script_exit_code":%s}\\n'
    # Frame only the Pod stream; tee saves the original bytes. JSON escaping
    # keeps user CR, LF and literal protocol-looking text distinct from framing.
    display = f'-u -c {q(DISPLAY_PROGRAM)} {q(frame_prefix(run_id))}'
    return (
        "set -o pipefail\n"
        f"log={q(log)}\n"
        # Kubelet stores this outside the shared log volume, even when that volume is full.
        f'write_result() {{ (printf {q(result)} "$1" "$2" > {q(termination_log)}) 2>/dev/null || :; }}\n'
        # without a writable log the output would be lost while the job still reports success: stop before starting
        f'if ! {{ mkdir -p {q(str(Path(log).parent))} && : >> "$log"; }} 2>/dev/null; then\n'
        f'  echo "[kjob {ts}] cannot write the log file $log (missing directory, a file in the way, or no permission);'
        ' the script was not started"; write_result false null; exit 74; fi\n'
        f'cd {q(workdir)} || {{ echo "[kjob {ts}] cannot enter the working directory "{q(workdir)} | tee -a "$log"; '
        'if [ "$?" -eq 0 ]; then write_result true null; else write_result false null; fi; exit 1; }\n'
        'state="${TMPDIR:-/tmp}/kjob-$HOSTNAME-$$"\n'
        + ('display_output() {\n'
           'display_python=\n'
           f'for candidate in {q(sys.executable)} python3 python; do\n'
           f'  if "$candidate" {display} </dev/null >/dev/null 2>&1; then display_python="$candidate"; break; fi\n'
           'done\n'
           '  if [ -n "$display_python" ]; then\n'
           f'    "$display_python" {display} || {{ '
           f'echo; echo "[kjob {ts}] display formatter failed; continuing with plain output; the saved log is unchanged{run}"; cat; }}\n'
           f'  else echo "[kjob {ts}] display Python unavailable; using plain output; progress may be delayed{run}"; cat; fi\n'
           '}\n' if live_progress else "") +
        f"trap 'echo \"[kjob {ts}] stopping: SIGTERM received{run}\" | tee -a \"$log\"; "
        'p=$(cat "$state.pid" 2>/dev/null); [ -n "$p" ] && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }\' TERM INT\n'
        f'{{ echo "[kjob {ts}] start {name} on node $KJOB_NODE{run}"\n'
        # background commands start with SIGINT ignored; restore it so KeyboardInterrupt works in the script
        f"  if command -v setsid >/dev/null 2>&1; then (trap - INT QUIT; exec setsid {command}) &\n"
        f"  else (trap - INT QUIT; exec {command}) & fi\n"
        '  echo $! > "$state.pid"; wait $!; rc=$?\n'
        f'  echo "[kjob {ts}] exit code $rc{run}"; echo $rc > "$state.rc"; }} 2>&1 '
        '| { tee -a "$log"; echo $? > "$state.tee"; }' + (" | display_output" if live_progress else "") + " &\n"
        "pipeline_pid=$!\n"
        # Always reap the pipeline, even if a very short job finished before this shell got here.
        # A TERM trap can interrupt wait; wait again while the pipeline is still alive.
        "while :; do wait $pipeline_pid; kill -0 $pipeline_pid 2>/dev/null || break; done\n"
        'tee_rc=$(cat "$state.tee" 2>/dev/null || echo 1)\n'
        'rc=$(cat "$state.rc" 2>/dev/null || echo 1)\n'
        'if [ "$tee_rc" -ne 0 ]; then\n'
        '  write_result false "$rc"\n'
        '  [ "$rc" -ne 0 ] || rc=74\n'
        f'  echo "[kjob {ts}] log write failed: $log; recover the pod log with klogs {name}{run}"\n'
        f'  echo "[kjob {ts}] exit code $rc{run}"\n'
        'else write_result true "$rc"; fi\n'
        'rm -f "$state.pid" "$state.rc" "$state.tee"\n'
        'exit "$rc"\n')


@dataclass
class NotebookTemplate:
    """What a job inherits from the notebook pod this code runs in."""
    namespace: str
    pod_name: str
    node: str
    image: str
    working_dir: str
    volumes: list = field(default_factory=list)          # V1Volume
    volume_mounts: list = field(default_factory=list)    # V1VolumeMount
    security_context: object = None
    container_security_context: object = None
    effective_uid: int = field(default_factory=os.getuid)
    job_labels: dict = field(default_factory=dict)
    image_pull_secrets: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)          # volume name -> reason
    env: list = field(default_factory=list)              # V1EnvVar of the notebook container, minus PodDefault ones
    env_from: list = field(default_factory=list)         # V1EnvFromSource, minus PodDefault ones
    runtime_class_name: str | None = None
    tolerations: list = field(default_factory=list)      # V1Toleration set on the notebook itself
    poddefaults_known: bool = True                       # False: PodDefaults unreadable, nothing left to the webhook
    injected_mounts: list = field(default_factory=list)  # V1VolumeMount a PodDefault adds to the job again

    @classmethod
    def from_current_pod(cls, shm_size: str = DEFAULT_SHM) -> "NotebookTemplate":
        _load_config()
        ns = current_namespace()
        core = client.CoreV1Api()
        pod = core.read_namespaced_pod(current_notebook(), ns)
        main = main_container(pod)
        injected, injected_env, injected_env_from, injected_tol = set(), set(), set(), set()
        job_labels = {}
        labels = pod.metadata.labels or {}
        known = True
        try:
            for pd in client.CustomObjectsApi().list_namespaced_custom_object("kubeflow.org", "v1alpha1", ns, "poddefaults")["items"]:
                selector = pd["spec"].get("selector") or {}
                # every label any selector looks at goes to the job, so the job matches exactly the PodDefaults the
                # notebook matches (a NotIn/DoesNotExist selector would otherwise start matching the job)
                keys = set(selector.get("matchLabels") or {}) | {e["key"] for e in selector.get("matchExpressions") or []}
                job_labels.update({k: labels[k] for k in keys if k in labels})
                if not _matches(selector, labels):
                    continue
                injected |= {v["name"] for v in pd["spec"].get("volumes", [])}
                injected_env |= {e["name"] for e in pd["spec"].get("env", [])}
                injected_env_from |= {_env_from_key(e) for e in pd["spec"].get("envFrom", [])}
                injected_tol |= {_toleration_key(t) for t in pd["spec"].get("tolerations", [])}
        except ApiException as e:
            if e.status not in (403, 404):
                raise
            known = False   # the webhook will not add anything to the job either, so keep the notebook's copies
        api = client.ApiClient()
        mounts = {m.name: m for m in main.volume_mounts or []}
        nfs = _nfs_mounts()
        identity = {k: getattr(main.security_context, k, None)
                    for k in ("run_as_user", "run_as_group", "run_as_non_root")}
        identity = {k: v for k, v in identity.items() if v is not None}
        t = cls(namespace=ns, pod_name=pod.metadata.name, node=pod.spec.node_name, image=main.image,
                working_dir=os.getcwd(), security_context=pod.spec.security_context,
                container_security_context=client.V1SecurityContext(**identity) if identity else None,
                effective_uid=identity.get("run_as_user", getattr(pod.spec.security_context, "run_as_user", None)
                                           if pod.spec.security_context else None),
                job_labels=job_labels,
                image_pull_secrets=pod.spec.image_pull_secrets or [],
                env=[_own_container_refs(e, main.name) for e in main.env or [] if e.name not in injected_env | KJOB_ENV
                     and not e.name.startswith("CUDA_MPS_")],
                env_from=[e for e in main.env_from or []
                          if _env_from_key(api.sanitize_for_serialization(e)) not in injected_env_from],
                runtime_class_name=pod.spec.runtime_class_name,
                tolerations=[x for x in pod.spec.tolerations or []      # node.kubernetes.io/*: added by Kubernetes
                             if not (x.key or "").startswith("node.kubernetes.io/")
                             and _toleration_key(api.sanitize_for_serialization(x)) not in injected_tol],
                poddefaults_known=known)
        if t.effective_uid is None:
            t.effective_uid = os.getuid()
        for v in pod.spec.volumes or []:
            m = mounts.get(v.name)
            if m is None:
                t.skipped[v.name] = "not mounted in the notebook container"
            elif v.name in injected:
                t.skipped[v.name] = "added automatically by a PodDefault"
                if _is_shared(v):                                     # the job gets it again: still shared storage
                    source = v.nfs or v.persistent_volume_claim
                    t.injected_mounts.append(client.V1VolumeMount(
                        name=v.name, mount_path=m.mount_path,
                        read_only=bool(m.read_only or (source is not None and source.read_only))))
            elif known and v.name.startswith(MANAGED_VOLUME_PREFIXES):
                t.skipped[v.name] = "outdated group/namespace share mount (restart the notebook to refresh it)"
            elif _is_token(v):
                t.skipped[v.name] = "service-account token"
            elif v.persistent_volume_claim is not None and m.mount_path in nfs:
                if m.sub_path or m.sub_path_expr:
                    raise ValueError(f"NFS-backed PVC {v.name} uses subPath/subPathExpr; mount the full PVC before submitting.")
                server, path = nfs[m.mount_path]
                readonly = bool(v.persistent_volume_claim.read_only or m.read_only)
                t.volumes.append(client.V1Volume(name=v.name, nfs=client.V1NFSVolumeSource(server=server, path=path, read_only=readonly)))
                t.volume_mounts.append(client.V1VolumeMount(name=v.name, mount_path=m.mount_path, read_only=readonly))
            elif v.empty_dir is not None:
                # Only manifest() resizes /dev/shm. Scratch volumes keep their own medium and size limit.
                t.volumes.append(copy.deepcopy(v))
                t.volume_mounts.append(copy.deepcopy(m))
            else:  # PVC on non-NFS storage, nfs, secret, configMap, ...: reuse as-is
                t.volumes.append(v)
                t.volume_mounts.append(copy.deepcopy(m))
        return t

    def shared_paths(self) -> list[str]:
        """Mount points whose content the job sees too: network storage (emptyDir and hostPath are per pod/node)."""
        names = {v.name for v in self.volumes if _is_shared(v)}
        return [m.mount_path for m in self.volume_mounts if m.name in names] + [m.mount_path for m in self.injected_mounts]


@dataclass
class JobSpec:
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    gpus: int = 1
    cpus: int = DEFAULT_CPUS
    memory: str = DEFAULT_MEMORY
    shm: str | None = DEFAULT_SHM                     # None: same as memory
    gpu_types: list[str] = field(default_factory=list)   # e.g. ["a5000", "v100"]
    node: str | None = None                              # pin to one host
    exclude_nodes: list[str] = field(default_factory=list)
    interpreter: str | None = None                       # default: this Python for .py, bash for .sh
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    log_dir: str = str(Path.home() / "kjob-logs")
    ttl_seconds: int = 3600                              # safety net if nobody monitors the job
    time_limit: int | None = None                        # seconds; the job is stopped after this (like --time)
    live_progress: bool = False                          # framed pod log: CR progress shows at once (see job_shell)

    def validate(self) -> None:
        if not isinstance(self.gpus, int) or self.gpus < 0:
            raise ValueError("GPU count must be a nonnegative integer.")
        if not isinstance(self.cpus, int) or self.cpus < 1:
            raise ValueError("CPU count must be a positive integer.")
        if not isinstance(self.ttl_seconds, int) or self.ttl_seconds < 0:
            raise ValueError("TTL must be a nonnegative integer.")
        if self.time_limit is not None and (not isinstance(self.time_limit, int) or self.time_limit < 1):
            raise ValueError("Time limit must be a positive number of seconds.")
        for key, value in (("memory", self.memory), ("shared memory", self.shm or self.memory)):
            try:
                amount = parse_quantity(value)
                if not amount.is_finite() or amount <= 0:
                    raise ValueError()
            except (ValueError, TypeError, ArithmeticError):
                raise ValueError(f"Invalid {key} quantity: {value!r}") from None
        products = [GPU_PRODUCTS.get(g, g) for g in self.gpu_types]
        shared = [p.endswith("-SHARED") for p in products]
        if any(shared) and not all(shared):
            raise ValueError("Choose either whole GPUs or shared GPUs; do not mix them in --gpu-type.")
        if self.gpus > 1 and any(shared):
            raise ValueError("MPS supports only one shared GPU unit per container; use a whole GPU type for -G > 1.")
        if not self.gpus and self.gpu_types:
            raise ValueError("CPU-only jobs cannot select a GPU type.")
        protected = {"KJOB_NAME", "KJOB_NODE", "PYTHONUNBUFFERED", "NVIDIA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY",
                     "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"}
        if not self.gpus:
            protected.add("CUDA_VISIBLE_DEVICES")
        if any(shared):
            protected.add("CUDA_VISIBLE_DEVICES")
        if protected.intersection(self.env):
            raise ValueError("Reserved job environment variables cannot be overridden: " + ", ".join(sorted(protected.intersection(self.env))))
        for k in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise ValueError(f"Invalid environment variable name: {k!r}")

    def command(self) -> str:
        script = str(Path(self.script).resolve()) if Path(self.script).exists() else self.script
        if self.interpreter:
            prog = [self.interpreter, script]
        elif script.endswith(".py"):
            prog = [sys.executable, script]
        elif script.endswith(".sh"):
            prog = ["bash", script]
        else:
            prog = [script]
        return " ".join(shlex.quote(p) for p in prog + list(self.args))

    def log_path(self) -> str:
        return str(Path(self.log_dir) / f"{self.name}.log")


class KJob:
    """Create / monitor / delete one PyTorchJob built from a NotebookTemplate."""

    def __init__(self, spec: JobSpec, template: NotebookTemplate | None = None):
        spec.validate()
        self.spec = spec
        self.spec.name = sanitize_name(spec.name)
        self.t = template or NotebookTemplate.from_current_pod(spec.shm or spec.memory)
        self.ns = self.t.namespace
        self.core = client.CoreV1Api()
        self.custom = client.CustomObjectsApi()
        self.pod_name = f"{self.spec.name}-master-0"
        self.run_id = uuid.uuid4().hex[:12]   # printed in the log's start line; ties a run to this submission

    # ---------- manifest ----------
    def manifest(self) -> dict:
        s, t = self.spec, self.t
        s.validate()
        workdir = s.workdir or t.working_dir
        log = s.log_path()
        shell = job_shell(s.name, s.command(), workdir, log, self.run_id, live_progress=s.live_progress)
        res = {"cpu": str(s.cpus), "memory": s.memory}
        if s.gpus:
            res["nvidia.com/gpu"] = str(s.gpus)  # CPU-only jobs must not request GPUs at all
        # the notebook's own env; --env replaces a value in place, so later $(VAR) references still expand
        env = [client.V1EnvVar(name=e.name, value=s.env[e.name]) if e.name in s.env else e for e in t.env]
        env += [client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
               client.V1EnvVar(name="KJOB_NAME", value=s.name),
               client.V1EnvVar(name="KJOB_NODE", value_from=client.V1EnvVarSource(
                   field_ref=client.V1ObjectFieldSelector(field_path="spec.nodeName")))]
        if not s.gpus:  # CUDA images may default NVIDIA_VISIBLE_DEVICES=all, exposing the node's GPUs without a request
            hide = {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": "-1"}
            env += [client.V1EnvVar(name=k, value=v) for k, v in hide.items()]
        inherited = {e.name for e in t.env}
        env += [client.V1EnvVar(name=k, value=v) for k, v in s.env.items() if k not in inherited]
        products = [GPU_PRODUCTS.get(g, g) for g in s.gpu_types]
        mps = bool(s.gpus) and any(p.endswith("-SHARED") for p in products)
        if mps and t.effective_uid != 0:
            raise ValueError("This installation only enables shared MPS jobs for UID 0 Notebooks; the administrator must also verify the server UID and health. Use a whole GPU type for a non-root Notebook.")
        shm = {m.name for m in t.volume_mounts if m.mount_path == "/dev/shm"}
        volumes = [v for v in t.volumes if v.name not in shm]
        volume_mounts = [m for m in t.volume_mounts if m.name not in shm]
        if not mps:  # --shm sizes /dev/shm whether or not the notebook has one (the device plugin provides it for MPS)
            name = next(iter(shm), "kjob-shm")
            volumes.append(client.V1Volume(name=name, empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit=s.shm or s.memory)))
            volume_mounts.append(client.V1VolumeMount(name=name, mount_path="/dev/shm"))
        container = client.V1Container(
            name="pytorch", image=t.image, command=["bash", "-lc", shell], working_dir=workdir, env=env,
            termination_message_policy="FallbackToLogsOnError",
            env_from=t.env_from or None,
            security_context=t.container_security_context,
            resources=client.V1ResourceRequirements(requests=res, limits=res), volume_mounts=volume_mounts)
        terms, fields = [], []      # nodes are pinned by metadata.name: the hostname label may differ from it
        if s.node:
            fields.append(client.V1NodeSelectorRequirement(key="metadata.name", operator="In", values=[s.node]))
        if s.gpu_types:
            terms.append(client.V1NodeSelectorRequirement(key="nvidia.com/gpu.product", operator="In", values=products))
        if s.gpus and not mps:
            terms.append(client.V1NodeSelectorRequirement(key=SHARING_STRATEGY_LABEL, operator="NotIn", values=["mps"]))
        elif mps:
            terms.append(client.V1NodeSelectorRequirement(key=SHARING_STRATEGY_LABEL, operator="In", values=["mps"]))
        if s.exclude_nodes:
            fields.append(client.V1NodeSelectorRequirement(key="metadata.name", operator="NotIn", values=s.exclude_nodes))
        affinity = client.V1Affinity(node_affinity=client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                node_selector_terms=[client.V1NodeSelectorTerm(match_expressions=terms or None, match_fields=fields or None)]))) \
            if terms or fields else None
        pod_spec = client.V1PodSpec(containers=[container], volumes=volumes, affinity=affinity,
                                    security_context=t.security_context, image_pull_secrets=t.image_pull_secrets or None,
                                    runtime_class_name=t.runtime_class_name, tolerations=t.tolerations or None,
                                    automount_service_account_token=False,  # the script does not need API access
                                    # Enforced by the kubelet from Pod start, so queueing time does not count
                                    # (runPolicy.activeDeadlineSeconds is only checked when the operator reconciles).
                                    active_deadline_seconds=s.time_limit)
        template = client.V1PodTemplateSpec(
            # kjob/log-stream tells klogs/kwait that the pod log is framed; plain logs are read as they are
            metadata=client.V1ObjectMeta(annotations={"sidecar.istio.io/inject": "false",
                                                      **({"kjob/log-stream": self.run_id} if s.live_progress else {})},
                                         labels={**t.job_labels, "app.kubernetes.io/managed-by": MANAGED_BY}),
            spec=pod_spec)
        api = client.ApiClient()
        return {
            "apiVersion": f"{GROUP}/{VERSION}", "kind": "PyTorchJob",
            "metadata": {"name": s.name, "namespace": self.ns,
                         "labels": {"app.kubernetes.io/managed-by": MANAGED_BY, "kjob/notebook": t.pod_name},
                         "annotations": {"kjob/command": s.command()[:2000], "kjob/log": log,
                                         "kjob/run-id": self.run_id,
                                         "kjob/workdir": workdir,
                                         "kjob/submitted": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "spec": {
                # RestartPolicy Never already makes script failures terminal. A zero backoffLimit
                # on older operators deletes a failed Pod before recording Failed and can recreate it.
                "runPolicy": {"cleanPodPolicy": "None", "ttlSecondsAfterFinished": s.ttl_seconds},
                "pytorchReplicaSpecs": {"Master": {"replicas": 1, "restartPolicy": "Never",
                                                   "template": api.sanitize_for_serialization(template)}},
            },
        }

    # ---------- lifecycle ----------
    def submit(self) -> str:
        if self.spec.gpus and self.spec.node and not self.spec.gpu_types:
            labels = self._node_labels(self.spec.node)
            if labels.get(SHARING_STRATEGY_LABEL) == "mps":
                product = labels.get("nvidia.com/gpu.product", "")
                if not product.endswith("-SHARED"):
                    raise ValueError("MPS node does not advertise a -SHARED product label; ask the administrator to configure GPU selection.")
                self.spec.gpu_types = [product]
        self.spec.validate()
        try:
            quotas = self.core.list_namespaced_resource_quota(self.ns).items
        except ApiException as e:
            if e.status != 403:
                raise
            quotas = []
        requested = {"requests.cpu": self.spec.cpus, "requests.memory": self.spec.memory,
                     "requests.nvidia.com/gpu": self.spec.gpus}
        for q in quotas:
            hard = q.status.hard or {}
            for key, value in requested.items():
                if key in hard and parse_quantity(value) > parse_quantity(hard[key]):
                    raise ValueError(f"Request {key}={value} exceeds namespace quota {q.metadata.name} hard limit {hard[key]}; waiting cannot satisfy it.")
        self._warn_unshared_paths()
        obj = self.custom.create_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.manifest())
        self.uid = obj["metadata"]["uid"]
        record_history(self)
        return self.spec.name

    def _node_labels(self, name: str) -> dict:
        """Labels of a pinned node: read directly, else from the published cluster status, else assume a whole-GPU node."""
        try:
            return self.core.read_node(name).metadata.labels or {}
        except ApiException as e:
            if e.status != 403:
                raise
        try:
            node = next((n for n in cluster.load_snapshot(self.core)["nodes"] if n["name"] == name), None)
        except (cluster.ClusterViewUnavailable, ApiException, ValueError):
            node = None
        if node is None:
            print(f"kjob: cannot inspect node {name}; requesting whole GPUs there (use -g ...-shared for an MPS node)",
                  file=sys.stderr)
            return {}
        labels = {SHARING_STRATEGY_LABEL: node["sharing"]} if node["sharing"] else {}
        if node["product"]:
            labels["nvidia.com/gpu.product"] = node["product"]
        return labels

    def _warn_unshared_paths(self) -> None:
        shared = self.t.shared_paths()
        script = Path(self.spec.script).expanduser().resolve()
        paths = [self.spec.workdir or self.t.working_dir, self.spec.log_dir]
        if script.is_file():
            paths.append(str(script))
        elif "/" in self.spec.script or not shutil.which(self.spec.script):
            raise ValueError(f"Script is not a regular file: {script}")
        # else: a command on PATH (e.g. nvidia-smi); the job uses the same image, so it is there too
        mounts = self.t.volume_mounts + self.t.injected_mounts
        for p in paths:
            resolved = str(Path(p).expanduser().resolve())
            covering = [m for m in mounts if resolved == m.mount_path or
                        resolved.startswith(m.mount_path.rstrip("/") + "/")]
            closest = max(covering, key=lambda m: len(m.mount_path), default=None)
            if not closest or closest.mount_path not in shared:
                raise ValueError(f"{resolved} is not on a shared volume; place scripts, working directory and logs on the Notebook workspace.")
        log = str(Path(self.spec.log_dir).expanduser().resolve())
        covering = max((m for m in mounts if log == m.mount_path or
                        log.startswith(m.mount_path.rstrip("/") + "/")), key=lambda m: len(m.mount_path))
        volume = next((v for v in self.t.volumes if v.name == covering.name), None)   # None: added by a PodDefault
        source = volume and (volume.nfs or volume.persistent_volume_claim)
        if covering.read_only or (source is not None and source.read_only):
            raise ValueError("Log directory is on a read-only volume; choose a writable workspace.")

    def state(self) -> str:
        try:
            obj = self.custom.get_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.spec.name)
            uid = obj["metadata"]["uid"]
            if getattr(self, "uid", uid) != uid:
                return "Deleted"
            self.uid = uid
            # kwait starts without the run id. An evicted container is SIGKILLed before the wrapper can report it,
            # so take it from the CR: kacct matches the evicted/log-error events by it.
            run_id = (obj["metadata"].get("annotations") or {}).get("kjob/run-id")
            if run_id and not getattr(self, "run_id", None):
                self.run_id = run_id
            return job_state(obj)
        except ApiException as e:
            if e.status == 404:
                return "Deleted"
            raise

    def delete(self) -> None:
        if not getattr(self, "uid", None) and self.state() == "Deleted":
            return
        delete_job(self.ns, self.spec.name, uid=self.uid)

    def _pod(self):
        pod = self.core.read_namespaced_pod(self.pod_name, self.ns)
        if getattr(self, "uid", None) and not any(o.uid == self.uid and o.kind == "PyTorchJob"
                                                 for o in pod.metadata.owner_references or []):
            raise ApiException(status=404, reason="Pod belongs to a different job UID")
        return pod

    def wait_for_pod(self, timeout: float | None = None, report=print) -> str:
        """Block until the pod leaves Pending (or the job ends); report scheduling problems while waiting."""
        t0, seen = time.time(), set()
        while True:
            st = self.state()
            if st == "Deleted":
                return st
            try:
                pod = self._pod()
                phase = pod.status.phase
                if phase != "Pending":
                    return phase
                for c in pod.status.conditions or []:
                    if c.type == "PodScheduled" and c.status == "False" and c.message and c.message not in seen:
                        seen.add(c.message)
                        report(f"kjob: {self.spec.name} pending: {c.message}")
                for cs in pod.status.container_statuses or []:
                    w = cs.state.waiting
                    if w and w.reason not in (None, "ContainerCreating") and w.reason not in seen:
                        seen.add(w.reason)
                        report(f"kjob: {self.spec.name} waiting: {w.reason} {w.message or ''}")
            except ApiException as e:
                if e.status != 404:
                    raise
                for ev in job_events(self.ns, self.spec.name, uid=getattr(self, "uid", None)):
                    if ev not in seen:
                        seen.add(ev)
                        report(f"kjob: {self.spec.name} event: {ev}")
                    if "FailedValidation" in ev.split(":", 1)[0]:
                        return "Invalid"      # the operator rejected the job; it will never get a pod
            if st in TERMINAL:
                return st
            if timeout is not None and time.time() - t0 > timeout:
                return "Pending"
            time.sleep(3)

    def stream_logs(self, since_seconds: int | None = None) -> Iterator[str]:
        """Yield lines and carriage-return updates until the container exits (or the stream is cut)."""
        pod = self._pod()
        since = {"since_seconds": since_seconds} if since_seconds else {}
        resp = self.core.read_namespaced_pod_log(self.pod_name, self.ns, container="pytorch", follow=True,
                                                 _preload_content=False, **since)
        buf = ""
        progress = False
        try:
            for chunk in pod_output_chunks(resp, pod):
                buf += chunk
                while "\n" in buf or "\r" in buf or len(buf) >= LOG_FRAGMENT_CHARS:
                    ends = [i for i in (buf.find("\n"), buf.find("\r")) if i >= 0]
                    end = min(ends) if ends else -1
                    if end < 0 or end > LOG_FRAGMENT_CHARS:
                        yield LogFragment(buf[:LOG_FRAGMENT_CHARS])
                        buf = buf[LOG_FRAGMENT_CHARS:]
                    else:
                        progress = buf[end] == "\r"
                        yield buf[:end + 1] if buf[end] == "\r" else buf[:end]
                        buf = buf[end + 1:]
                # tqdm starts updates with CR. Display the following text now,
                # without waiting for the next update's CR or adding a newline.
                if progress and buf:
                    yield LogFragment(buf)
                    buf = ""
            if buf:
                yield buf
        finally:
            resp.close()

    def exec(self, command: list[str] | None = None, stdin=None, out=None, err=None) -> int:
        """Run a command in the running job container (like srun --jobid=ID --pty bash); return its exit code.

        Without a command, open an interactive login shell (needs a terminal)."""
        from kubernetes.stream import stream
        stdin, out, err = stdin or sys.stdin, out or sys.stdout, err or sys.stderr
        pod = self._pod()
        if pod.status.phase != "Running":
            raise ValueError(f"{self.spec.name} is not running (pod {pod.status.phase}); see kshow {self.spec.name}.")
        tty = not command
        if tty and not stdin.isatty():
            raise ValueError("An interactive shell needs a terminal; give a command instead, e.g. kexec NAME nvidia-smi.")
        resp = stream(self.core.connect_get_namespaced_pod_exec, self.pod_name, self.ns, container="pytorch",
                      command=command or ["bash", "-l"], stdin=tty, stdout=True, stderr=True, tty=tty,
                      _preload_content=False)
        saved = None
        try:
            if tty:
                import termios
                import tty as tty_mod
                fd = stdin.fileno()
                saved = termios.tcgetattr(fd)
                tty_mod.setraw(fd)
                size = shutil.get_terminal_size()
                resp.write_channel(4, json.dumps({"Width": size.columns, "Height": size.lines}))
            while resp.is_open():
                resp.update(timeout=0.1)
                if resp.peek_stdout():
                    out.write(resp.read_stdout())
                    out.flush()
                if resp.peek_stderr():
                    err.write(resp.read_stderr())
                    err.flush()
                if tty:
                    import select
                    if select.select([fd], [], [], 0)[0]:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        resp.write_stdin(data.decode(errors="replace"))
        finally:
            if saved is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            resp.close()
        code = resp.returncode
        return code if code is not None else 0

    def stop_reason(self) -> str | None:
        """Why the pod ended, when the pod says so (DeadlineExceeded at the -t limit)."""
        try:
            return self._pod().status.reason
        except ApiException:
            return None

    def evicted(self, wait: float = 0) -> bool:
        """Whether the Pod was evicted. Kubelet can set the reason a moment after the log stream ends, so wait up to
        `wait` seconds, but not for a Pod that is gone or has not started."""
        deadline = time.monotonic() + wait
        while True:
            try:
                status = self._pod().status
            except ApiException:
                return False
            if status.reason == "Evicted":
                return True
            if status.phase == "Pending" or time.monotonic() >= deadline:
                return False
            time.sleep(1)

    def is_evicted_log_error(self, line: str, wait: float = 0) -> bool:
        """Recognize a missing-runtime-container diagnostic only for an evicted Pod."""
        return bool(RUNTIME_GONE.match(line) and self.evicted(wait))

    def main_running(self) -> bool:
        """Whether the job's own container still runs (its log stream can end before it does)."""
        try:
            pod = self._pod()
        except ApiException:
            return False
        return any(cs.name == "pytorch" and cs.state is not None and cs.state.running is not None
                   for cs in pod.status.container_statuses or [])

    def user_lines(self, report, since_seconds: int | None = None) -> Iterator[str]:
        """stream_logs() without kjob's display notices and the runtime's after-eviction diagnostic."""
        held = []
        for line in self.stream_logs(since_seconds=since_seconds):
            if re.match(r"\[kjob [^\]]*\] display ", line):
                report(line)
                continue
            if RUNTIME_GONE.match(line):
                held.append(line)   # decided at the end: the line arrives before the Pod says Evicted
                continue
            yield from held         # more output followed, so it was the script's own
            held = []
            yield line
        if held and self.evicted(wait=EVICTION_GRACE_SECONDS):
            for line in held:
                report(f"kjob: {self.spec.name} log stream unavailable after eviction: {line}")
        else:
            yield from held

    def is_evicted_log_exception(self, error: ApiException) -> bool:
        """A log request refused because the evicted container is gone. On 400 the Pod still exists and its Evicted
        reason can lag, as for the diagnostic line above; 404 means the Pod is gone, so there is nothing to wait for."""
        return error.status in (400, 404) and self.evicted(wait=EVICTION_GRACE_SECONDS if error.status == 400 else 0)

    def exit_code(self) -> int | None:
        try:
            pod = self._pod()
        except ApiException as e:
            if e.status == 404:
                return None
            raise
        for cs in pod.status.container_statuses or []:     # only the job's own container: a sidecar's 0 is not ours
            if cs.name == "pytorch" and cs.state.terminated is not None:
                self.exit_reason = cs.state.terminated.reason  # e.g. OOMKilled, Error, Completed
                try:
                    result = json.loads(cs.state.terminated.message or "{}")
                    if (isinstance(result, dict) and result.get("kjob") == 1
                            and result.get("run_id") == getattr(self, "run_id", result.get("run_id"))):
                        self.run_id = result.get("run_id")
                        self.log_failed = result.get("log_saved") is False
                except (ValueError, TypeError):
                    pass  # old wrappers, OOM, or kubelet's fallback text: no structured result
                return cs.state.terminated.exit_code
        return None

    def follow(self, out=sys.stdout, report=None) -> int:
        """Wait for the pod, stream its output, wait for the job to end; return the script's exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        self.monitor_incomplete = False
        self.main_finished = False
        last_code = None
        phase = self.wait_for_pod(report=report)
        if phase == "Invalid":
            report(f"kjob: {self.spec.name} was rejected by the training operator (see the event above)")
            self.invalid = True
            return 1
        if phase != "Deleted":
            try:
                report(f"kjob: {self.spec.name} running on {self._pod().spec.node_name}")
                since, last_output = None, time.monotonic()
                while True:
                    try:
                        for line in self.user_lines(report, since_seconds=since):
                            last_output = time.monotonic()
                            m = re.match(r"\[kjob [^\]]*\] exit code (\d+)( run [0-9a-f]+)?$", line)
                            if m:
                                last_code = int(m.group(1))
                            else:
                                if re.match(r"\[kjob [^\]]*\] log write failed:", line):
                                    self.log_failed = True
                                print(line, end="" if isinstance(line, LogFragment) or line.endswith("\r") else "\n",
                                      file=out, flush=True)
                    except HTTPError:
                        pass  # connection cut mid-stream: decided below like a stream that ended
                    if last_code is not None:
                        break  # the wrapper reported the script's end
                    # The stream can end while the script runs: kubelet closes idle streams (no output for hours),
                    # the API server restarts, the network drops. The pause lets a container that just exited show
                    # as terminated, so its last lines are not printed twice.
                    time.sleep(RECONNECT_SECONDS)
                    if not self.main_running():
                        break
                    if since is None:
                        report(f"kjob: {self.spec.name} log stream interrupted while the job runs; following again "
                               "(the last few lines may repeat)")
                    since = int(time.monotonic() - last_output) + 1  # covers the gap; at most ~1 s repeats
            except ApiException as e:
                if self.is_evicted_log_exception(e):
                    report(f"kjob: {self.spec.name} log stream unavailable after eviction; "
                           f"use klogs {self.spec.name} to read the saved log")
                elif e.status == 404:
                    report(f"kjob: {self.spec.name} pod disappeared; checking job status")
                else:
                    raise
        st = self.state()
        code = self.exit_code()
        deadline = time.monotonic() + STATUS_GRACE_SECONDS
        while st not in TERMINAL:  # pod finished; the operator updates the job status shortly after
            if code is not None:
                # A crashed or lagging operator must not make a finished script wait forever.
                st = "Succeeded" if code == 0 else "Failed"
                break
            if time.monotonic() >= deadline:
                self.monitor_incomplete = True
                report(f"kjob: {self.spec.name} log stream ended but completion could not be confirmed. "
                       f"Keeping the job; reconnect with kwait {self.spec.name} or inspect kshow {self.spec.name}.")
                return 1
            time.sleep(2)
            st = self.state()
            code = self.exit_code()
        self.main_finished = code is not None
        code = code if code is not None else last_code
        if getattr(self, "log_failed", False):
            code = code if code not in (None, 0) else 74
            record_event("log-error", self.spec.name, run_id=getattr(self, "run_id", None), exit_code=code)
        reason = self.stop_reason() or getattr(self, "exit_reason", None)
        if reason == "DeadlineExceeded":       # stopped at -t: not a success even if the script saved and exited 0
            report(f"kjob: {self.spec.name} stopped at the time limit (-t); the script exited with {code} after SIGTERM. "
                   f"Exit status 124")
            return code if code not in (None, 0) else 124
        if reason == "Evicted":
            final = code if code not in (None, 0) else 1    # a script that saved and exited 0 still did not finish
            # kacct reads this after the job is deleted; the log alone looks like a time limit or kcancel
            record_event("evicted", self.spec.name, run_id=getattr(self, "run_id", None), exit_code=final)
            report(f"kjob: {self.spec.name} Evicted (script exit code {code})")
            return final
        detail = f", {reason}" if reason and reason not in ("Completed", "Error") else ""
        report(f"kjob: {self.spec.name} {st} (exit code {code}{detail})")
        return code if code is not None else (0 if st == "Succeeded" else 1)

    def cleanup_finished(self, report=print) -> bool:
        """Keep the last copy of the output when saving to the shared log failed."""
        if getattr(self, "monitor_incomplete", False):
            return False  # a monitoring failure is not a cancellation request
        self.exit_code()  # also reads the termination message when kwait was interrupted after completion
        if getattr(self, "log_failed", False):
            report(f"kjob: {self.spec.name} log was not saved completely; keeping the finished job and Pod. "
                   f"Recover with klogs {self.spec.name} > /path/on/writable/storage/{self.spec.name}.log, "
                   f"then kcancel {self.spec.name}. The configured --ttl still applies.")
            return False
        self.delete()
        return True

    def run(self, out=sys.stdout, cleanup: bool = True, report=None) -> int:
        """srun semantics: submit, stream output, wait, delete the job, return the exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        self.submit()
        report(f"kjob: submitted {self.spec.name} (log: {self.spec.log_path()})")
        try:
            code = self.follow(out=out, report=report)
        except (KeyboardInterrupt, BrokenPipeError) as e:   # Ctrl-C, or the reader of our output (| head) went away
            report(f"kjob: {'interrupted' if isinstance(e, KeyboardInterrupt) else 'output closed'}; "
                   f"cancelling {self.spec.name}")
            if cleanup:
                self.delete()
            return 130 if isinstance(e, KeyboardInterrupt) else 141
        if cleanup:
            self.cleanup_finished(report)
        return code


HISTORY_FILE = Path.home() / ".kjob" / "history.jsonl"


def record_history(job: "KJob") -> None:
    """Append the submitted job to ~/.kjob/history.jsonl (read by `kjob acct`); failures are ignored."""
    s = job.spec
    rec = {"name": s.name, "namespace": job.ns, "submitted": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "gpus": s.gpus, "gpu_types": s.gpu_types, "node": s.node, "cpus": s.cpus, "memory": s.memory,
           "time_limit": s.time_limit, "command": s.command()[:500], "log": s.log_path(), "run_id": job.run_id}
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _history_lines() -> list[dict]:
    try:
        lines = HISTORY_FILE.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def read_history() -> list[dict]:
    """Submission records (oldest first)."""
    return [r for r in _history_lines() if "event" not in r]


def read_events() -> list[dict]:
    """Other records, e.g. {"event": "cancel", "name": ..., "time": ...} written by kcancel."""
    return [r for r in _history_lines() if "event" in r]


def record_event(event: str, name: str, **details) -> None:
    rec = {"event": event, "name": name, "time": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           **details}
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def log_runs(path: str) -> list[dict]:
    """Runs recorded in a job log by the wrapper's [kjob ...] lines: start, end, exit code, stopped (SIGTERM), and the
    submission's run id (None in logs written before kjob printed it)."""
    runs = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"\[kjob (\S+)\] (start|exit code (\d+)|stopping)(?:.* run ([0-9a-f]+)\s*$)?", line)
                if not m:
                    continue
                rid = m.group(4)
                if m.group(2) == "start":
                    runs.append({"start": m.group(1), "end": None, "code": None, "stopped": False, "id": rid})
                    continue
                # exit/stopping lines name their run (older logs do not: then it is the latest run)
                run = next((r for r in reversed(runs) if r["id"] == rid), None) if rid else None
                if run is None:
                    if not runs:
                        runs.append({"start": None, "end": None, "code": None, "stopped": False, "id": rid})
                    run = runs[-1]
                if m.group(2) == "stopping":
                    run["stopped"], run["end"] = True, m.group(1)
                elif m.group(3):
                    run["end"], run["code"] = m.group(1), int(m.group(3))
    except OSError:
        pass
    return runs


def log_result(path: str, run_id: str | None = None) -> tuple[str | None, str | None, int | None]:
    """(start, end, exit code) of the last run in a job log, or of the run printed with run_id."""
    runs = [r for r in log_runs(path) if not run_id or r["id"] == run_id]
    return (runs[-1]["start"], runs[-1]["end"], runs[-1]["code"]) if runs else (None, None, None)


# ---------- namespace-level helpers (used by the CLI) ----------
def job_state(obj: dict) -> str:
    conds = (obj.get("status") or {}).get("conditions") or []
    active = [c["type"] for c in conds if str(c.get("status", "False")).lower() == "true"]
    for terminal in ("Failed", "Succeeded"):
        if terminal in active:
            return terminal
    return active[-1] if active else "Pending"


def list_jobs(ns: str | None = None, everyone: bool = False, notebook: str | None = None) -> list[dict]:
    """kjob jobs in the namespace (everyone: all PyTorchJobs); notebook: only jobs submitted from that notebook pod."""
    _load_config()
    ns = ns or current_namespace()
    sel = None if everyone else f"app.kubernetes.io/managed-by={MANAGED_BY}"
    if notebook and not everyone:
        sel += f",kjob/notebook={notebook}"
    return client.CustomObjectsApi().list_namespaced_custom_object(GROUP, VERSION, ns, PLURAL, label_selector=sel)["items"]


def delete_job(ns: str, name: str, uid: str | None = None) -> bool:
    api = client.CustomObjectsApi()
    try:
        if not uid:
            uid = api.get_namespaced_custom_object(GROUP, VERSION, ns, PLURAL, name)["metadata"]["uid"]
        api.delete_namespaced_custom_object(
            GROUP, VERSION, ns, PLURAL, name,
            body=client.V1DeleteOptions(propagation_policy="Background", preconditions=client.V1Preconditions(uid=uid)))
        return True
    except ApiException as e:
        if e.status in (404, 409):
            return False
        raise


def job_events(ns: str, name: str, limit: int = 5, uid: str | None = None) -> list[str]:
    """Recent warnings about the job; uid excludes events left by an earlier job with the same name."""
    sel = f"involvedObject.name={name}" + (f",involvedObject.uid={uid}" if uid else "")
    try:
        evs = client.CoreV1Api().list_namespaced_event(ns, field_selector=sel).items
    except ApiException:
        return []
    evs.sort(key=lambda e: e.last_timestamp or e.event_time or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc))
    return [f"{e.reason}: {e.message}" for e in evs[-limit:] if e.type == "Warning"]


def quota(ns: str | None = None) -> list[tuple[str, str, str]]:
    _load_config()
    ns = ns or current_namespace()
    rows = []
    for q in client.CoreV1Api().list_namespaced_resource_quota(ns).items:
        for k, hard in (q.status.hard or {}).items():
            rows.append((k, (q.status.used or {}).get(k, "0"), hard))
    return rows
