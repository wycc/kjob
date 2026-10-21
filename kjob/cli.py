"""kjob command line: Slurm-like commands for PyTorchJobs that mirror the current notebook.

Submitting and following your jobs
  kjob submit  [opts] SCRIPT [ARGS...]   like sbatch: submit and return (job deleted by TTL after it finishes)
  kjob run     [opts] SCRIPT [ARGS...]   like srun:   submit, stream output live, delete the job, exit with its code
  kjob wait    NAME                       follow a submitted job until it ends, then delete it
  kjob queue   [-A] [-w]                  like squeue: your jobs (-A: every GPU workload in the cluster)
  kjob show    NAME                       like scontrol show job: resources, state, node, events, log file
  kjob logs    NAME [-f]                  print (or follow) a job's output; falls back to the saved log file
  kjob exec    NAME [CMD...]              run a command in a running job, or open a shell (srun --jobid --pty)
  kjob cancel  NAME... | --finished | --all-mine [--all-notebooks]   like scancel
  kjob clean   [--all-notebooks]          delete finished kjob jobs (same as cancel --finished)
  kjob acct    [-n N]                     like sacct: jobs you submitted, with exit code and run time
Cluster and environment
  kjob sinfo   [-g TYPE] [-w]             like sinfo: GPUs per type and node (total / used / free / waiting)
  kjob info                               what jobs inherit from this notebook, quota usage
  kjob install-skills [--dir DIR]         reinstall the AI-assistant skills (automatic: ~/.agents/skills)
Short aliases (installed by pip, also in bin/): ksub, ksrun, kwait, kqueue, kshow, klogs, kexec, kcancel, kclean, kacct,
ksinfo, kinfo. Scripts may carry options in '#KJOB' or '#SBATCH' lines at the top; command-line options win.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from kubernetes.client.rest import ApiException
from kubernetes.config import ConfigException
from urllib3.exceptions import HTTPError

from . import cluster, core

SKILLS_DIR = Path(__file__).parent / "skills"


# ---------- option parsing ----------
def parse_time(value: str) -> int:
    """Slurm time formats (MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS) or 90m / 12h / 2d, in seconds."""
    v = value.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([smhd])", v)
    if m:
        total = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        if total <= 0:
            raise ValueError
        return total
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        days = int(d)
        parts = [int(x) for x in v.split(":")]
        parts += [0] * (3 - len(parts))        # D-HH, D-HH:MM, D-HH:MM:SS
        h, mi, s = parts
    else:
        parts = [int(x) for x in v.split(":")]
        if len(parts) == 1:
            h, mi, s = 0, parts[0], 0           # minutes
        elif len(parts) == 2:
            h, mi, s = 0, parts[0], parts[1]    # minutes:seconds
        else:
            h, mi, s = parts
    total = days * 86400 + h * 3600 + mi * 60 + s
    if total <= 0:
        raise ValueError
    return total


def _time_arg(value: str) -> int:
    try:
        return parse_time(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"invalid time {value!r} (examples: 30, 2:00:00, 1-12:00:00, 90m, 12h)")


def _nonneg(v: str) -> int:
    if not v.isdigit():
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {v!r}")
    return int(v)


def _slurm_mem(v: str) -> str:
    """Slurm sizes are binary: 32G -> 32Gi; a bare number is MiB."""
    m = re.fullmatch(r"(\d+)([KMGT]?)B?", v.upper())
    if not m:
        return v
    return m.group(1) + ({"": "M"}.get(m.group(2), m.group(2))) + "i"


_SBATCH_SAME = {"-J": "-J", "--job-name": "-J", "-c": "-c", "--cpus-per-task": "-c", "-t": "-t", "--time": "-t",
                "-w": "-w", "--nodelist": "-w", "-x": "-x", "--exclude": "-x", "-D": "-D", "--chdir": "-D"}


def _translate_sbatch(tokens: list[str], warn) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"-[A-Za-z]\S+", tok):            # attached short form: -c4, -Jname
            tokens = tokens[:i] + [tok[:2], tok[2:]] + tokens[i + 1:]
            tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            key, val = tok.split("=", 1)
        elif (tok in _SBATCH_SAME or tok in ("--mem", "--gres", "--gpus", "-G", "--gpus-per-node")) and i + 1 < len(tokens):
            key, val = tok, tokens[i + 1]
            i += 1
        elif tok.startswith("-") and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            key, val = tok, tokens[i + 1]      # unknown option with a value, e.g. -p gp1d
            i += 1
        else:
            key, val = tok, None
        i += 1
        if key in ("-t", "--time") and val and val.upper() in ("UNLIMITED", "INFINITE"):
            continue                           # no limit is the kjob default
        if key in _SBATCH_SAME and val is not None:
            out += [_SBATCH_SAME[key], val]
        elif key == "--mem" and val and re.fullmatch(r"0+[KMGT]?B?", val.upper()):
            warn("kjob: ignoring #SBATCH --mem=0 (all node memory); set --mem explicitly")
        elif key == "--mem" and val:
            out += ["--mem", _slurm_mem(val)]
        elif key in ("--gpus", "-G", "--gpus-per-node") and val:
            parts = val.split(":")
            if len(parts) == 2:
                out += ["-g", parts[0]]
            out += ["-G", parts[-1]]
        elif key == "--gres" and val and val.startswith("gpu"):
            parts = val.split(":")          # gpu:N or gpu:TYPE:N
            if len(parts) == 3:
                out += ["-g", parts[1]]
            out += ["-G", parts[-1] if len(parts) > 1 else "1"]
        else:
            warn(f"kjob: ignoring #SBATCH {tok}" + (f" {val}" if val is not None and "=" not in tok else ""))
    return out


def script_directives(script: str, warn=lambda m: print(m, file=sys.stderr)) -> list[str]:
    """Options from '#KJOB ...' (kjob syntax) and '#SBATCH ...' lines before the first command of the script."""
    try:
        with open(script, errors="replace") as f:
            lines = [next(f, "") for _ in range(200)]
    except OSError:
        return []
    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#!"):
            continue
        if not s.startswith("#"):
            break                              # like sbatch: directives end at the first command
        m = re.match(r"#\s*(KJOB|SBATCH)\s+(.*)", s)
        if not m:
            continue
        tokens = shlex.split(m.group(2), comments=True)
        out += tokens if m.group(1) == "KJOB" else _translate_sbatch(tokens, warn)
    return out


def _add_job_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("-J", "--job-name", help="job name (default: script name + unique suffix)")
    p.add_argument("-G", "--gpus", type=int, default=1, help="GPUs (default 1; 0 for CPU-only)")
    p.add_argument("-c", "--cpus", type=int, default=core.DEFAULT_CPUS, help="CPU cores (default 1)")
    p.add_argument("--mem", default=core.DEFAULT_MEMORY, help="memory, e.g. 32Gi (default 2Gi)")
    p.add_argument("--shm", default=core.DEFAULT_SHM, help="/dev/shm size, e.g. 8Gi (default 256Mi)")
    p.add_argument("-g", "--gpu-type", default="", help=f"comma list of GPU types: {', '.join(core.GPU_PRODUCTS)}, a "
                   "name shown by ksinfo, or a raw nvidia.com/gpu.product value")
    p.add_argument("-w", "--nodelist", default=None, help="run on this node (kubernetes.io/hostname)")
    p.add_argument("-x", "--exclude", default="", help="comma list of nodes to avoid")
    p.add_argument("-t", "--time", type=_time_arg, default=None,
                   help="time limit; the job is stopped after it (30, 2:00:00, 1-12:00:00, 12h)")
    p.add_argument("--python", default=None, help="interpreter for the script (default: this Python for .py)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment variable (repeatable)")
    p.add_argument("-D", "--chdir", default=None, help="working directory in the job (default: current directory)")
    p.add_argument("-o", "--log-dir", default=str(Path.home() / "kjob-logs"), help="where the job's output is saved")
    p.add_argument("--ttl", type=int, default=3600, help="delete a finished job after this many seconds if unmonitored")
    p.add_argument("script", help="script to run (.py with Python, .sh with bash, anything else executed directly)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def resolve_gpu_types(values: list[str], node: str | None, snapshot=None) -> tuple[list[str], str | None]:
    """Map -g values to nvidia.com/gpu.product labels using the GPU types ksinfo shows.

    Exact type or host names win; otherwise a part of a name must match exactly one type ("a5000", "V100-SXM2").
    A host name, or a type that exists only on an unlabeled node, becomes a node pin (-w); a host name that is also
    part of that host's own labeled type (e.g. rtxpro6000) selects the type, so it can be combined with others."""
    out = []
    snap = snapshot
    for v in values:
        alias = v.lower() if v.lower() in core.GPU_PRODUCTS else None
        label_like = v.split("-")[0] in ("NVIDIA", "Tesla", "Quadro")
        if snap is None:
            try:
                snap = cluster.load_snapshot()
            except (cluster.ClusterViewUnavailable, ApiException, HTTPError, ValueError):
                snap = {}
                if alias or label_like:
                    out.append(alias or v)  # nothing to check it against
                    continue
                raise ValueError(f"Unknown GPU type {v!r}: use a short name ({', '.join(core.GPU_PRODUCTS)}), a full "
                                 "nvidia.com/gpu.product value, or -w NODE (the cluster GPU list is not available "
                                 "to check other names).") from None
        nodes = snap.get("nodes", [])
        if not nodes and (alias or label_like):     # cluster view unavailable (checked above for this value)
            out.append(alias or v)
            continue
        if not nodes:
            raise ValueError(f"Unknown GPU type {v!r}: use a short name ({', '.join(core.GPU_PRODUCTS)}), a full "
                             "nvidia.com/gpu.product value, or -w NODE (the cluster GPU list is not available "
                             "to check other names).")
        if alias:
            if not any(n["product"] == core.GPU_PRODUCTS[alias] for n in nodes):
                raise ValueError(f"GPU type {v!r} ({core.GPU_PRODUCTS[alias]}) is not in this cluster; see ksinfo.")
            out.append(alias)
            continue
        types = {}
        for n in nodes:
            types.setdefault(cluster.node_type(n), []).append(n)
        q = _norm(v)
        host = [n for n in nodes if _norm(n["name"]) == q]
        exact = [t for t in types if _norm(t) == q]
        partial = [t for t in types if q and q in _norm(t)]
        if host and not exact and cluster.node_type(host[0]) in partial and host[0]["product"]:
            host = []                           # "rtxpro6000" names the host and its labeled type: use the type
        if host and not exact:
            if len(values) > 1 or node not in (None, host[0]["name"]):
                raise ValueError(f"{v!r} is a node name; use it alone or with -w {host[0]['name']}.")
            node = host[0]["name"]
            continue
        hits = exact or partial
        if not hits:
            raise ValueError(f"GPU type {v!r} matches nothing in this cluster; see ksinfo for the names.")
        if len(hits) > 1:
            raise ValueError(f"GPU type {v!r} matches several types ({', '.join(sorted(hits))}); be more specific "
                             "or list the ones you want with commas.")
        t = hits[0]
        if types[t][0]["product"]:
            out.append(types[t][0]["product"])
        elif len(types[t]) == 1 and len(values) == 1 and node in (None, types[t][0]["name"]):
            node = types[t][0]["name"]          # unlabeled GPU node: select it by host name
        else:
            raise ValueError(f"{t} has no GPU product label; select it alone with -w {types[t][0]['name']}.")
    if node and out and snap and snap.get("nodes"):
        pinned = next((n for n in snap["nodes"] if n["name"] == node), None)
        wanted = {core.GPU_PRODUCTS.get(g, g) for g in out}
        if pinned and pinned["product"] not in wanted:
            raise ValueError(f"Node {node} has {cluster.node_type(pinned)}, not {', '.join(sorted(wanted))}; "
                             "the job could never start.")
    return out, node


def _check_node(node: str | None, gpus: int) -> None:
    """Reject a -w host that is not a GPU node, when the cluster view can tell."""
    if not node or not gpus:
        return
    try:
        names = {n["name"] for n in cluster.load_snapshot().get("nodes", [])}
    except (cluster.ClusterViewUnavailable, ApiException, HTTPError, ValueError):
        return
    if names and node not in names:
        raise ValueError(f"{node} is not a GPU node in this cluster; GPU nodes: {', '.join(sorted(names))}.")


def _spec(a) -> core.JobSpec:
    name = a.job_name or core.unique_name(Path(a.script).stem)
    if any("=" not in e for e in a.env):
        raise ValueError("--env must be KEY=VALUE.")
    env = dict(e.split("=", 1) for e in a.env)
    if a.gpus == 0 and a.gpu_type:
        raise ValueError("CPU-only jobs (-G 0) cannot select a GPU type (-g).")
    gpu_types, node = resolve_gpu_types([g for g in a.gpu_type.split(",") if g], a.nodelist)
    if a.nodelist and a.gpus:
        _check_node(node, a.gpus)
    spec = core.JobSpec(name=name, script=a.script, args=a.args, gpus=a.gpus, cpus=a.cpus, memory=a.mem, shm=a.shm,
                        gpu_types=gpu_types, node=node,
                        exclude_nodes=[n for n in a.exclude.split(",") if n], interpreter=a.python, env=env,
                        workdir=str(Path(a.chdir or os.getcwd()).expanduser().resolve()),
                        log_dir=str(Path(a.log_dir).expanduser().resolve()), ttl_seconds=a.ttl, time_limit=a.time)
    spec.validate()
    return spec


# ---------- output helpers ----------
def _age(ts: str | None) -> str:
    if not ts:
        return "-"
    s = int((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds())
    return _dur(s)


def _dur(s: int | None) -> str:
    if s is None:
        return "-"
    return f"{s // 86400}d{s % 86400 // 3600}h" if s >= 86400 else f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _table(hdr: list[str], rows: list[list]) -> None:
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)).rstrip())
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _snapshot_banner(snap: dict) -> str:
    if snap.get("source") == "live":
        return "cluster status (live)"
    age = cluster.snapshot_age(snap)
    stale = "  WARNING: old data, the status service may be down" if age > 300 else ""
    return f"cluster status (published {_dur(age)} ago){stale}"


def _job_name(name: str) -> str:
    """The name kjob gave a job submitted with -J NAME (e.g. 001 -> j-001)."""
    try:
        return core.sanitize_name(name)
    except ValueError:
        return name


def _job_for(name: str) -> core.KJob:
    """KJob handle for an existing job (only the name is needed for monitoring)."""
    j = core.KJob.__new__(core.KJob)
    core._load_config()
    name = _job_name(name)
    j.spec = core.JobSpec(name=name, script="")
    j.ns = core.current_namespace()
    j.core, j.custom = core.client.CoreV1Api(), core.client.CustomObjectsApi()
    j.pod_name = f"{name}-master-0"
    return j


# ---------- job commands ----------
def cmd_submit(a) -> int:
    job = core.KJob(_spec(a))
    name = job.submit()
    print(f"Submitted job {name}  (log: {job.spec.log_path()})")
    return 0


def cmd_run(a) -> int:
    code = core.KJob(_spec(a)).run()
    if code == 141:                          # our output pipe was closed (ksrun ... | head): exit quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return code


def cmd_wait(a) -> int:
    j = _job_for(a.name)
    a.name = j.spec.name
    report = lambda m: print(m, file=sys.stderr, flush=True)
    if j.state() == "Deleted":
        report(f"kjob: no job {a.name} in namespace {j.ns} (already deleted? see kacct / klogs {a.name})")
        return 1
    try:
        return j.follow(report=report)
    except KeyboardInterrupt:
        report("kjob: stopped following (job keeps running)")
        return 130
    finally:
        if not a.keep and (j.state() in core.TERMINAL or getattr(j, "invalid", False)):
            j.delete()


def _queue_rows(everyone: bool):
    pods = {p.metadata.name: p for p in core.client.CoreV1Api().list_namespaced_pod(core.current_namespace()).items}
    rows = []
    for o in core.list_jobs(everyone=everyone):
        name = o["metadata"]["name"]
        pod = pods.get(f"{name}-master-0")
        tmpl = o["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
        res = (tmpl.get("resources") or {}).get("requests", {})
        rows.append([name, core.job_state(o), (o["metadata"].get("labels") or {}).get("kjob/notebook", "-"),
                     pod.status.phase if pod else "-", pod.spec.node_name if pod and pod.spec.node_name else "-",
                     res.get("nvidia.com/gpu", "0"), res.get("cpu", "-"), _age(o["metadata"].get("creationTimestamp")),
                     (o["metadata"].get("annotations") or {}).get("kjob/command", "")[-60:]])
    return rows


def _cluster_queue(a) -> None:
    snap = cluster.load_snapshot()
    nodes = {n["name"]: n for n in snap.get("nodes", [])}
    mine = core.current_namespace()
    rows = []
    for w in snap.get("workloads", []):
        gtype = cluster.workload_type(w, nodes)
        if a.user and w["namespace"] not in a.user.split(","):
            continue
        if a.state and w["state"].lower() != a.state.lower():
            continue
        if a.gpu_type and not any(cluster.matches_type(a.gpu_type, n) for n in nodes.values()
                                  if cluster.node_type(n) in gtype.split(",")):
            continue
        rows.append({**w, "type": gtype})
    rows.sort(key=lambda w: (w["state"] != "Running", w["type"], -w["age_s"]))
    print(_snapshot_banner(snap))
    table = [[("*" if w["namespace"] == mine else "") + w["namespace"], w["kind"], w["name"], w["state"], w["gpus"],
              w["type"], w["node"] or "-", _dur(w["age_s"]), w["reason"][:70]] for w in rows]
    _table(["USER", "KIND", "NAME", "STATE", "GPU", "GPU_TYPE", "NODE", "TIME", "WAITING_FOR"], table)
    run = [w for w in rows if w["state"] == "Running"]
    pend = [w for w in rows if w["state"] == "Pending"]
    print(f"running: {len(run)} workloads / {sum(w['gpus'] for w in run)} GPUs;  "
          f"pending: {len(pend)} workloads / {sum(w['gpus'] for w in pend)} GPUs   (* = your namespace)")


def cmd_queue(a) -> int:
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")
        if a.all_users or a.user:
            _cluster_queue(a)
        else:
            _table(["NAME", "STATE", "NOTEBOOK", "POD", "NODE", "GPU", "CPU", "AGE", "COMMAND"], _queue_rows(a.all))
        if not a.watch:
            return 0
        time.sleep(5)


def cmd_show(a) -> int:
    j = _job_for(a.name)
    a.name = j.spec.name
    try:
        o = j.custom.get_namespaced_custom_object(core.GROUP, core.VERSION, j.ns, core.PLURAL, a.name)
    except ApiException as e:
        if e.status != 404:
            raise
        rec = next((r for r in reversed(core.read_history()) if r["name"] == a.name), None)
        if not rec:
            print(f"kjob: no job {a.name} in namespace {j.ns} and no history record", file=sys.stderr)
            return 1
        start, end, code = core.log_result(rec["log"])
        print(f"JobName={a.name}  (deleted from the cluster; from ~/.kjob/history.jsonl)")
        for k in ("submitted", "gpus", "gpu_types", "node", "cpus", "memory", "time_limit", "command", "log"):
            print(f"  {k:12s} {rec.get(k)}")
        print(f"  {'started':12s} {start or '-'}\n  {'ended':12s} {end or '-'}\n  {'exit code':12s} {code if code is not None else '-'}")
        return 0
    meta, spec = o["metadata"], o["spec"]
    ann = meta.get("annotations") or {}
    pod_spec = spec["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
    c = pod_spec["containers"][0]
    res = (c.get("resources") or {}).get("requests", {})
    terms = ((((pod_spec.get("affinity") or {}).get("nodeAffinity") or {})
              .get("requiredDuringSchedulingIgnoredDuringExecution") or {}).get("nodeSelectorTerms") or [{}])[0]
    rp = spec.get("runPolicy") or {}
    print(f"JobName={a.name}  Namespace={j.ns}  State={core.job_state(o)}")
    print(f"  submitted    {ann.get('kjob/submitted', meta.get('creationTimestamp'))}  (age {_age(meta.get('creationTimestamp'))})")
    print(f"  notebook     {(meta.get('labels') or {}).get('kjob/notebook', '-')}")
    print(f"  command      {ann.get('kjob/command', '-')}")
    print(f"  workdir      {ann.get('kjob/workdir', '-')}")
    print(f"  log          {ann.get('kjob/log', '-')}")
    print(f"  resources    gpu={res.get('nvidia.com/gpu', '0')} cpu={res.get('cpu')} mem={res.get('memory')} image={c.get('image')}")
    for e in terms.get("matchExpressions", []):
        print(f"  constraint   {e['key']} {e['operator']} {','.join(e.get('values', []))}")
    limit = pod_spec.get("activeDeadlineSeconds")
    print(f"  time limit   {_dur(limit) if limit else 'none'}"
          f"   kept after end {str(rp['ttlSecondsAfterFinished']) + 's' if 'ttlSecondsAfterFinished' in rp else '-'}")
    for cond in (o.get("status") or {}).get("conditions") or []:
        print(f"  condition    {cond['type']}={cond['status']} {cond.get('reason', '')} {cond.get('message', '')[:120]}")
    try:
        pod = j._pod()
        reason = f", {pod.status.reason}" if pod.status.reason else ""     # e.g. DeadlineExceeded at the time limit
        print(f"  pod          {pod.status.phase}{reason} on {pod.spec.node_name or '-'} (started {pod.status.start_time or '-'})")
        code = j.exit_code()
        if code is not None:
            print(f"  exit code    {code} ({getattr(j, 'exit_reason', '')})")
    except ApiException as e:
        if e.status != 404:
            raise
        print("  pod          not created yet or already removed")
    for ev in core.job_events(j.ns, a.name, uid=meta.get("uid")) + core.job_events(j.ns, j.pod_name):
        print(f"  event        {ev[:160]}")
    return 0


def cmd_exec(a) -> int:
    j = _job_for(a.name)
    try:
        j._pod()
    except ApiException as e:
        if e.status != 404:
            raise
        raise ValueError(f"{j.spec.name} has no running pod (not started yet, or finished); see kshow {j.spec.name}.") from None
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    return j.exec(command)


def cmd_logs(a) -> int:
    j = _job_for(a.name)
    a.name = j.spec.name
    state = j.state()
    report = lambda m: print(m, file=sys.stderr, flush=True)
    try:
        if state == "Deleted":
            raise ApiException(status=404)
        if a.follow:
            if j.wait_for_pod(report=report) in ("Deleted", "Invalid"):   # still pending: wait instead of failing
                raise ApiException(status=404)
            for line in j.stream_logs():
                print(line, flush=True)
        else:
            pod = j._pod()
            if pod.status.phase == "Pending":
                report(f"kjob: {a.name} has not started yet (Pending); kshow {a.name} shows why, klogs -f waits")
                return 0
            print(j.core.read_namespaced_pod_log(j.pod_name, j.ns), end="")
        return 0
    except KeyboardInterrupt:
        return 130
    except ApiException as e:
        if e.status == 400:
            report(f"kjob: {a.name} has not started yet; try klogs -f {a.name}")
            return 0
        if e.status != 404:
            raise
    try:
        obj = j.custom.get_namespaced_custom_object(core.GROUP, core.VERSION, j.ns, core.PLURAL, a.name)
        log = (obj["metadata"].get("annotations") or {}).get("kjob/log")
    except ApiException:
        log = None
    rec = next((r for r in reversed(core.read_history()) if r["name"] == a.name), None)
    log = log or (rec or {}).get("log") or str(Path(a.log_dir) / f"{a.name}.log")
    if os.path.exists(log):
        print(f"kjob: pod not available; showing {log}", file=sys.stderr)
        print(open(log).read(), end="")
        return 0
    print(f"kjob: no pod and no log file for {a.name}", file=sys.stderr)
    return 1


def cmd_cancel(a) -> int:
    ns = core.current_namespace()
    if not (a.names or a.finished or a.all_mine):
        raise ValueError("give job names, --finished or --all-mine")
    a.names = [_job_name(n) for n in a.names]
    targets = {n: None for n in a.names}
    if a.finished or a.all_mine:
        notebook = None if a.all_notebooks else core.current_notebook()
        for o in core.list_jobs(notebook=notebook):
            if a.all_mine or core.job_state(o) in core.TERMINAL:
                targets[o["metadata"]["name"]] = o["metadata"].get("uid")
    missing = 0
    for n, uid in targets.items():
        deleted = core.delete_job(ns, n, uid=uid)
        print(f"{'deleted' if deleted else 'not found or replaced'}: {n}")
        missing += not deleted and n in a.names
    return 1 if missing else 0


def _ts(x: str) -> dt.datetime:
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))


def cmd_acct(a) -> int:
    hist = core.read_history()
    try:
        live = {o["metadata"]["name"]: core.job_state(o) for o in core.list_jobs()}
    except (ApiException, HTTPError, ConfigException):
        live = {}              # history still works without cluster access
    by_log = {}
    for r in hist:
        by_log.setdefault(r.get("log", ""), []).append(r)
    run_of = {}                 # id(record) -> run: a run belongs to the latest submission made before it started
    last_run = {}
    for log, recs in by_log.items():
        runs = core.log_runs(log)
        last_run[log] = runs[-1] if runs else None
        for run in runs:
            if not run["start"]:
                continue
            start = _ts(run["start"]) + dt.timedelta(seconds=60)      # allow some clock skew between nodes
            before = [x for x in recs if _ts(x["submitted"]) <= start]
            if before:
                run_of[id(before[-1])] = run
    rows = []
    for r in hist:
        log = r.get("log", "")
        run = run_of.get(id(r))
        last = r is by_log[log][-1]
        code = run and run["code"]
        if run and run["stopped"]:
            state = "STOPPED"      # time limit or kcancel
        elif code is not None:
            state = "COMPLETED" if code == 0 else "FAILED"
        elif last and r["name"] in live:
            state = {"Created": "PENDING"}.get(live[r["name"]], live[r["name"]].upper())
        else:
            state = "ENDED?"       # deleted before it finished, or never started
        elapsed = None
        if run and run["start"]:
            end = run["end"]
            if not end and run is last_run[log] and not (last and r["name"] in live):   # no end line: last write
                try:
                    end = dt.datetime.fromtimestamp(os.path.getmtime(log), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                except OSError:
                    end = None
            if end or (last and r["name"] in live):
                t1 = _ts(end) if end else dt.datetime.now(dt.timezone.utc)
                elapsed = max(int((t1 - _ts(run["start"])).total_seconds()), 0)
        where = ",".join(r.get("gpu_types") or []) or r.get("node") or ("cpu" if not r.get("gpus") else "any")
        rows.append([r["name"], r["submitted"].replace("T", " ")[:16], r.get("gpus", 0), where, state,
                     code if code is not None else "-", _dur(elapsed)])
    _table(["NAME", "SUBMITTED(UTC)", "GPU", "GPU_TYPE/NODE", "STATE", "EXIT", "ELAPSED"], rows[-a.n:] if a.n else rows)
    return 0


# ---------- cluster / environment ----------
def _my_tolerations() -> list | None:
    """Tolerations a job from this notebook gets from the namespace PodDefaults (None when unknown)."""
    try:
        ns = core.current_namespace()
        api = core.client.CoreV1Api()
        labels = api.read_namespaced_pod(core.current_notebook(), ns).metadata.labels or {}
        pds = core.client.CustomObjectsApi().list_namespaced_custom_object("kubeflow.org", "v1alpha1", ns, "poddefaults")["items"]
    except (ApiException, HTTPError, ValueError):
        return None
    return [t for pd in pds if core._matches(pd["spec"].get("selector", {}), labels)
            for t in pd["spec"].get("tolerations", [])]


def _tolerated(taint: str, tolerations: list) -> bool:
    key, _, value = taint.partition("=")
    for t in tolerations:
        if t.get("operator") == "Exists" and not t.get("key"):
            return True                        # tolerates every taint
        if t.get("key") == key and (t.get("operator") == "Exists" or t.get("value", "") == value):
            return True
    return False


def cmd_sinfo(a) -> int:
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")
        _sinfo(a)
        if not a.watch:
            return 0
        time.sleep(10)


def _sinfo(a) -> None:
    snap = cluster.load_snapshot()
    nodes = [n for n in snap.get("nodes", []) if not a.gpu_type or cluster.matches_type(a.gpu_type, n)]
    by_name = {n["name"]: n for n in snap.get("nodes", [])}
    tol = _my_tolerations()
    types = {}
    for n in nodes:
        t = types.setdefault(cluster.node_type(n), {"nodes": 0, "gpus": 0, "used": 0, "free": 0, "mem": n["memory_mib"],
                                                     "restricted": set(), "pend": 0, "pend_jobs": 0})
        t["nodes"] += 1
        t["gpus"] += n["gpus"]
        t["used"] += n["gpus_used"]
        if n["ready"] and n["schedulable"]:
            t["free"] += max(n["gpus"] - n["gpus_used"], 0)
        t["restricted"] |= set(n["restricted"])
    for w in snap.get("workloads", []):
        if w["state"] == "Pending" and w["gpus"]:
            for tname in cluster.workload_type(w, by_name).split(","):
                if tname in types:
                    types[tname]["pend"] += w["gpus"]
                    types[tname]["pend_jobs"] += 1
    print(_snapshot_banner(snap))
    rows = []
    for name, t in sorted(types.items()):
        note = ""
        if t["restricted"]:
            ok = None if tol is None else all(_tolerated(x, tol) for x in t["restricted"])
            note = "restricted" + ("" if ok is None else " (your namespace: allowed)" if ok else " (your namespace: no access)")
        rows.append([name, t["nodes"], t["gpus"], t["used"], t["free"], f"{t['pend']} ({t['pend_jobs']} jobs)" if t["pend"] else "-",
                     f"{round(t['mem'] / 1024)}GB" if t["mem"] else "-", note])
    _table(["GPU_TYPE", "NODES", "GPUS", "USED", "FREE", "WAITING_GPUS", "GPU_MEM", "NOTE"], rows)
    print()
    rows = [[n["name"], cluster.node_type(n), f"{n['gpus_used']}/{n['gpus']}", f"{_num(n['cpus_used'])}/{_num(n['cpus'])}",
             f"{_num(n['memory_used_gib'])}/{_num(n['memory_gib'])}",
             ("up" if n["ready"] else "down") + ("" if n["schedulable"] else ",drain")] for n in nodes]
    _table(["NODE", "GPU_TYPE", "GPU(used/total)", "CPU(req/total)", "MEM_GiB(req/total)", "STATE"], rows)
    pend_any = [w for w in snap.get("workloads", []) if w["state"] == "Pending" and cluster.workload_type(w, by_name) == "any"]
    if pend_any and not a.gpu_type:
        print(f"\nwaiting without a GPU type: {len(pend_any)} workloads / {sum(w['gpus'] for w in pend_any)} GPUs")
    if a.gpu_type:
        print()
        _cluster_queue(argparse.Namespace(user=None, state=None, gpu_type=a.gpu_type))


def cmd_info(a) -> int:
    t = core.NotebookTemplate.from_current_pod()
    print(f"notebook  {t.pod_name}  (namespace {t.namespace}, node {t.node})")
    print(f"image     {t.image}")
    print(f"workdir   {t.working_dir}  (jobs start here unless -D is given)")
    print("volumes inherited by jobs:")
    for v, m in zip(t.volumes, t.volume_mounts):
        kind = f"nfs {v.nfs.server}:{v.nfs.path}" if v.nfs else ("emptyDir" + (" (memory)" if v.empty_dir and v.empty_dir.medium == "Memory" else "")) if v.empty_dir else \
            f"pvc {v.persistent_volume_claim.claim_name}" if v.persistent_volume_claim else "other"
        print(f"  {m.mount_path:24s} {kind}")
    for n, why in t.skipped.items():
        if why != "not mounted in the notebook container":     # sidecar volumes: nothing the user mounted
            print(f"  (not copied: {n} - {why})")
    print("quota (used / hard):")
    for k, used, hard in core.quota():
        print(f"  {k:28s} {used} / {hard}")
    print("GPU type short names for -g:", ", ".join(f"{k}={v}" for k, v in core.GPU_PRODUCTS.items()))
    print("cluster GPU types and free GPUs: ksinfo")
    return 0


# The skills live in ~/.agents/skills, the shared Agent Skills location read by Codex, OpenCode, Gemini CLI, Cursor and
# others. Claude Code reads only ~/.claude/skills, so that gets a symlink per skill instead of a second copy.
SKILLS_HOME = "~/.agents/skills"
CLAUDE_SKILLS = "~/.claude/skills"
SKILL_MARKER = ".kjob-installed"       # checksum of the SKILL.md kjob wrote; a copy the user edited is left alone


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def install_skills(dest: Path, force: bool = False) -> list[Path]:
    """Copy the bundled skills into dest and return the files written. Unless forced, keep files the user edited."""
    written = []
    for src in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        target = dest / src.parent.name / "SKILL.md"
        marker = target.parent / SKILL_MARKER
        current = _sha(target)
        if current == _sha(src):
            continue
        if current is not None and not force:
            try:
                recorded = marker.read_text().strip()
            except OSError:
                recorded = None
            if current != recorded:
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        marker.write_text(_sha(target) + "\n")
        written.append(target)
    return written


def link_claude_skills(home: Path, claude: Path) -> list[Path]:
    """Point ~/.claude/skills/<skill> at ~/.agents/skills/<skill>; replace an unedited copy an older kjob made."""
    linked = []
    for src in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        name = src.parent.name
        target, link = home / name, claude / name
        if link.is_symlink():
            if Path(os.readlink(link)) == target:
                continue
            link.unlink()
        elif link.exists():
            copy = link / "SKILL.md"
            try:
                recorded = (link / SKILL_MARKER).read_text().strip()
            except OSError:
                recorded = None
            if _sha(copy) != recorded:
                continue                       # not ours, or edited by the user: keep it
            shutil.rmtree(link)
        claude.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        linked.append(link)
    return linked


def setup_skills(force: bool = False) -> list[str]:
    """Install the skills into ~/.agents/skills and link them from ~/.claude/skills; return what changed."""
    home, claude = Path.home() / SKILLS_HOME[2:], Path.home() / CLAUDE_SKILLS[2:]
    changed = [str(p) for p in install_skills(home, force=force)]
    changed += [f"{p} -> {home / p.name}" for p in link_claude_skills(home, claude)]
    return changed


def auto_install_skills() -> None:
    """Keep the kjob skills installed and current for AI assistants (pip has no post-install step)."""
    if os.environ.get("KJOB_NO_SKILLS"):
        return
    try:
        changed = setup_skills()
    except OSError:
        return
    if changed:
        print("kjob: installed AI skills (kjob-install, kjob-jobs) in ~/.agents/skills, linked from ~/.claude/skills; "
              "set KJOB_NO_SKILLS=1 to turn this off", file=sys.stderr)


def cmd_install_skills(a) -> int:
    changed = [str(p) for p in install_skills(Path(a.dir).expanduser(), force=True)] if a.dir else setup_skills(force=True)
    for c in changed:
        print(f"installed {c}")
    if not changed:
        print("kjob skills are up to date")
    return 0


def cmd_publish_status(a) -> int:
    if a.once:
        cluster.publish(cluster.build_snapshot(hide_names=a.hide_names))
        return 0
    cluster.publish_loop(a.interval, hide_names=a.hide_names, report=lambda m: print(m, flush=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kjob", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("submit", cmd_submit, "submit and return (sbatch)"), ("run", cmd_run, "submit, stream output, clean up (srun)")]:
        p = sub.add_parser(name, help=help_)
        _add_job_opts(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("wait", help="follow a job until it ends, then delete it")
    p.add_argument("name")
    p.add_argument("--keep", action="store_true", help="do not delete the job when it ends")
    p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("queue", help="list your jobs, or every GPU workload with -A (squeue)")
    p.add_argument("-A", "--all-users", action="store_true", help="every GPU workload in the cluster, all users")
    p.add_argument("-u", "--user", default=None, help="with -A: only these namespaces (comma list); implies -A")
    p.add_argument("--state", choices=["running", "pending"], help="with -A: only running or only pending")
    p.add_argument("-g", "--gpu-type", default=None, help="with -A: only this GPU type / node (e.g. v100, rtxpro6000)")
    p.add_argument("--all", action="store_true", help="also list PyTorchJobs in your namespace not created by kjob")
    p.add_argument("-w", "--watch", action="store_true", help="refresh every 5 s")
    p.set_defaults(fn=cmd_queue)
    p = sub.add_parser("show", help="details of one job (scontrol show job)")
    p.add_argument("name")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("exec", help="run a command (or a shell) inside a running job (srun --jobid --pty)")
    p.add_argument("name")
    p.add_argument("command", nargs=argparse.REMAINDER, help="command to run; none opens an interactive bash")
    p.set_defaults(fn=cmd_exec)
    p = sub.add_parser("logs", help="print or follow a job's output")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-o", "--log-dir", default=str(Path.home() / "kjob-logs"))
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("cancel", help="delete jobs (scancel)")
    p.add_argument("names", nargs="*")
    p.add_argument("--finished", action="store_true", help="delete all finished kjob jobs")
    p.add_argument("--all-mine", action="store_true", help="delete every kjob job submitted from this notebook")
    p.add_argument("--all-notebooks", action="store_true", help="--finished / --all-mine: include jobs from other notebooks")
    p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("clean", help="delete finished kjob jobs submitted from this notebook")
    p.add_argument("--all-notebooks", action="store_true", help="include jobs from other notebooks")
    p.set_defaults(fn=lambda a: cmd_cancel(argparse.Namespace(names=[], finished=True, all_mine=False,
                                                              all_notebooks=a.all_notebooks)))
    p = sub.add_parser("acct", help="jobs you submitted, with exit code and run time (sacct)")
    p.add_argument("-n", type=_nonneg, default=20, help="last N jobs (default 20, 0 = all)")
    p.set_defaults(fn=cmd_acct)
    p = sub.add_parser("sinfo", help="GPU types and nodes: total / used / free / waiting (sinfo)")
    p.add_argument("-g", "--gpu-type", default=None, help="only this GPU type or node, and list its workloads")
    p.add_argument("-w", "--watch", action="store_true", help="refresh every 10 s")
    p.set_defaults(fn=cmd_sinfo)
    p = sub.add_parser("info", help="what jobs inherit from this notebook, quota usage")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("install-skills", help="reinstall the kjob skills for AI assistants (done automatically: "
                       "~/.agents/skills, linked from ~/.claude/skills)")
    p.add_argument("--dir", default=None, help="copy them into another skills directory instead")
    p.set_defaults(fn=cmd_install_skills)
    p = sub.add_parser("publish-status", help="(administrators) publish the cluster status ConfigMap")
    for sp in sub.choices.values():
        sp.set_defaults(parser=sp)
    p.add_argument("--interval", type=int, default=30, help="seconds between updates")
    p.add_argument("--once", action="store_true", help="publish once and exit")
    p.add_argument("--hide-names", action="store_true", help="publish namespaces but not workload names")
    p.set_defaults(fn=cmd_publish_status)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    err = a.parser.error
    if a.cmd in ("submit", "run"):
        cli_only = a
        extra = script_directives(a.script)
        if extra:
            a = ap.parse_args(argv[:1] + extra + argv[1:])   # later (command-line) options override directives
            if cli_only.gpus == 0 and not cli_only.gpu_type:
                a.gpu_type = ""                             # -G 0 on the command line drops a directive's GPU type
    if getattr(a, "args", None) and a.args[:1] == ["--"]:
        a.args = a.args[1:]
    try:
        if a.cmd not in ("install-skills", "publish-status"):
            auto_install_skills()
        if a.cmd not in ("install-skills", "acct"):
            try:
                core._load_config()
            except ConfigException:
                ap.exit(1, "kjob: no Kubernetes access; run kjob inside a Kubeflow Notebook (or set KUBECONFIG)\n")
        return a.fn(a)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:                 # e.g. klogs ... | head
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except ValueError as e:
        err(str(e))
    except cluster.ClusterViewUnavailable as e:
        ap.exit(1, f"kjob: {e}\n")
    except ApiException as e:
        if e.status == 409 and a.cmd in ("submit", "run"):
            ap.exit(1, f"kjob: a job named {_job_name(getattr(a, 'job_name', '') or '')} already exists; choose another -J or "
                       f"kcancel it first\n")
        ap.exit(1, f"kjob: Kubernetes API error ({e.status}): {e.reason}\n")
    except HTTPError as e:  # connection refused/reset, timeouts: the job itself is unaffected
        ap.exit(1, f"kjob: lost connection to the Kubernetes API ({type(e).__name__}); submitted jobs keep running, "
                   f"check them later with kqueue / kwait NAME\n")


# console scripts installed by pip (pyproject.toml); the bin/ wrappers do the same for a source checkout
def ksub() -> int: return main(["submit", *sys.argv[1:]])
def ksrun() -> int: return main(["run", *sys.argv[1:]])
def kwait() -> int: return main(["wait", *sys.argv[1:]])
def kqueue() -> int: return main(["queue", *sys.argv[1:]])
def kshow() -> int: return main(["show", *sys.argv[1:]])
def klogs() -> int: return main(["logs", *sys.argv[1:]])
def kexec() -> int: return main(["exec", *sys.argv[1:]])
def kcancel() -> int: return main(["cancel", *sys.argv[1:]])
def kclean() -> int: return main(["clean", *sys.argv[1:]])
def kacct() -> int: return main(["acct", *sys.argv[1:]])
def ksinfo() -> int: return main(["sinfo", *sys.argv[1:]])
def kinfo() -> int: return main(["info", *sys.argv[1:]])
