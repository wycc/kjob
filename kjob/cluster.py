"""Cluster-wide view: GPU nodes and every GPU workload (like sinfo / squeue on a Slurm cluster).

A Notebook's service account can only see its own namespace, so the cluster view comes from one of two places:

* the Kubernetes API directly, when the caller may list nodes and pods in all namespaces (administrators), or
* the ConfigMap ``kjob-system/kjob-cluster-status`` that ``kjob publish-status`` refreshes
  (see deploy/cluster-status.yaml). It holds only scheduling facts: namespace, workload name, GPUs, node, state.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time

from kubernetes import client
from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity

STATUS_NAMESPACE = os.environ.get("KJOB_STATUS_NAMESPACE", "kjob-system")
STATUS_CONFIGMAP = "kjob-cluster-status"
STATUS_KEY = "status.json"
PRODUCT_LABEL = "nvidia.com/gpu.product"
GPU = "nvidia.com/gpu"


class ClusterViewUnavailable(RuntimeError):
    pass


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(ts) -> str | None:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


def _requests(pod, key: str) -> float:
    """Effective request, including overlapping restartable init sidecars and Pod overhead."""
    def req(c):
        v = ((c.resources and c.resources.requests) or {}).get(key)
        return float(parse_quantity(v)) if v is not None else 0.0
    sidecars, init_peak = 0.0, 0.0
    for c in pod.spec.init_containers or []:
        if getattr(c, "restart_policy", None) == "Always":
            sidecars += req(c)
            init_peak = max(init_peak, sidecars)
        else:
            init_peak = max(init_peak, sidecars + req(c))
    total = max(sum(req(c) for c in pod.spec.containers or []) + sidecars, init_peak)
    overhead = (pod.spec.overhead or {}).get(key)
    return total + (float(parse_quantity(overhead)) if overhead is not None else 0.0)


def _affinity_values(pod, key: str) -> list[str]:
    na = pod.spec.affinity and pod.spec.affinity.node_affinity
    req = na and na.required_during_scheduling_ignored_during_execution
    values = []
    for term in (req.node_selector_terms if req else []) or []:
        for e in term.match_expressions or []:
            if e.key == key and e.operator == "In":
                values += e.values or []
        if key == "kubernetes.io/hostname":     # kjob pins nodes by name: matchFields metadata.name
            values += [v for e in term.match_fields or [] if e.key == "metadata.name" and e.operator == "In"
                       for v in e.values or []]
    if pod.spec.node_selector and key in pod.spec.node_selector:
        values.append(pod.spec.node_selector[key])
    return values


def _workload(pod) -> tuple[str, str]:
    """(kind, name): Notebook, kjob, PyTorchJob or Pod."""
    labels = pod.metadata.labels or {}
    owners = pod.metadata.owner_references or []
    job = next((o.name for o in owners if o.kind == "PyTorchJob"), None)
    if job:
        return ("kjob" if labels.get("app.kubernetes.io/managed-by") == "kjob" else "PyTorchJob"), job
    if "notebook-name" in labels:
        return "Notebook", labels["notebook-name"]
    return "Pod", pod.metadata.name


def build_snapshot(core_api: client.CoreV1Api | None = None, hide_names: bool = False) -> dict:
    """Read nodes and pods (needs cluster-wide list permission) and summarise GPU usage."""
    core_api = core_api or client.CoreV1Api()
    nodes = core_api.list_node().items
    pods = core_api.list_pod_for_all_namespaces(field_selector="status.phase!=Succeeded,status.phase!=Failed").items
    out_nodes = {}
    for n in nodes:
        alloc = n.status.allocatable or {}
        if float(parse_quantity(alloc.get(GPU, "0"))) <= 0:
            continue
        labels = n.metadata.labels or {}
        ready = any(c.type == "Ready" and c.status == "True" for c in n.status.conditions or [])
        out_nodes[n.metadata.name] = {
            "name": n.metadata.name, "product": labels.get(PRODUCT_LABEL, ""),
            "memory_mib": int(labels.get("nvidia.com/gpu.memory", "0") or 0),
            "sharing": labels.get("nvidia.com/gpu.sharing-strategy", ""),
            "gpus": int(float(parse_quantity(alloc.get(GPU, "0")))), "gpus_used": 0,
            "cpus": float(parse_quantity(alloc.get("cpu", "0"))), "cpus_used": 0.0,
            "memory_gib": round(float(parse_quantity(alloc.get("memory", "0"))) / 2**30, 1), "memory_used_gib": 0.0,
            "ready": ready, "schedulable": not n.spec.unschedulable,
            "restricted": sorted({f"{t.key}={t.value}" if t.value else t.key for t in n.spec.taints or []
                                  if t.effect in ("NoSchedule", "NoExecute")
                                  and not t.key.startswith("node.kubernetes.io/")}),  # cordon/not-ready: shown as STATE
            "taints": [{"key": t.key, "value": t.value or "", "effect": t.effect} for t in n.spec.taints or []
                       if t.effect in ("NoSchedule", "NoExecute") and not t.key.startswith("node.kubernetes.io/")],
        }
    workloads = []
    now = _now()
    for p in pods:
        gpus = _requests(p, GPU)
        kind, name = _workload(p)
        node = p.spec.node_name
        if node in out_nodes:
            out_nodes[node]["gpus_used"] += int(gpus)
            out_nodes[node]["cpus_used"] += _requests(p, "cpu")
            out_nodes[node]["memory_used_gib"] += _requests(p, "memory") / 2**30
        if not gpus and kind not in ("kjob", "PyTorchJob"):
            continue
        reason = ""
        if p.status.phase == "Pending":
            for c in p.status.conditions or []:
                if c.type == "PodScheduled" and c.status == "False":
                    reason = (c.message or c.reason or "")[:200]
                    if hide_names:      # scheduler messages can name other users' volumes
                        reason = re.sub(r'"[^"]*"', '"-"', reason)
        start = p.status.start_time or p.metadata.creation_timestamp
        workloads.append({
            "namespace": p.metadata.namespace, "kind": kind, "name": "-" if hide_names else name,
            "pod": "-" if hide_names else p.metadata.name,
            "state": "Running" if p.status.phase == "Running" and node else "Pending",
            "gpus": int(gpus), "cpus": _requests(p, "cpu"), "memory_gib": round(_requests(p, "memory") / 2**30, 1),
            "node": node or "", "products": _affinity_values(p, PRODUCT_LABEL),
            "nodes_wanted": _affinity_values(p, "kubernetes.io/hostname"),
            "since": _iso(start), "age_s": int((now - start).total_seconds()) if start else 0, "reason": reason,
        })
    for n in out_nodes.values():
        n["cpus_used"] = round(n["cpus_used"], 1)
        n["memory_used_gib"] = round(n["memory_used_gib"], 1)
    return {"generated": _iso(now), "nodes": sorted(out_nodes.values(), key=lambda n: n["name"]), "workloads": workloads}


def node_type(node: dict) -> str:
    """Display name for a node's GPU type; nodes without a product label are shown by host name."""
    return node["product"] or node["name"]


def workload_type(w: dict, nodes: dict) -> str:
    """GPU type a workload runs on, or asks for while pending."""
    if w["node"] in nodes:
        return node_type(nodes[w["node"]])
    if w["products"]:
        return ",".join(w["products"])
    wanted = [nodes[n] for n in w["nodes_wanted"] if n in nodes]
    return ",".join(sorted({node_type(n) for n in wanted})) if wanted else "any"


def publish(snapshot: dict, core_api: client.CoreV1Api | None = None) -> None:
    core_api = core_api or client.CoreV1Api()
    body = client.V1ConfigMap(metadata=client.V1ObjectMeta(name=STATUS_CONFIGMAP, namespace=STATUS_NAMESPACE,
                                                           labels={"app.kubernetes.io/managed-by": "kjob"}),
                              data={STATUS_KEY: json.dumps(snapshot, separators=(",", ":"))})
    try:   # patch only the data: keeps the labels and kubectl annotations of the pre-created ConfigMap
        core_api.patch_namespaced_config_map(STATUS_CONFIGMAP, STATUS_NAMESPACE, {"data": body.data})
    except ApiException as e:
        if e.status != 404:
            raise
        core_api.create_namespaced_config_map(STATUS_NAMESPACE, body)


def publish_loop(interval: int = 30, hide_names: bool = False, report=print) -> None:
    core_api = client.CoreV1Api()
    while True:
        try:
            snap = build_snapshot(core_api, hide_names)
            publish(snap, core_api)
            report(f"kjob: published {len(snap['nodes'])} GPU nodes, {len(snap['workloads'])} workloads")
        except Exception as e:  # keep publishing after transient API errors
            report(f"kjob: publish failed: {type(e).__name__}: {e}")
        time.sleep(interval)


def load_snapshot(core_api: client.CoreV1Api | None = None) -> dict:
    """Live data when permitted, otherwise the published ConfigMap."""
    core_api = core_api or client.CoreV1Api()
    try:
        snap = build_snapshot(core_api)
        snap["source"] = "live"
        return snap
    except ApiException as e:
        if e.status not in (401, 403):
            raise
    try:
        cm = core_api.read_namespaced_config_map(STATUS_CONFIGMAP, STATUS_NAMESPACE)
    except ApiException as e:
        if e.status not in (401, 403, 404):
            raise
        raise ClusterViewUnavailable(
            "The cluster-wide view is not available here: this account cannot list nodes, and the "
            f"{STATUS_NAMESPACE}/{STATUS_CONFIGMAP} ConfigMap is missing. Ask the administrator to deploy "
            "deploy/cluster-status.yaml.") from None
    snap = json.loads((cm.data or {}).get(STATUS_KEY) or "{}")
    if not snap.get("generated"):
        raise ClusterViewUnavailable(f"{STATUS_NAMESPACE}/{STATUS_CONFIGMAP} has no data yet; the status service "
                                     "publishes it every 30 s.")
    snap["source"] = "published"
    return snap


def snapshot_age(snap: dict) -> int:
    ts = snap.get("generated")
    if not ts:
        return 0
    return int((_now() - _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds())


def matches_type(query: str, node: dict) -> bool:
    """Case-insensitive match of a -g / filter value against a node's product label or host name."""
    q = query.lower().replace(" ", "").replace("-", "")
    return any(q and q in s.lower().replace("-", "") for s in (node["product"], node["name"]))
