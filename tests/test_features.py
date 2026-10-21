"""Tests for the cluster view, history, time limit and script directives (no cluster needed)."""
import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kubernetes import client

from kjob import cli, cluster, core
from test_kjob import manifest


def k8s(obj, kind):
    return client.ApiClient()._ApiClient__deserialize(obj, kind)


def node(name, gpus, product=None, taint=None):
    labels = {"nvidia.com/gpu.product": product} if product else {}
    return k8s({"metadata": {"name": name, "labels": labels},
                "spec": {"taints": [{"key": taint, "value": "x", "effect": "NoSchedule"}] if taint else None},
                "status": {"allocatable": {"nvidia.com/gpu": str(gpus), "cpu": "64", "memory": "256Gi"},
                           "conditions": [{"type": "Ready", "status": "True"}]}}, "V1Node")


def pod(ns, name, gpus, node_name=None, phase="Running", labels=None, owner=None, affinity_host=None):
    spec = {"containers": [{"name": "c", "image": "i", "resources": {"requests": {"nvidia.com/gpu": str(gpus), "cpu": "2"}}}],
            "nodeName": node_name}
    if affinity_host:
        spec["affinity"] = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
            {"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": [affinity_host]}]}]}}}
    meta = {"namespace": ns, "name": name, "labels": labels or {}, "creationTimestamp": "2026-01-01T00:00:00Z"}
    if owner:
        meta["ownerReferences"] = [{"apiVersion": "kubeflow.org/v1", "kind": "PyTorchJob", "name": owner, "uid": "u"}]
    status = {"phase": phase}
    if phase == "Pending":
        status["conditions"] = [{"type": "PodScheduled", "status": "False", "message": "0/8 nodes are available"}]
    return k8s({"metadata": meta, "spec": spec, "status": status}, "V1Pod")


def fake_api():
    api = mock.Mock()
    api.list_node.return_value.items = [node("a5000-1", 10, "NVIDIA-RTX-A5000"), node("rtxpro6000", 8, taint="restricted-gpu"),
                                        node("cpu-1", 0)]
    api.list_pod_for_all_namespaces.return_value.items = [
        pod("alice", "nb-0", 2, "a5000-1", labels={"notebook-name": "nb"}),
        pod("bob", "train-master-0", 4, "rtxpro6000", labels={"app.kubernetes.io/managed-by": "kjob"}, owner="train"),
        pod("carol", "big-0", 8, phase="Pending", labels={"notebook-name": "big"}, affinity_host="rtxpro6000"),
        pod("dave", "cpu-pod", 0, "cpu-1"),
    ]
    return api


class SnapshotTest(unittest.TestCase):
    def test_counts_gpus_per_node_and_classifies_workloads(self):
        snap = cluster.build_snapshot(fake_api())
        nodes = {n["name"]: n for n in snap["nodes"]}
        self.assertEqual(set(nodes), {"a5000-1", "rtxpro6000"})            # CPU-only node left out
        self.assertEqual((nodes["a5000-1"]["gpus_used"], nodes["rtxpro6000"]["gpus_used"]), (2, 4))
        self.assertEqual(nodes["rtxpro6000"]["restricted"], ["restricted-gpu=x"])
        kinds = {w["name"]: (w["kind"], w["state"]) for w in snap["workloads"]}
        self.assertEqual(kinds, {"nb": ("Notebook", "Running"), "train": ("kjob", "Running"), "big": ("Notebook", "Pending")})
        big = next(w for w in snap["workloads"] if w["name"] == "big")
        self.assertEqual(cluster.workload_type(big, nodes), "rtxpro6000")      # unlabeled node shown by host name
        self.assertIn("0/8 nodes", big["reason"])

    def test_falls_back_to_published_configmap(self):
        api = fake_api()
        api.list_node.side_effect = core.ApiException(status=403)
        api.read_namespaced_config_map.return_value = client.V1ConfigMap(data={"status.json": json.dumps(
            {"generated": "2026-01-01T00:00:00Z", "nodes": [], "workloads": []})})
        self.assertEqual(cluster.load_snapshot(api)["source"], "published")
        api.read_namespaced_config_map.return_value = client.V1ConfigMap(data={"status.json": ""})
        with self.assertRaisesRegex(cluster.ClusterViewUnavailable, "no data yet"):
            cluster.load_snapshot(api)
        api.read_namespaced_config_map.side_effect = core.ApiException(status=404)
        with self.assertRaises(cluster.ClusterViewUnavailable):
            cluster.load_snapshot(api)

    def test_type_matching_and_resolution(self):
        snap = cluster.build_snapshot(fake_api())
        self.assertEqual(cli.resolve_gpu_types(["a5000"], None, snap), (["a5000"], None))
        self.assertEqual(cli.resolve_gpu_types(["rtx-a5000"], None, snap), (["NVIDIA-RTX-A5000"], None))
        self.assertEqual(cli.resolve_gpu_types(["rtxpro6000"], None, snap), ([], "rtxpro6000"))
        self.assertEqual(cli.resolve_gpu_types(["RTX PRO 6000"], None, snap), ([], "rtxpro6000"))
        self.assertEqual(cli.resolve_gpu_types(["a5000-1"], None, snap), ([], "a5000-1"))     # host name pins
        for bad in ("h100", "rtx", "NVIDIA-H100"):      # unknown, ambiguous (A5000 and rtxpro6000), absent label
            with self.assertRaises(ValueError, msg=bad):
                cli.resolve_gpu_types([bad], None, snap)
        with self.assertRaisesRegex(ValueError, "several types"):
            cli.resolve_gpu_types(["rtx"], None, snap)
        with mock.patch.object(cluster, "load_snapshot", side_effect=cluster.ClusterViewUnavailable("x")):
            self.assertEqual(cli.resolve_gpu_types(["NVIDIA-A30"], None), (["NVIDIA-A30"], None))
            with self.assertRaisesRegex(ValueError, "Unknown GPU type"):
                cli.resolve_gpu_types(["rtxpro6000"], None)

    def test_host_name_that_names_its_labeled_type_selects_the_type(self):
        api = fake_api()
        api.list_node.return_value.items = [
            node("hs5800-2", 8, "Tesla-V100-SXM2-16GB"),
            node("rtxpro6000", 8, "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition", taint="restricted-gpu")]
        snap = cluster.build_snapshot(api)
        rtx = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
        self.assertEqual(cli.resolve_gpu_types(["rtxpro6000"], None, snap), ([rtx], None))
        self.assertEqual(cli.resolve_gpu_types(["v100", "rtxpro6000"], None, snap), (["v100", rtx], None))
        self.assertEqual(cli.resolve_gpu_types(["hs5800-2"], None, snap), ([], "hs5800-2"))     # plain host name pins
        with self.assertRaisesRegex(ValueError, "node name"):
            cli.resolve_gpu_types(["v100", "hs5800-2"], None, snap)

    def test_cordon_taint_is_not_a_restriction(self):
        api = fake_api()
        cordoned = node("hs5800-1", 8, "Tesla-V100-SXM2-16GB", taint="node.kubernetes.io/unschedulable")
        api.list_node.return_value.items = [cordoned]
        self.assertEqual(cluster.build_snapshot(api)["nodes"][0]["restricted"], [])

    def test_pinned_node_must_be_a_gpu_node(self):
        snap = cluster.build_snapshot(fake_api())
        with mock.patch.object(cluster, "load_snapshot", return_value=snap):
            cli._check_node("a5000-1", 1)
            with self.assertRaisesRegex(ValueError, "not a GPU node"):
                cli._check_node("typo-node", 1)
            cli._check_node("cpu-1", 0)                 # CPU jobs may go anywhere

    def test_sinfo_and_queue_output(self):
        snap = {**cluster.build_snapshot(fake_api()), "source": "live"}
        out = io.StringIO()
        with mock.patch.object(cluster, "load_snapshot", return_value=snap), mock.patch.object(cli, "_my_tolerations", return_value=[]), \
                mock.patch.object(core, "current_namespace", return_value="bob"), mock.patch("sys.stdout", out):
            cli._sinfo(argparse.Namespace(gpu_type="rtxpro6000"))
        text = out.getvalue()
        self.assertIn("rtxpro6000", text)
        self.assertIn("no access", text)
        self.assertIn("8 (1 jobs)", text)          # carol waits for 8 GPUs on rtxpro6000
        self.assertIn("*bob", text)
        self.assertNotIn("alice", text)            # filtered to rtxpro6000


class TimeAndDirectiveTest(unittest.TestCase):
    def test_time_formats(self):
        for bad in ("0", "0m", "abc", "1:2:3:4"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                cli._time_arg(bad)
        for v, s in [("30", 1800), ("10:30", 630), ("2:00:00", 7200), ("1-12", 129600), ("1-00:30", 88200),
                     ("1-0:0:5", 86405), ("90m", 5400), ("12h", 43200), ("2d", 172800)]:
            self.assertEqual(cli.parse_time(v), s, v)
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._time_arg("soon")

    def test_time_limit_in_manifest(self):
        m = manifest(time_limit=7200)
        self.assertEqual(m["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["activeDeadlineSeconds"], 7200)
        self.assertNotIn("activeDeadlineSeconds", m["spec"]["runPolicy"])     # would also count queueing time
        self.assertNotIn("activeDeadlineSeconds", manifest()["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"])

    def test_directives(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n#SBATCH -J train --cpus-per-task=8\n#SBATCH --mem=64G -t 1-00:00:00\n"
                    "#SBATCH --gres=gpu:v100:2 -p gp1d --account=ABC -N 1\n#KJOB --shm 16Gi\n\npython train.py\n#KJOB -G 7\n")
        err = io.StringIO()
        extra = cli.script_directives(f.name, warn=lambda m: err.write(m + "\n"))
        self.assertEqual(extra, ["-J", "train", "-c", "8", "--mem", "64Gi", "-t", "1-00:00:00", "-g", "v100", "-G", "2",
                                 "--shm", "16Gi"])                       # stops at the first command
        self.assertIn("-p", err.getvalue())
        self.assertIn("--account", err.getvalue())
        self.assertIn("-N 1", err.getvalue())
        self.assertNotIn("SBATCH 1", err.getvalue())
        a = cli.build_parser().parse_args(["submit"] + extra + ["-G", "1", f.name])
        self.assertEqual((a.gpus, a.cpus, a.time, a.job_name), (1, 8, 86400, "train"))   # command line wins
        Path(f.name).unlink()


class HistoryTest(unittest.TestCase):
    def test_history_and_log_result(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "j.log"
            log.write_text("[kjob 2099-01-01T00:00:00Z] start j on node n\nhello\n[kjob 2099-01-01T01:00:05Z] exit code 3\n")
            self.assertEqual(core.log_result(str(log)), ("2099-01-01T00:00:00Z", "2099-01-01T01:00:05Z", 3))
            job = mock.Mock(ns="ns")
            job.spec = core.JobSpec(name="j", script="x.py", log_dir=d)
            with mock.patch.object(core, "HISTORY_FILE", Path(d) / "h" / "history.jsonl"):
                core.record_history(job)
                self.assertEqual(core.read_history()[0]["name"], "j")
                out = io.StringIO()
                with mock.patch.object(core, "list_jobs", return_value=[]), mock.patch("sys.stdout", out):
                    cli.cmd_acct(argparse.Namespace(n=20))
            self.assertIn("FAILED", out.getvalue())
            self.assertIn("1h00m", out.getvalue())


class SkillsTest(unittest.TestCase):
    def test_install_skills(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", io.StringIO()):
            cli.cmd_install_skills(argparse.Namespace(dir=d))                     # --dir: only there
            names = sorted(p.parent.name for p in Path(d).glob("*/SKILL.md"))
        self.assertEqual(names, ["kjob-install", "kjob-jobs"])

    def test_updates_kjob_copies_but_keeps_user_edits(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            self.assertEqual(len(cli.install_skills(dest)), 2)
            self.assertEqual(cli.install_skills(dest), [])                        # already current
            jobs, inst = dest / "kjob-jobs" / "SKILL.md", dest / "kjob-install" / "SKILL.md"
            jobs.write_text("old version\n")                                       # an older kjob wrote this
            (jobs.parent / cli.SKILL_MARKER).write_text(cli._sha(jobs) + "\n")
            inst.write_text("edited by the user\n")                               # user's own change
            self.assertEqual(cli.install_skills(dest), [jobs])
            self.assertEqual(inst.read_text(), "edited by the user\n")
            self.assertEqual(cli.install_skills(dest, force=True), [inst])        # kjob install-skills overwrites

    def test_auto_install_agents_dir_with_claude_links(self):
        env = {k: v for k, v in cli.os.environ.items() if k != "KJOB_NO_SKILLS"}
        with tempfile.TemporaryDirectory() as home, mock.patch.object(cli.Path, "home", return_value=Path(home)), \
                mock.patch("sys.stderr", io.StringIO()) as err:
            h = Path(home)
            with mock.patch.dict(cli.os.environ, {"KJOB_NO_SKILLS": "1"}):
                cli.auto_install_skills()
            self.assertFalse((h / ".agents").exists())                             # turned off
            old = h / ".claude" / "skills" / "kjob-jobs"                          # copy made by the previous version
            old.mkdir(parents=True)
            (old / "SKILL.md").write_text("old\n")
            (old / cli.SKILL_MARKER).write_text(cli._sha(old / "SKILL.md") + "\n")
            mine = h / ".claude" / "skills" / "kjob-install"                      # a copy the user edited
            mine.mkdir()
            (mine / "SKILL.md").write_text("my notes\n")
            with mock.patch.dict(cli.os.environ, env, clear=True):
                cli.auto_install_skills()
                cli.auto_install_skills()                                         # second run: nothing to do
            self.assertTrue((h / ".agents" / "skills" / "kjob-jobs" / "SKILL.md").is_file())
            self.assertTrue(old.is_symlink())
            self.assertEqual(old.resolve(), (h / ".agents" / "skills" / "kjob-jobs").resolve())
            self.assertIn("kjob-jobs", (old / "SKILL.md").read_text())           # reads through the link
            self.assertFalse(mine.is_symlink())
            self.assertEqual((mine / "SKILL.md").read_text(), "my notes\n")
        self.assertEqual(err.getvalue().count("installed AI skills"), 1)


class PublishTest(unittest.TestCase):
    def test_creates_configmap_when_missing_and_can_hide_names(self):
        api = fake_api()
        api.patch_namespaced_config_map.side_effect = core.ApiException(status=404)
        snap = cluster.build_snapshot(api, hide_names=True)
        cluster.publish(snap, api)
        body = api.create_namespaced_config_map.call_args[0][1]
        data = json.loads(body.data["status.json"])
        self.assertEqual({w["name"] for w in data["workloads"]}, {"-"})
        self.assertEqual({w["namespace"] for w in data["workloads"]}, {"alice", "bob", "carol"})


class NamesAndWrapperTest(unittest.TestCase):
    def test_job_names_start_with_a_letter(self):
        self.assertEqual(core.sanitize_name("1digit"), "j-1digit")
        self.assertTrue(core.unique_name("2stage").startswith("j-2stage-"))
        self.assertTrue(core.unique_name("測試").startswith("job-"))

    def test_wrapper_forwards_sigterm_and_keeps_exit_code(self):
        import signal, subprocess, sys, time
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "s.py"
            script.write_text("import signal,sys,time\n"
                              "signal.signal(signal.SIGTERM, lambda *a: (print('saving', flush=True), sys.exit(7)))\n"
                              "print('ready', flush=True)\ntime.sleep(30)\n")
            log = Path(d) / "logs" / "j.log"
            sh = core.job_shell("j", f"{sys.executable} {script}", d, str(log))
            p = subprocess.Popen(["bash", "-c", sh], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                if log.exists() and "ready" in log.read_text():
                    break
                time.sleep(0.1)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=10), 7)
            text = log.read_text()
            self.assertIn("stopping: SIGTERM", text)
            self.assertIn("saving", text)
            runs = core.log_runs(str(log))
            self.assertEqual((len(runs), runs[0]["stopped"], runs[0]["code"]), (1, True, 7))
            p = subprocess.Popen(["bash", "-c", core.job_shell("j", "bash -c 'exit 3'", d, str(log))],
                                 stdout=subprocess.DEVNULL)
            self.assertEqual(p.wait(timeout=10), 3)
            self.assertEqual([r["code"] for r in core.log_runs(str(log))], [7, 3])
            cmd = f"{sys.executable} -c 'import signal; print(signal.getsignal(signal.SIGINT))'"
            subprocess.run(["bash", "-c", core.job_shell("j", cmd, d, str(log))], stdout=subprocess.DEVNULL, timeout=10)
            self.assertIn("default_int_handler", log.read_text())     # not ignored like other background commands

    def test_path_command_allowed_but_missing_file_rejected(self):
        from test_regressions import handle
        j = handle()
        j.t.volume_mounts = [client.V1VolumeMount(name="w", mount_path="/")]
        j.t.volumes = [client.V1Volume(name="w", nfs=client.V1NFSVolumeSource(server="s", path="/"))]
        j.spec.workdir, j.spec.log_dir = "/tmp", "/tmp"
        j.spec.script = "bash"                         # a command on PATH, like nvidia-smi
        j._warn_unshared_paths()
        j.spec.script = "/no/such/file.py"
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            j._warn_unshared_paths()


class CliBehaviourTest(unittest.TestCase):
    def test_cli_gpu0_drops_directive_gpu_type(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n#SBATCH --gres=gpu:v100:1 -c4\n#SBATCH --time=UNLIMITED --mem=0\necho hi\n")
        self.assertEqual(cli.script_directives(f.name, warn=lambda m: None),
                         ["-g", "v100", "-G", "1", "-c", "4"])           # UNLIMITED and --mem=0 dropped
        seen = {}
        with mock.patch.object(core, "_load_config"), \
                mock.patch.object(cli, "cmd_submit", side_effect=lambda a: seen.update(vars(a)) or 0):
            cli.main(["submit", "-G", "0", f.name])
        self.assertEqual((seen["gpus"], seen["gpu_type"], seen["cpus"]), (0, "", 4))
        Path(f.name).unlink()

    def test_kcancel_requires_targets_and_reports_missing(self):
        with mock.patch.object(core, "current_namespace", return_value="ns"):
            with self.assertRaises(ValueError):
                cli.cmd_cancel(argparse.Namespace(names=[], finished=False, all_mine=False, all_notebooks=False))
            with mock.patch.object(core, "delete_job", return_value=False), mock.patch("sys.stdout", io.StringIO()):
                self.assertEqual(cli.cmd_cancel(argparse.Namespace(names=["x"], finished=False, all_mine=False,
                                                                   all_notebooks=False)), 1)

    def test_hide_names_redacts_reasons(self):
        api = fake_api()
        pend = api.list_pod_for_all_namespaces.return_value.items[2]
        pend.status.conditions[0].message = 'persistentvolumeclaim "secret-data" not found'
        snap = cluster.build_snapshot(api, hide_names=True)
        reason = next(w["reason"] for w in snap["workloads"] if w["state"] == "Pending")
        self.assertNotIn("secret-data", reason)


class AcctTest(unittest.TestCase):
    def test_reused_name_stopped_and_pending(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "a.log"
            log.write_text("[kjob 2026-01-01T00:00:00Z] start a on node n\n[kjob 2026-01-01T00:00:10Z] exit code 0\n"
                           "[kjob 2026-01-01T01:00:00Z] start a on node n\n[kjob 2026-01-01T01:00:30Z] stopping: SIGTERM\n"
                           "[kjob 2026-01-01T01:00:31Z] exit code 143\n")
            hist = [{"name": "a", "submitted": "2026-01-01T00:00:00Z", "log": str(log), "gpus": 1},
                    {"name": "a", "submitted": "2026-01-01T00:59:00Z", "log": str(log), "gpus": 1},
                    {"name": "p", "submitted": "2026-01-01T02:00:00Z", "log": str(Path(d) / "p.log"), "gpus": 1}]
            out = io.StringIO()
            with mock.patch.object(core, "read_history", return_value=hist), \
                    mock.patch.object(core, "list_jobs", return_value=[{"metadata": {"name": "p"},
                                                                         "status": {"conditions": [{"type": "Created", "status": "True"}]}}]), \
                    mock.patch("sys.stdout", out):
                cli.cmd_acct(argparse.Namespace(n=0))
        lines = out.getvalue().splitlines()[1:]
        self.assertIn("COMPLETED", lines[0])
        self.assertIn("0m10s", lines[0])
        self.assertIn("STOPPED", lines[1])
        self.assertIn("143", lines[1])
        self.assertIn("PENDING", lines[2])


class AcctPairingTest(unittest.TestCase):
    def test_submission_that_never_started_is_not_paired(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "a.log"
            log.write_text("[kjob 2026-01-01T01:00:00Z] start a on node n\n[kjob 2026-01-01T01:00:30Z] exit code 3\n")
            hist = [{"name": "a", "submitted": "2026-01-01T00:00:00Z", "log": str(log), "gpus": 1},   # cancelled while pending
                    {"name": "a", "submitted": "2026-01-01T00:59:59Z", "log": str(log), "gpus": 1}]
            out = io.StringIO()
            with mock.patch.object(core, "read_history", return_value=hist), \
                    mock.patch.object(core, "list_jobs", return_value=[]), mock.patch("sys.stdout", out):
                cli.cmd_acct(argparse.Namespace(n=0))
        first, second = out.getvalue().splitlines()[1:]
        self.assertIn("ENDED?", first)
        self.assertIn("FAILED", second)
        self.assertIn("0m30s", second)


class GpuTypeChecksTest(unittest.TestCase):
    def test_alias_must_exist_and_match_pinned_node(self):
        snap = cluster.build_snapshot(fake_api())
        self.assertEqual(cli.resolve_gpu_types(["A5000"], None, snap), (["a5000"], None))
        with self.assertRaisesRegex(ValueError, "not in this cluster"):
            cli.resolve_gpu_types(["a30"], None, snap)
        with self.assertRaisesRegex(ValueError, "could never start"):
            cli.resolve_gpu_types(["a5000"], "rtxpro6000", snap)

    def test_cpu_job_cannot_pick_gpu_type(self):
        a = cli.build_parser().parse_args(["submit", "-G", "0", "-g", "rtxpro6000", "x.py"])
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            cli._spec(a)

    def test_user_names_are_normalised(self):
        self.assertEqual(cli._job_name("001"), "j-001")
        self.assertEqual(cli._job_name("KV_Dup"), "kv-dup")


class FakeExecStream:
    """Minimal stand-in for kubernetes.stream.ws_client.WSClient."""
    def __init__(self, stdout, stderr, code):
        self._out, self._err, self.returncode, self._open = stdout, stderr, code, True

    def is_open(self):
        return self._open

    def update(self, timeout=0):
        self._open = bool(self._out or self._err) and self._open

    def peek_stdout(self):
        return bool(self._out)

    def read_stdout(self):
        data, self._out = self._out, ""
        return data

    def peek_stderr(self):
        return bool(self._err)

    def read_stderr(self):
        data, self._err = self._err, ""
        return data

    def close(self):
        self._open = False


class ExecTest(unittest.TestCase):
    def job(self, phase="Running"):
        from test_regressions import handle, pod
        j = handle()
        j._pod = mock.Mock(return_value=pod(phase=phase))
        return j

    def test_runs_command_and_returns_exit_code(self):
        j = self.job()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("kubernetes.stream.stream", return_value=FakeExecStream("GPU 0: V100\n", "warn\n", 3)) as st:
            code = j.exec(["nvidia-smi", "-L"], stdin=io.StringIO(), out=out, err=err)
        self.assertEqual((code, out.getvalue(), err.getvalue()), (3, "GPU 0: V100\n", "warn\n"))
        kw = st.call_args.kwargs
        self.assertEqual((kw["command"], kw["container"], kw["tty"], kw["stdin"]), (["nvidia-smi", "-L"], "pytorch", False, False))

    def test_shell_needs_terminal_and_running_pod(self):
        with self.assertRaisesRegex(ValueError, "needs a terminal"):
            self.job().exec(None, stdin=io.StringIO())
        with self.assertRaisesRegex(ValueError, "not running"):
            self.job("Pending").exec(["true"], stdin=io.StringIO())

    def test_cli_parses_command_after_name(self):
        a = cli.build_parser().parse_args(["exec", "train", "nvidia-smi", "-L"])
        self.assertEqual((a.name, a.command), ("train", ["nvidia-smi", "-L"]))
