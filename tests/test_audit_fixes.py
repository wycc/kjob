"""Regression tests for the 2026-10-22 notebook audit (F01-F17): each reproduces the audit's trigger."""
import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kubernetes import client

from kjob import cli, cluster, core
from test_kjob import notebook_pod, PODDEFAULTS, template, pod_spec
from test_regressions import load_template, manifest_from, api_error, run_main


def fixture(shm=True):
    """A notebook with an NFS home and (optionally) a memory /dev/shm, like the audit's fixture."""
    mounts = [client.V1VolumeMount(name="home", mount_path="/home/jovyan")]
    volumes = [client.V1Volume(name="home", nfs=client.V1NFSVolumeSource(server="nfs.test", path="/home"))]
    if shm:
        mounts.append(client.V1VolumeMount(name="shm", mount_path="/dev/shm"))
        volumes.append(client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(medium="Memory")))
    return client.V1Pod(metadata=client.V1ObjectMeta(name="nb-0", namespace="me"),
                        spec=client.V1PodSpec(node_name="n1", volumes=volumes, containers=[
                            client.V1Container(name="notebook", image="img", volume_mounts=mounts)]))


def job(t, **kw):
    with mock.patch.object(core.client, "CoreV1Api"), mock.patch.object(core.client, "CustomObjectsApi"):
        return core.KJob(core.JobSpec(name="t", script="bash", gpus=0, **kw), t)


def mounts(m):
    return {v["mountPath"]: v["name"] for v in pod_spec(m)["containers"][0].get("volumeMounts", [])}


class StorageAndShmTests(unittest.TestCase):
    def test_f01_shm_is_added_when_the_notebook_has_none(self):
        m = manifest_from(load_template(fixture(shm=False), {"items": []}), shm="8Gi")
        name = mounts(m)["/dev/shm"]
        vol = next(v for v in pod_spec(m)["volumes"] if v["name"] == name)
        self.assertEqual(vol["emptyDir"], {"medium": "Memory", "sizeLimit": "8Gi"})
        m = manifest_from(load_template(fixture(shm=True), {"items": []}), shm="8Gi")    # existing one is resized
        self.assertEqual(sum(1 for p in mounts(m) if p == "/dev/shm"), 1)

    def test_default_shm_follows_memory_so_dataloader_workers_fit(self):
        m = manifest_from(load_template(fixture(shm=False), {"items": []}), memory="16Gi")
        vol = next(v for v in pod_spec(m)["volumes"] if v["name"] == mounts(m)["/dev/shm"])
        self.assertEqual(vol["emptyDir"]["sizeLimit"], "16Gi")

    def test_f03_projected_config_is_kept_and_only_tokens_are_dropped(self):
        p = fixture()
        p.spec.volumes += [
            client.V1Volume(name="config", projected=client.V1ProjectedVolumeSource(sources=[
                client.V1VolumeProjection(config_map=client.V1ConfigMapProjection(name="cfg"))])),
            client.V1Volume(name="token", projected=client.V1ProjectedVolumeSource(sources=[
                client.V1VolumeProjection(service_account_token=client.V1ServiceAccountTokenProjection(path="token"))]))]
        p.spec.containers[0].volume_mounts += [client.V1VolumeMount(name="config", mount_path="/etc/cfg"),
                                               client.V1VolumeMount(name="token", mount_path="/var/run/tok")]
        t = load_template(p, {"items": []})
        self.assertIn("config", [v.name for v in t.volumes])
        self.assertIn("service-account", t.skipped["token"])

    def test_f04_f05_emptydir_and_hostpath_are_not_shared_storage(self):
        for kind in ("emptyDir", "hostPath"):
            p = fixture()
            src = {"empty_dir": client.V1EmptyDirVolumeSource()} if kind == "emptyDir" else \
                {"host_path": client.V1HostPathVolumeSource(path="/scratch")}
            p.spec.volumes.append(client.V1Volume(name="scratch", **src))
            p.spec.containers[0].volume_mounts.append(client.V1VolumeMount(name="scratch", mount_path="/scratch"))
            pds = {"items": [{"spec": {"volumes": [{"name": "scratch", "emptyDir": {}}]}}]} if kind == "emptyDir" else {"items": []}
            t = load_template(p, pds)
            self.assertNotIn("/scratch", t.shared_paths(), kind)
            j = job(t, workdir="/scratch", log_dir="/scratch")
            with self.assertRaisesRegex(ValueError, "not on a shared volume"):
                j._warn_unshared_paths()


class EnvTests(unittest.TestCase):
    def test_f02_reference_to_the_notebook_container_points_at_the_job_container(self):
        p = fixture()
        p.spec.containers[0].env = [client.V1EnvVar(name="MEM", value_from=client.V1EnvVarSource(
            resource_field_ref=client.V1ResourceFieldSelector(container_name="notebook", resource="limits.memory")))]
        env = pod_spec(manifest_from(load_template(p, {"items": []})))["containers"][0]["env"]
        ref = next(e for e in env if e["name"] == "MEM")["valueFrom"]["resourceFieldRef"]
        self.assertEqual(ref.get("containerName", "pytorch"), "pytorch")

    def test_f15_override_keeps_the_position_so_dependent_values_expand(self):
        p = fixture()
        p.spec.containers[0].env = [client.V1EnvVar(name="BASE", value="/old"), client.V1EnvVar(name="CACHE", value="$(BASE)/cache")]
        env = pod_spec(manifest_from(load_template(p, {"items": []}), env={"BASE": "/new", "EXTRA": "1"}))["containers"][0]["env"]
        names = [e["name"] for e in env]
        self.assertLess(names.index("BASE"), names.index("CACHE"))
        self.assertEqual(next(e["value"] for e in env if e["name"] == "BASE"), "/new")
        self.assertEqual(names.count("BASE"), 1)
        self.assertIn("EXTRA", names)


class ExitCodeAndLogTests(unittest.TestCase):
    def pod(self, statuses):
        return client.V1Pod(metadata=client.V1ObjectMeta(), spec=client.V1PodSpec(containers=[]),
                            status=client.V1PodStatus(phase="Failed", container_statuses=[
                                client.V1ContainerStatus(name=n, image="i", image_id="i", ready=False, restart_count=0,
                                    state=client.V1ContainerState(terminated=client.V1ContainerStateTerminated(exit_code=c)))
                                for n, c in statuses]))

    def test_f07_exit_code_and_logs_come_from_the_pytorch_container(self):
        j = job(template())
        with mock.patch.object(j.core, "read_namespaced_pod", return_value=self.pod([("a-helper", 0), ("pytorch", 137)])):
            self.assertEqual(j.exit_code(), 137)
        j.core.read_namespaced_pod_log.return_value.stream.return_value = [b"hi\n"]
        with mock.patch.object(j, "_pod"):
            list(j.stream_logs())
        self.assertEqual(j.core.read_namespaced_pod_log.call_args.kwargs.get("container"), "pytorch")

    def test_f06_unwritable_log_fails_before_the_script_runs(self):
        with tempfile.TemporaryDirectory() as d:
            blocked = Path(d) / "notdir"
            blocked.write_text("x")
            marker = Path(d) / "ran"
            shell = core.job_shell("j", "touch " + str(marker), d, str(blocked / "j.log"), "r1")
            p = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=10, env={"TMPDIR": d, "PATH": "/usr/bin:/bin"})
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertIn("cannot write the log", p.stdout + p.stderr)

    def test_f13_exit_lines_carry_the_run_id(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "j.log"
            log.write_text("[kjob 2026-01-01T00:00:01Z] start j on node n run aaa\n[kjob 2026-01-01T00:00:02Z] start j on node m run bbb\n"
                           "[kjob 2026-01-01T00:00:03Z] exit code 0 run bbb\n[kjob 2026-01-01T00:00:04Z] exit code 137 run aaa\n")
            self.assertEqual(core.log_result(str(log), "bbb")[2], 0)
            self.assertEqual(core.log_result(str(log), "aaa")[2], 137)
            shell = core.job_shell("j", "true", d, str(log), "ccc")
            self.assertIn("exit code $rc run ccc", shell)

    def test_f12_old_records_match_the_nearest_earlier_submission(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "j.log"
            log.write_text("[kjob 2026-01-01T00:00:01Z] start j on node n\n[kjob 2026-01-01T00:00:10Z] exit code 0\n")
            hist = [{"name": "j", "submitted": ts, "log": str(log), "gpus": 0} for ts in ("2026-01-01T00:00:00Z", "2026-01-01T00:00:20Z")]
            out = io.StringIO()
            with mock.patch.object(core, "read_history", return_value=hist), \
                 mock.patch.object(core, "list_jobs", return_value=[{"metadata": {"name": "j"}}]), \
                 mock.patch.object(core, "job_state", return_value="Created"), mock.patch("sys.stdout", out):
                cli.cmd_acct(argparse.Namespace(n=0))
            self.assertEqual([r.split()[5] for r in out.getvalue().splitlines()[1:]], ["COMPLETED", "PENDING"])


class SchedulingTests(unittest.TestCase):
    def test_f08_requests_include_init_containers_and_overhead(self):
        p = fixture()
        p.spec.containers[0].resources = client.V1ResourceRequirements(requests={"cpu": "1"})
        p.spec.init_containers = [client.V1Container(name="init", resources=client.V1ResourceRequirements(requests={"cpu": "8"}))]
        p.spec.overhead = {"cpu": "500m"}
        self.assertEqual(cluster._requests(p, "cpu"), 8.5)

    def test_f09_toleration_effect_must_match(self):
        noexec = [{"key": "dedicated", "operator": "Equal", "value": "team", "effect": "NoExecute"}]
        self.assertFalse(cli._tolerated("dedicated=team:NoSchedule", noexec))
        self.assertTrue(cli._tolerated("dedicated=team:NoExecute", noexec))
        self.assertTrue(cli._tolerated("dedicated=team:NoSchedule", [{"key": "dedicated", "operator": "Exists"}]))
        self.assertTrue(cli._tolerated("dedicated=team", noexec))          # older status data without the effect

    def test_f10_stale_cluster_data_does_not_hard_reject_a_full_label(self):
        stale = {"generated": "2000-01-01T00:00:00Z", "source": "published",
                 "nodes": [{"name": "old", "product": "Tesla-V100-SXM2-16GB", "restricted": []}]}
        self.assertEqual(cli.resolve_gpu_types(["NVIDIA-H100-80GB-HBM3"], None, stale), (["NVIDIA-H100-80GB-HBM3"], None))
        fresh = {**stale, "source": "live"}
        with self.assertRaisesRegex(ValueError, "matches nothing"):
            cli.resolve_gpu_types(["NVIDIA-H100-80GB-HBM3"], None, fresh)

    def test_f11_job_matches_the_same_poddefaults_as_the_notebook(self):
        p = fixture()
        p.metadata.labels = {"skip-extra": "true"}
        selector = {"matchExpressions": [{"key": "skip-extra", "operator": "DoesNotExist"}]}
        t = load_template(p, {"items": [{"spec": {"selector": selector}}]})
        labels = manifest_from(t)["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["metadata"]["labels"]
        self.assertFalse(core._matches(selector, labels))

    def test_f14_node_pin_uses_the_node_name_field(self):
        spec = pod_spec(manifest_from(template(), node="node-a", exclude_nodes=["node-b"]))
        term = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]
        fields = term.get("matchFields", [])
        self.assertIn({"key": "metadata.name", "operator": "In", "values": ["node-a"]}, fields)
        self.assertIn({"key": "metadata.name", "operator": "NotIn", "values": ["node-b"]}, fields)
        self.assertFalse(any(e["key"] == "kubernetes.io/hostname" for e in term.get("matchExpressions", [])))
        pod = client.ApiClient()._ApiClient__deserialize({"metadata": {}, "spec": {**spec, "containers": spec["containers"]}}, "V1Pod")
        self.assertEqual(cluster._affinity_values(pod, "kubernetes.io/hostname"), ["node-a"])   # cluster view still sees the pin


class SkillAndMessageTests(unittest.TestCase):
    def test_f16_directories_kjob_did_not_install_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            home, claude = Path(d) / "agents", Path(d) / "claude"
            mine = claude / "kjob-jobs"
            mine.mkdir(parents=True)
            (mine / "notes.txt").write_text("keep me")
            other = Path(d) / "elsewhere"
            other.mkdir()
            (claude / "kjob-install").symlink_to(other, target_is_directory=True)
            cli.link_claude_skills(home, claude)
            self.assertTrue((mine / "notes.txt").exists())
            self.assertFalse(mine.is_symlink())
            self.assertEqual(Path(cli.os.readlink(claude / "kjob-install")), other)

    def test_f17_admission_rejections_are_not_reported_as_missing_permissions(self):
        code, err = run_main(api_error(403, 'pods "x" is forbidden: no PriorityClass with name high was found'))
        self.assertIn("no PriorityClass", err)
        self.assertNotIn("default-editor", err)
        code, err = run_main(api_error(400, 'admission webhook "poddefaults.kubeflow.org" denied the request: merging volumes conflict', "BadRequest"))
        self.assertIn("admission", err)
        self.assertNotIn("default-editor", err)


class LiveFindingTests(unittest.TestCase):
    """Found by running real jobs in the test notebook after the audit fixes."""

    def following(self, lines, reason=None, pod_code=None, state="Succeeded"):
        j = job(template())
        out, msgs = io.StringIO(), []
        with mock.patch.object(j, "wait_for_pod", return_value="Running"), mock.patch.object(j, "_pod"), \
             mock.patch.object(j, "stream_logs", return_value=iter(lines)), mock.patch.object(j, "state", return_value=state), \
             mock.patch.object(j, "exit_code", return_value=pod_code), mock.patch.object(j, "stop_reason", return_value=reason):
            code = j.follow(out=out, report=msgs.append)
        return code, out.getvalue(), msgs

    def test_exit_line_with_run_id_is_read_and_not_printed(self):
        code, out, _ = self.following(["[kjob 2026-01-01T00:00:00Z] start t on node n run abc", "hello",
                                       "[kjob 2026-01-01T00:00:01Z] exit code 3 run abc"], state="Failed")
        self.assertEqual(code, 3)
        self.assertNotIn("exit code", out)

    def test_time_limit_is_not_reported_as_success(self):
        code, _, msgs = self.following(["[kjob t] exit code 0 run abc"], reason="DeadlineExceeded", pod_code=0, state="Failed")
        self.assertEqual(code, 124)
        self.assertIn("time limit", msgs[-1])

    def test_cancelling_a_running_job_shows_cancelled_in_kacct(self):
        with tempfile.TemporaryDirectory() as d:
            hist = Path(d) / "history.jsonl"
            log = Path(d) / "c.log"
            log.write_text("[kjob 2026-01-01T00:00:05Z] start c on node n run r1\n")
            hist.write_text(json.dumps({"name": "c", "submitted": "2026-01-01T00:00:00Z", "log": str(log), "gpus": 0, "run_id": "r1"}) + "\n")
            live = [{"metadata": {"name": "c", "uid": "u"}}, {"metadata": {"name": "done", "uid": "v"}}]
            states = {"c": "Running", "done": "Succeeded"}
            with mock.patch.object(core, "HISTORY_FILE", hist), mock.patch.object(core, "current_namespace", return_value="me"), \
                 mock.patch.object(core, "list_jobs", return_value=live), \
                 mock.patch.object(core, "job_state", side_effect=lambda o: states[o["metadata"]["name"]]), \
                 mock.patch.object(core, "delete_job", return_value=True), mock.patch("sys.stdout", io.StringIO()):
                cli.cmd_cancel(argparse.Namespace(names=["c", "done"], finished=False, all_mine=False, all_notebooks=False))
            events = [json.loads(l) for l in hist.read_text().splitlines() if "event" in l]
            self.assertEqual([(e["event"], e["name"]) for e in events], [("cancel", "c")])     # finished one: no event
            with mock.patch.object(core, "HISTORY_FILE", hist):
                self.assertEqual([r["name"] for r in core.read_history()], ["c"])            # events are not submissions
            out = io.StringIO()
            with mock.patch.object(core, "HISTORY_FILE", hist), mock.patch.object(core, "list_jobs", return_value=[]), \
                 mock.patch("sys.stdout", out):
                cli.cmd_acct(argparse.Namespace(n=0))
            self.assertIn("CANCELLED", out.getvalue())


if __name__ == "__main__":
    unittest.main()
