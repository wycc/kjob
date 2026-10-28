"""Offline tests: the notebook pod and API calls are mocked, nothing talks to a cluster."""
import argparse
import unittest
from unittest import mock

from kubernetes import client

from kjob import cli, core

NS, NOTEBOOK = "demo-user", "training-0"
NFS = "nfs.example.test"


def notebook_pod() -> client.V1Pod:
    """A notebook started before the group-share PodDefault renamed its group volumes (legacy and current volume names)."""
    volumes = {
        "training-workspace": {"persistentVolumeClaim": {"claimName": "training-workspace"}},
        "dshm": {"emptyDir": {"medium": "Memory"}},
        "gs-group-a": {"nfs": {"server": NFS, "path": "/export/groupshare/group-a"}},              # outdated
        "gs-group-a-7ef7d056": {"nfs": {"server": NFS, "path": "/export/groupshare/group-a-7ef7d056"}},
        "ns-demo-user": {"nfs": {"server": NFS, "path": "/export/namespaces/demo-user"}},
        "datasets": {"nfs": {"server": NFS, "path": "/export/datasets"}},                                    # not managed
        "kube-api-access-x": {"projected": {"sources": []}},
    }
    mounts = {"training-workspace": "/home/jovyan", "dshm": "/dev/shm", "gs-group-a": "/mnt/groups/group-a",
              "gs-group-a-7ef7d056": "/mnt/groups/group-a-7ef7d056", "ns-demo-user": "/mnt/namespaces/demo-user",
              "datasets": "/datasets", "kube-api-access-x": "/var/run/secrets/kubernetes.io/serviceaccount"}
    pod = {
        "metadata": {"name": NOTEBOOK, "namespace": NS},
        "spec": {
            "nodeName": "node-05",
            "containers": [
                {"name": "training", "image": "registry.example.com/notebooks/pytorch:1.0",
                 "volumeMounts": [{"name": n, "mountPath": p} for n, p in mounts.items()]},
                {"name": "istio-proxy", "image": "istio/proxyv2"},
            ],
            "volumes": [{"name": n, **src} for n, src in volumes.items()],
            "securityContext": {"fsGroup": 100},
        },
    }
    return client.ApiClient()._ApiClient__deserialize(pod, "V1Pod")


PODDEFAULTS = {"items": [
    {"spec": {"volumes": [{"name": "gs-group-a-7ef7d056"}]}},
    {"spec": {"volumes": [{"name": "ns-demo-user"}]}},
]}
PROC_NFS = {"/home/jovyan": (NFS, "/export/demo-user-training-workspace-pvc-1")}


def template() -> core.NotebookTemplate:
    core_api, custom_api = mock.Mock(), mock.Mock()
    core_api.read_namespaced_pod.return_value = notebook_pod()
    custom_api.list_namespaced_custom_object.return_value = PODDEFAULTS
    with mock.patch.object(core, "_load_config"), \
            mock.patch.object(core, "current_namespace", return_value=NS), \
            mock.patch.object(core, "current_notebook", return_value=NOTEBOOK), \
            mock.patch.object(core, "_nfs_mounts", return_value=PROC_NFS), \
            mock.patch.object(core.os, "getuid", return_value=0), \
            mock.patch.object(core.os, "getcwd", return_value="/home/jovyan/project"), \
            mock.patch.object(core.client, "CoreV1Api", return_value=core_api), \
            mock.patch.object(core.client, "CustomObjectsApi", return_value=custom_api):
        return core.NotebookTemplate.from_current_pod()


def manifest(**spec) -> dict:
    with mock.patch.object(core.client, "CoreV1Api"), mock.patch.object(core.client, "CustomObjectsApi"):
        job = core.KJob(core.JobSpec(name="t", script="/home/jovyan/train.py", **spec), template())
    return job.manifest()


def pod_spec(m: dict) -> dict:
    return m["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]


def env(m: dict) -> dict:
    return {e["name"]: e.get("value") for e in pod_spec(m)["containers"][0]["env"]}


class TemplateTest(unittest.TestCase):
    def test_volumes(self):
        t = template()
        self.assertEqual({v.name for v in t.volumes}, {"training-workspace", "dshm", "datasets"})
        self.assertIn("PodDefault", t.skipped["gs-group-a-7ef7d056"])
        self.assertIn("PodDefault", t.skipped["ns-demo-user"])
        self.assertIn("outdated", t.skipped["gs-group-a"])
        self.assertIn("service-account", t.skipped["kube-api-access-x"])

    def test_rwo_workspace_becomes_nfs(self):
        ws = next(v for v in template().volumes if v.name == "training-workspace")
        self.assertIsNone(ws.persistent_volume_claim)
        self.assertEqual((ws.nfs.server, ws.nfs.path), PROC_NFS["/home/jovyan"])

    def test_shared_paths(self):
        self.assertEqual(sorted(template().shared_paths()), ["/datasets", "/home/jovyan",   # + PodDefault mounts: jobs get them too
                                                             "/mnt/groups/group-a-7ef7d056", "/mnt/namespaces/demo-user"])


class ManifestTest(unittest.TestCase):
    def test_cpu_job_hides_gpus(self):
        m = manifest(gpus=0)
        self.assertEqual(env(m)["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(env(m)["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertNotIn("nvidia.com/gpu", pod_spec(m)["containers"][0]["resources"]["requests"])

    def test_cpu_job_env_override(self):
        for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            with self.assertRaisesRegex(ValueError, "cannot be overridden"):
                manifest(gpus=0, env={key: "0"})

    def test_gpu_job(self):
        m = manifest(gpus=2, gpu_types=["a30"])
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", env(m))
        res = pod_spec(m)["containers"][0]["resources"]
        self.assertEqual(res["requests"]["nvidia.com/gpu"], "2")
        expr = pod_spec(m)["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
        self.assertEqual(expr["nodeSelectorTerms"][0]["matchExpressions"][0]["values"], ["NVIDIA-A30"])

    def test_mps_job_uses_device_plugin_shm(self):
        m = manifest(gpus=1, gpu_types=["a30-shared"])
        self.assertNotIn("/dev/shm", [vm["mountPath"] for vm in pod_spec(m)["containers"][0]["volumeMounts"]])
        self.assertNotIn("dshm", [v["name"] for v in pod_spec(m)["volumes"]])
        self.assertIn("dshm", [v.name for v in template().volumes])  # the template itself is not modified

    def test_untyped_gpu_job_avoids_mps(self):
        terms = lambda m: pod_spec(m)["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"][0]["matchExpressions"]
        self.assertIn({"key": "nvidia.com/gpu.sharing-strategy", "operator": "NotIn", "values": ["mps"]}, terms(manifest(gpus=1)))
        self.assertIn("/dev/shm", [vm["mountPath"] for vm in pod_spec(manifest(gpus=1))["containers"][0]["volumeMounts"]])
        self.assertIn({"key": "nvidia.com/gpu.sharing-strategy", "operator": "NotIn", "values": ["mps"]},
                      terms(manifest(gpus=1, node="node-06")))
        self.assertIsNone(pod_spec(manifest(gpus=0)).get("affinity"))

    def test_no_service_account_token(self):
        self.assertIs(pod_spec(manifest())["automountServiceAccountToken"], False)

    def test_no_stale_group_mounts(self):
        mounts = [vm["mountPath"] for vm in pod_spec(manifest())["containers"][0]["volumeMounts"]]
        self.assertFalse([p for p in mounts if p.startswith(("/mnt/groups", "/mnt/namespaces"))])

    def test_labels_and_submitted(self):
        md = manifest()["metadata"]
        self.assertEqual(md["labels"]["kjob/notebook"], NOTEBOOK)
        self.assertRegex(md["annotations"]["kjob/submitted"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


class FollowTest(unittest.TestCase):
    def test_job_deleted_while_waiting(self):
        with mock.patch.object(core, "_load_config"), mock.patch.object(core, "current_namespace", return_value=NS), \
                mock.patch.object(core.client, "CoreV1Api") as c, mock.patch.object(core.client, "CustomObjectsApi") as o:
            o.return_value.get_namespaced_custom_object.side_effect = core.ApiException(status=404)
            c.return_value.read_namespaced_pod.side_effect = core.ApiException(status=404)
            c.return_value.list_namespaced_event.return_value.items = []
            j = cli._job_for("gone")
            msgs = []
            self.assertEqual(j.follow(report=msgs.append), 1)
        self.assertEqual(msgs, ["kjob: gone Deleted (exit code None)"])


class EventsTest(unittest.TestCase):
    def test_events_filtered_by_uid(self):
        with mock.patch.object(core.client, "CoreV1Api") as c:
            c.return_value.list_namespaced_event.return_value.items = []
            core.job_events(NS, "q-b", uid="u1")
            core.job_events(NS, "q-b")
        sels = [k.kwargs["field_selector"] for k in c.return_value.list_namespaced_event.call_args_list]
        self.assertEqual(sels, ["involvedObject.name=q-b,involvedObject.uid=u1", "involvedObject.name=q-b"])


def job(name, notebook, state):
    return {"metadata": {"name": name, "labels": {"kjob/notebook": notebook}},
            "status": {"conditions": [{"type": state, "status": "True"}]}}


class CliTest(unittest.TestCase):
    def test_list_jobs_selector(self):
        api = mock.Mock()
        api.list_namespaced_custom_object.return_value = {"items": []}
        with mock.patch.object(core, "_load_config"), mock.patch.object(core.client, "CustomObjectsApi", return_value=api):
            core.list_jobs(NS, notebook=NOTEBOOK)
            core.list_jobs(NS)
            core.list_jobs(NS, everyone=True, notebook=NOTEBOOK)
        sels = [c.kwargs["label_selector"] for c in api.list_namespaced_custom_object.call_args_list]
        self.assertEqual(sels, [f"app.kubernetes.io/managed-by=kjob,kjob/notebook={NOTEBOOK}",
                                "app.kubernetes.io/managed-by=kjob", None])

    def _cancel(self, **kw):
        a = argparse.Namespace(**{"names": [], "finished": False, "all_mine": False, "all_notebooks": False, **kw})
        jobs = [job("a", NOTEBOOK, "Succeeded"), job("b", NOTEBOOK, "Running")]
        with mock.patch.object(core, "current_namespace", return_value=NS), \
                mock.patch.object(core, "current_notebook", return_value=NOTEBOOK), \
                mock.patch.object(core, "list_jobs", return_value=jobs) as lj, \
                mock.patch.object(core, "delete_job", return_value=True) as dj, \
                mock.patch("builtins.print"):
            cli.cmd_cancel(a)
        return lj.call_args.kwargs["notebook"], [c.args[1] for c in dj.call_args_list]

    def test_cancel_finished_this_notebook(self):
        self.assertEqual(self._cancel(finished=True), (NOTEBOOK, ["a"]))

    def test_cancel_all_mine_all_notebooks(self):
        self.assertEqual(self._cancel(all_mine=True, all_notebooks=True), (None, ["a", "b"]))

    def test_aliases(self):
        for alias, cmd in [("ksub", "submit"), ("ksrun", "run"), ("kqueue", "queue"), ("kclean", "clean")]:
            with mock.patch.object(cli, "main", return_value=0) as m, mock.patch.object(cli.sys, "argv", [alias, "-x"]):
                getattr(cli, alias)()
            m.assert_called_once_with([cmd, "-x"])

    def test_clean_parses(self):
        with mock.patch.object(core, "_load_config"), mock.patch.object(cli, "cmd_cancel", return_value=0) as c:
            cli.main(["clean", "--all-notebooks"])
        self.assertTrue(c.call_args.args[0].all_notebooks)


if __name__ == "__main__":
    unittest.main()
