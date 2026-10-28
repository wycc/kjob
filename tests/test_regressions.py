"""Regression tests for submission failures and job/Pod replacement races."""
import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from kubernetes import client
from kjob import cli, core
from test_kjob import notebook_pod, PODDEFAULTS, PROC_NFS, template, manifest, pod_spec


def load_template(pod=None, defaults=None, mounts=None):
    c, o = mock.Mock(), mock.Mock()
    c.read_namespaced_pod.return_value = pod or notebook_pod()
    o.list_namespaced_custom_object.return_value = defaults if defaults is not None else PODDEFAULTS
    with mock.patch.object(core, '_load_config'), mock.patch.object(core, 'current_namespace', return_value='demo-ns'), \
         mock.patch.object(core, 'current_notebook', return_value='notebook-0'), \
         mock.patch.object(core, '_nfs_mounts', return_value=mounts if mounts is not None else PROC_NFS), \
         mock.patch.object(core.os, 'getuid', return_value=0), \
         mock.patch.object(core.client, 'CoreV1Api', return_value=c), \
         mock.patch.object(core.client, 'CustomObjectsApi', return_value=o):
        return core.NotebookTemplate.from_current_pod()


def handle(**kw):
    with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
        j = core.KJob(core.JobSpec(name='review-job', script='/home/jovyan/train.py', **kw), template())
    return j


def pod(uid='old', phase='Running', code=None):
    return client.V1Pod(metadata=client.V1ObjectMeta(owner_references=[client.V1OwnerReference(
        api_version='kubeflow.org/v1', kind='PyTorchJob', name='review-job', uid=uid)]),
        spec=client.V1PodSpec(containers=[], node_name='gpu-node'), status=client.V1PodStatus(
            phase=phase, container_statuses=[] if code is None else [client.V1ContainerStatus(
                name='pytorch', image='test', image_id='test', ready=False, restart_count=0,
                state=client.V1ContainerState(terminated=client.V1ContainerStateTerminated(exit_code=code)))]))


class StorageTests(unittest.TestCase):
    def test_readonly_pvc_and_mount_stay_readonly(self):
        for source_only in (False, True):
            p = notebook_pod()
            p.spec.volumes[0].persistent_volume_claim.read_only = source_only
            p.spec.containers[0].volume_mounts[0].read_only = not source_only
            t = load_template(p)
            self.assertTrue(t.volumes[0].nfs.read_only)
            self.assertTrue(t.volume_mounts[0].read_only)

    def test_nfs_subpath_is_rejected_before_wrong_export_is_mounted(self):
        p = notebook_pod()
        p.spec.containers[0].volume_mounts[0].sub_path = 'project'
        with self.assertRaisesRegex(ValueError, 'subPath'):
            load_template(p)

    def test_nonmatching_poddefault_does_not_remove_dataset(self):
        pd = {'items': [{'spec': {'selector': {'matchLabels': {'dataset': 'enabled'}},
                                  'volumes': [{'name': 'datasets'}]}}]}
        t = load_template(defaults=pd)
        self.assertIn('datasets', [v.name for v in t.volumes])

    def test_matching_selector_is_carried_to_job(self):
        p = notebook_pod()
        p.metadata.labels = {'dataset': 'enabled', 'unrelated': 'do-not-copy'}
        pd = {'items': [{'spec': {'selector': {'matchLabels': {'dataset': 'enabled'}},
                                  'volumes': [{'name': 'datasets'}]}}]}
        t = load_template(p, pd)
        self.assertNotIn('datasets', [v.name for v in t.volumes])
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            m = core.KJob(core.JobSpec(name='t', script='train.py'), t).manifest()
        self.assertEqual(m['spec']['pytorchReplicaSpecs']['Master']['template']['metadata']['labels']['dataset'], 'enabled')
        self.assertNotIn('unrelated', t.job_labels)

    def test_container_identity_is_preserved_without_extra_capabilities(self):
        p = notebook_pod()
        p.spec.containers[0].security_context = client.V1SecurityContext(
            run_as_user=1000, run_as_group=1000, run_as_non_root=True, privileged=True)
        t = load_template(p)
        self.assertEqual(t.effective_uid, 1000)
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            m = pod_spec(core.KJob(core.JobSpec(name='t', script='train.py'), t).manifest())
        self.assertEqual(m['containers'][0]['securityContext'], {'runAsUser': 1000, 'runAsGroup': 1000, 'runAsNonRoot': True})

    def test_nonroot_mps_is_rejected_instead_of_changing_uid(self):
        t = template()
        t.effective_uid = 1000
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            j = core.KJob(core.JobSpec(name='t', script='train.py', gpu_types=['a30-shared']), t)
        with self.assertRaisesRegex(ValueError, 'UID 0'):
            j.manifest()

    def test_proc_mounts_decodes_spaces(self):
        with mock.patch.object(core.Path, 'read_text', return_value='server:/export\\040space /mnt/a\\040b nfs4 rw 0 0\n'):
            self.assertEqual(core._nfs_mounts(), {'/mnt/a b': ('server', '/export space')})

    def test_script_on_emptydir_shadow_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root, 'tmp', 'train.sh')
            script.parent.mkdir()
            script.write_text('echo ok')
            j = handle(gpus=0)
            j.spec.script = str(script)
            j.spec.workdir = root
            j.spec.log_dir = root
            j.t.volumes = [client.V1Volume(name='home', nfs=client.V1NFSVolumeSource(server='s', path='/p')),
                           client.V1Volume(name='tmp', empty_dir=client.V1EmptyDirVolumeSource())]
            j.t.volume_mounts = [client.V1VolumeMount(name='home', mount_path=root),
                                client.V1VolumeMount(name='tmp', mount_path=str(script.parent))]
            with self.assertRaisesRegex(ValueError, 'not on a shared volume'):
                j._warn_unshared_paths()


class SubmissionTests(unittest.TestCase):
    def test_default_request_fits_teaching_quota(self):
        s = core.JobSpec(name='t', script='train.py')
        self.assertEqual((s.cpus, s.memory, s.shm), (1, '2Gi', None))

    def test_invalid_resources_and_mixed_sharing_fail_early(self):
        for values in ({'gpus': -1}, {'cpus': 0}, {'ttl_seconds': -1}, {'memory': 'nan'}, {'memory': 'bogus'},
                       {'shm': '0'}, {'gpus': 2, 'gpu_types': ['a30-shared']},
                       {'gpu_types': ['a30', 'a30-shared']}, {'gpus': 0, 'gpu_types': ['a30']}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                core.JobSpec(name='t', script='train.py', **values).validate()

    def test_reserved_gpu_runtime_environment_is_rejected(self):
        for key in ('NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY'):
            with self.assertRaises(ValueError):
                core.JobSpec(name='t', script='train.py', gpus=0, env={key: 'all'}).validate()

    def test_cpu_user_environment_is_supported(self):
        m = pod_spec(manifest(gpus=0, env={'EXPERIMENT': '42'}))
        self.assertIn({'name': 'EXPERIMENT', 'value': '42'}, m['containers'][0]['env'])

    def test_shared_request_has_mps_affinity(self):
        m = pod_spec(manifest(gpu_types=['a30-shared']))
        terms = m['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms'][0]['matchExpressions']
        self.assertIn({'key': core.SHARING_STRATEGY_LABEL, 'operator': 'In', 'values': ['mps']}, terms)

    def test_pinned_node_permission_failure_falls_back_to_published_status(self):
        j = handle(node='mps-node')
        j.core.read_node.side_effect = core.ApiException(status=403)
        snap = {'nodes': [{'name': 'mps-node', 'product': 'NVIDIA-A30-SHARED', 'sharing': 'mps'}]}
        with mock.patch.object(core.cluster, 'load_snapshot', return_value=snap):
            self.assertEqual(j._node_labels('mps-node')[core.SHARING_STRATEGY_LABEL], 'mps')
        with mock.patch.object(core.cluster, 'load_snapshot', side_effect=core.cluster.ClusterViewUnavailable('x')), \
                mock.patch('sys.stderr', new_callable=io.StringIO) as err:
            self.assertEqual(j._node_labels('mps-node'), {})
        self.assertIn('whole GPUs', err.getvalue())

    def test_authorized_pinned_mps_node_removes_shm(self):
        j = handle(node='mps-node')
        j.core.read_node.return_value = client.V1Node(metadata=client.V1ObjectMeta(labels={
            core.SHARING_STRATEGY_LABEL: 'mps', 'nvidia.com/gpu.product': 'NVIDIA-A30-SHARED'}))
        j.core.list_namespaced_resource_quota.return_value.items = []
        j.custom.create_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        with mock.patch.object(j, '_warn_unshared_paths'):
            j.submit()
        m = pod_spec(j.custom.create_namespaced_custom_object.call_args.args[-1])
        self.assertNotIn('/dev/shm', [vm['mountPath'] for vm in m['containers'][0]['volumeMounts']])
        self.assertEqual(j.uid, 'new')

    def test_impossible_quota_is_rejected_before_post(self):
        j = handle(memory='32Gi')
        j.core.list_namespaced_resource_quota.return_value.items = [client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name='q'), status=client.V1ResourceQuotaStatus(hard={'requests.memory': '18000Mi'}))]
        with self.assertRaisesRegex(ValueError, 'waiting cannot satisfy'):
            j.submit()
        j.custom.create_namespaced_custom_object.assert_not_called()

    def test_temporarily_full_quota_still_allows_operator_retry(self):
        j = handle()
        j.core.list_namespaced_resource_quota.return_value.items = [client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name='q'), status=client.V1ResourceQuotaStatus(
                hard={'requests.nvidia.com/gpu': '1'}, used={'requests.nvidia.com/gpu': '1'}))]
        j.custom.create_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        with mock.patch.object(j, '_warn_unshared_paths'):
            j.submit()
        j.custom.create_namespaced_custom_object.assert_called_once()

    def test_auto_names_are_unique_and_keep_suffix_for_long_stems(self):
        names = {core.unique_name('x' * 100) for _ in range(50)}
        self.assertEqual(len(names), 50)
        self.assertTrue(all(len(n) <= 50 for n in names))

    def test_malformed_env_is_user_error(self):
        p = argparse.ArgumentParser()
        cli._add_job_opts(p)
        with self.assertRaisesRegex(ValueError, 'KEY=VALUE'):
            cli._spec(p.parse_args(['--env', 'broken', 'train.py']))


class LifecycleTests(unittest.TestCase):
    def test_false_success_condition_does_not_mark_running_job_finished(self):
        self.assertEqual(core.job_state({'status': {'conditions': [
            {'type': 'Running', 'status': 'True'}, {'type': 'Succeeded', 'status': 'False'}]}}), 'Running')

    def test_replacement_job_is_not_followed(self):
        j = handle()
        j.uid = 'old'
        j.custom.get_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        self.assertEqual(j.state(), 'Deleted')
        self.assertEqual(j.uid, 'old')

    def test_replacement_pod_is_not_read(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod(uid='new')
        with self.assertRaises(core.ApiException):
            list(j.stream_logs())
        j.core.read_namespaced_pod_log.assert_not_called()

    def test_delete_has_uid_precondition(self):
        with mock.patch.object(core.client, 'CustomObjectsApi') as api:
            self.assertTrue(core.delete_job('demo-ns', 'job', uid='old'))
        body = api.return_value.delete_namespaced_custom_object.call_args.kwargs['body']
        self.assertEqual(body.preconditions.uid, 'old')

    def test_replacement_delete_conflict_is_safe(self):
        with mock.patch.object(core.client, 'CustomObjectsApi') as api:
            api.return_value.delete_namespaced_custom_object.side_effect = core.ApiException(status=409)
            self.assertFalse(core.delete_job('demo-ns', 'job', uid='old'))

    def test_deleted_while_streaming_ends_without_traceback(self):
        j = handle()
        j.uid = 'old'
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.core.read_namespaced_pod.return_value = pod()
        j.stream_logs = mock.Mock(side_effect=core.ApiException(status=404))
        j.state = mock.Mock(return_value='Deleted')
        j.exit_code = mock.Mock(return_value=None)
        self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 1)

    def test_completed_before_follow_still_prints_output(self):
        j = handle()
        j.uid = 'old'
        j.wait_for_pod = mock.Mock(return_value='Succeeded')
        j.core.read_namespaced_pod.return_value = pod(phase='Succeeded', code=0)
        j.stream_logs = mock.Mock(return_value=iter(['result=42', '[kjob now] exit code 0']))
        j.state = mock.Mock(return_value='Succeeded')
        output = io.StringIO()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertIn('result=42', output.getvalue())

    def test_monitor_api_failure_does_not_cancel_training(self):
        j = handle()
        j.submit = mock.Mock()
        j.follow = mock.Mock(side_effect=core.ApiException(status=500))
        j.delete = mock.Mock()
        with self.assertRaises(core.ApiException):
            j.run(report=lambda _: None)
        j.delete.assert_not_called()

    def test_stream_response_is_closed(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod()
        response = j.core.read_namespaced_pod_log.return_value
        response.stream.return_value = [b'hello\n']
        self.assertEqual(list(j.stream_logs()), ['hello'])
        response.close.assert_called_once()

    def test_oom_reason_is_reported(self):
        j = handle()
        j.uid = 'old'
        p = pod(phase='Failed', code=137)
        p.status.container_statuses[0].state.terminated.reason = 'OOMKilled'
        j.core.read_namespaced_pod.return_value = p
        j.wait_for_pod = mock.Mock(return_value='Failed')
        j.stream_logs = mock.Mock(return_value=iter([]))
        j.state = mock.Mock(return_value='Failed')
        msgs = []
        self.assertEqual(j.follow(out=io.StringIO(), report=msgs.append), 137)
        self.assertIn('OOMKilled', msgs[-1])

    def test_api_connection_loss_is_a_clean_error(self):
        from urllib3.exceptions import MaxRetryError
        with mock.patch.object(core, '_load_config'), \
                mock.patch.object(cli, 'cmd_queue', side_effect=MaxRetryError(None, '/api', 'refused')), \
                mock.patch('sys.stderr', new_callable=io.StringIO) as err, self.assertRaises(SystemExit) as ex:
            cli.main(['queue'])
        self.assertEqual(ex.exception.code, 1)
        self.assertIn('keep running', err.getvalue())

    def test_ctrl_c_logs_does_not_dump_entire_saved_log(self):
        j = handle()
        j.state = mock.Mock(return_value='Running')
        j.stream_logs = mock.Mock(side_effect=KeyboardInterrupt)
        with mock.patch.object(cli, '_job_for', return_value=j):
            self.assertEqual(cli.cmd_logs(argparse.Namespace(name='job', follow=True, log_dir='/tmp')), 130)


def manifest_from(t, **kw):
    with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
        return core.KJob(core.JobSpec(name='t', script='/home/jovyan/train.py', **kw), t).manifest()


class InheritanceTests(unittest.TestCase):
    """Notebook settings other users may rely on (found in review: only the author's setup was exercised)."""

    def notebook(self):
        p = notebook_pod()
        main = p.spec.containers[0]
        main.env = [client.V1EnvVar(name='NB_PREFIX', value='/notebook/x'),
                    client.V1EnvVar(name='API_TOKEN', value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(name='tok', key='t'))),
                    client.V1EnvVar(name='SHARED_DIR', value='/nfs'),                  # PodDefault adds it again
                    client.V1EnvVar(name='CUDA_VISIBLE_DEVICES', value='0')]           # kjob decides this one
        main.env_from = [client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name='app-cfg')),
                         client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name='pd-secret'))]
        p.spec.runtime_class_name = 'nvidia'
        p.spec.tolerations = [
            client.V1Toleration(key='node.kubernetes.io/not-ready', operator='Exists', effect='NoExecute', toleration_seconds=300),
            client.V1Toleration(key='dedicated', operator='Equal', value='team-a', effect='NoSchedule'),
            client.V1Toleration(key='example.com/gpu-access', operator='Equal', value='enabled', effect='NoSchedule')]
        return p

    DEFAULTS = {'items': PODDEFAULTS['items'] + [{'spec': {
        'env': [{'name': 'SHARED_DIR', 'value': '/nfs'}], 'envFrom': [{'secretRef': {'name': 'pd-secret'}}],
        'tolerations': [{'key': 'example.com/gpu-access', 'operator': 'Equal', 'value': 'enabled', 'effect': 'NoSchedule'}]}}]}

    def test_env_envfrom_runtime_class_and_tolerations_are_inherited(self):
        m = manifest_from(load_template(self.notebook(), self.DEFAULTS), gpus=0)
        spec = pod_spec(m)
        c = spec['containers'][0]
        env = {e['name']: e for e in c['env']}
        self.assertEqual(env['NB_PREFIX']['value'], '/notebook/x')
        self.assertEqual(env['API_TOKEN']['valueFrom']['secretKeyRef']['name'], 'tok')
        self.assertNotIn('SHARED_DIR', env)                                   # left to the PodDefault
        self.assertEqual([e['name'] for e in c['env']].count('CUDA_VISIBLE_DEVICES'), 1)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES']['value'], '-1')           # CPU job still hides GPUs
        self.assertEqual(c['envFrom'], [{'configMapRef': {'name': 'app-cfg'}}])
        self.assertEqual(spec['runtimeClassName'], 'nvidia')
        self.assertEqual(spec['tolerations'], [{'key': 'dedicated', 'operator': 'Equal', 'value': 'team-a',
                                                'effect': 'NoSchedule'}])

    def test_user_env_overrides_inherited_env(self):
        c = pod_spec(manifest_from(load_template(self.notebook(), self.DEFAULTS), env={'NB_PREFIX': 'mine'}))['containers'][0]
        self.assertEqual([e['value'] for e in c['env'] if e['name'] == 'NB_PREFIX'], ['mine'])

    def test_main_container_is_not_guessed_from_order(self):
        p = notebook_pod()
        helper = client.V1Container(name='helper', image='helper:1')
        p.spec.containers.insert(0, helper)
        with self.assertRaisesRegex(ValueError, 'container'):
            load_template(p)                                                   # two candidates, nothing says which
        p.metadata.annotations = {'kubectl.kubernetes.io/default-container': 'training'}
        self.assertEqual(load_template(p).image, notebook_pod().spec.containers[0].image)
        p.metadata.annotations = None
        p.metadata.labels = {'notebook-name': 'training'}
        self.assertEqual(load_template(p).image, notebook_pod().spec.containers[0].image)

    def test_unreadable_poddefaults_keep_shared_mounts(self):
        for status in (403, 404):
            c, o = mock.Mock(), mock.Mock()
            c.read_namespaced_pod.return_value = notebook_pod()
            o.list_namespaced_custom_object.side_effect = core.ApiException(status=status)
            with mock.patch.object(core, '_load_config'), mock.patch.object(core, 'current_namespace', return_value='mtest'), \
                 mock.patch.object(core, 'current_notebook', return_value='notebook-0'), \
                 mock.patch.object(core, '_nfs_mounts', return_value=PROC_NFS), \
                 mock.patch.object(core.os, 'getuid', return_value=0), \
                 mock.patch.object(core.client, 'CoreV1Api', return_value=c), \
                 mock.patch.object(core.client, 'CustomObjectsApi', return_value=o):
                t = core.NotebookTemplate.from_current_pod()
            names = {v.name for v in t.volumes}
            self.assertTrue({'gs-group-a', 'gs-group-a-7ef7d056', 'ns-demo-user'} <= names, status)
            self.assertFalse(t.poddefaults_known)


    def test_poddefault_mounts_count_as_shared(self):
        """A notebook whose only shared storage is a PodDefault mount (e.g. the namespace share) can submit."""
        t = load_template()
        self.assertIn('/mnt/namespaces/demo-user', t.shared_paths())
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            j = core.KJob(core.JobSpec(name='t', script='bash', workdir='/mnt/namespaces/demo-user',
                                       log_dir='/mnt/namespaces/demo-user/logs'), t)
            j._warn_unshared_paths()                                           # no error
        pd = {'items': [{'spec': {'volumes': [{'name': 'ns-demo-user', 'nfs': {'server': 's', 'path': '/p', 'readOnly': True}}]}}]}
        p = notebook_pod()
        p.spec.volumes[4].nfs.read_only = True                                 # ns-demo-user, read-only share
        t = load_template(p, pd)
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            j = core.KJob(core.JobSpec(name='t', script='bash', workdir='/home/jovyan',
                                       log_dir='/mnt/namespaces/demo-user/logs'), t)
            with self.assertRaisesRegex(ValueError, 'read-only'):
                j._warn_unshared_paths()


class AccountingTests(unittest.TestCase):
    def test_resubmitted_job_does_not_take_the_old_result(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'j.log'
            hist = Path(d) / 'history.jsonl'
            old = {'name': 'j', 'submitted': '2099-01-01T00:00:00Z', 'log': str(log), 'gpus': 1, 'run_id': 'aaa'}
            new = {**old, 'submitted': '2099-01-01T00:00:25Z', 'run_id': 'bbb'}       # resubmitted 20 s after start
            hist.write_text(core.json.dumps(old) + '\n' + core.json.dumps(new) + '\n')
            log.write_text('[kjob 2099-01-01T00:00:05Z] start j on node n run aaa\n'
                           '[kjob 2099-01-01T00:00:06Z] exit code 0\n')
            out = io.StringIO()
            with mock.patch.object(core, 'HISTORY_FILE', hist), \
                 mock.patch.object(core, 'list_jobs', return_value=[{'metadata': {'name': 'j'}}]), \
                 mock.patch.object(core, 'job_state', return_value='Created'), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=20))
            rows = [l.split() for l in out.getvalue().splitlines()[1:]]
            self.assertEqual([r[5] for r in rows], ['COMPLETED', 'PENDING'])
            self.assertEqual(core.log_result(str(log), 'bbb'), (None, None, None))     # kshow of the new one
            self.assertEqual(core.log_result(str(log), 'aaa')[2], 0)

    def test_records_from_older_kjob_still_match_by_time(self):
        with tempfile.TemporaryDirectory() as d:
            log, hist = Path(d) / 'j.log', Path(d) / 'history.jsonl'
            hist.write_text(core.json.dumps({'name': 'j', 'submitted': '2099-01-01T00:00:00Z', 'log': str(log), 'gpus': 1}) + '\n')
            log.write_text('[kjob 2099-01-01T00:00:05Z] start j on node n\n[kjob 2099-01-01T00:00:06Z] exit code 2\n')
            out = io.StringIO()
            with mock.patch.object(core, 'HISTORY_FILE', hist), mock.patch.object(core, 'list_jobs', return_value=[]), \
                 mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=20))
            self.assertIn('FAILED', out.getvalue())

    def test_job_prints_and_records_its_run_id(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(core.client, 'CoreV1Api'), \
                mock.patch.object(core.client, 'CustomObjectsApi'):
            j = core.KJob(core.JobSpec(name='j', script='/home/jovyan/train.py', log_dir=d), template())
            shell = pod_spec(j.manifest())['containers'][0]['command'][2]
            self.assertIn(f'run {j.run_id}', shell)
            with mock.patch.object(core, 'HISTORY_FILE', Path(d) / 'h.jsonl'):
                core.record_history(j)
                self.assertEqual(core.read_history()[0]['run_id'], j.run_id)


def run_main(exc, cmd='queue'):
    """cli.main with the command raising exc; returns (exit code, stderr)."""
    err = io.StringIO()
    def boom(a):
        raise exc
    with mock.patch.object(cli, 'cmd_queue', boom), mock.patch.object(cli, 'cmd_submit', boom), \
         mock.patch.object(cli, 'auto_install_skills'), mock.patch.object(core, '_load_config'), mock.patch('sys.stderr', err):
        try:
            code = cli.main([cmd] + (['x.py'] if cmd == 'submit' else []))
        except SystemExit as e:
            code = e.code
    return code, err.getvalue()


def api_error(status, message, reason='Forbidden'):
    e = core.ApiException(status=status, reason=reason)
    e.body = core.json.dumps({'kind': 'Status', 'message': message, 'reason': reason, 'code': status})
    return e


class ErrorOutputTests(unittest.TestCase):
    def test_forbidden_shows_kubernetes_reason_and_what_to_do(self):
        code, err = run_main(api_error(403, 'pytorchjobs.kubeflow.org is forbidden: User "system:serviceaccount:me:default" '
                                            'cannot create resource "pytorchjobs" in API group "kubeflow.org" in the namespace "me"'))
        self.assertEqual(code, 1)
        self.assertIn('cannot create resource "pytorchjobs"', err)                # what is missing, and for whom
        self.assertIn('default-editor', err)                                       # how a Notebook normally gets it
        self.assertIn('ADMIN.md', err)

    def test_quota_rejection_is_not_reported_as_a_permission_problem(self):
        code, err = run_main(api_error(403, 'exceeded quota: kf-resource-quota, requested: requests.nvidia.com/gpu=4'))
        self.assertIn('exceeded quota', err)
        self.assertNotIn('default-editor', err)

    def test_other_api_errors_keep_the_message(self):
        code, err = run_main(api_error(422, 'PyTorchJob.kubeflow.org "x" is invalid: spec.foo: Invalid value', 'Invalid'))
        self.assertIn('422', err)
        self.assertIn('is invalid: spec.foo', err)

    def test_unreachable_api_mentions_sandbox_and_does_not_claim_a_submission(self):
        code, err = run_main(core_http_error(), cmd='submit')
        self.assertEqual(code, 1)
        self.assertIn('cannot reach the Kubernetes API', err)
        self.assertIn('sandbox', err)
        self.assertIn('kqueue', err)                                               # how to check whether it got in

    def test_file_permission_error_names_the_path(self):
        code, err = run_main(PermissionError(13, 'Permission denied', '/home/jovyan/kjob-logs'))
        self.assertEqual(code, 1)
        self.assertIn('/home/jovyan/kjob-logs', err)
        self.assertNotIn('Traceback', err)

    def test_refusals_print_the_reason_without_the_usage_text(self):
        code, err = run_main(ValueError('Namespace me has no access to the requested GPUs on gpu-node-restricted'), cmd='submit')
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith('kjob: Namespace me has no access'), err)
        self.assertNotIn('usage:', err)

    def test_unexpected_error_is_one_line_unless_debugging(self):
        code, err = run_main(RuntimeError('boom'))
        self.assertEqual(code, 1)
        self.assertIn('RuntimeError: boom', err)
        self.assertIn('KJOB_DEBUG=1', err)
        with mock.patch.dict(cli.os.environ, {'KJOB_DEBUG': '1'}), self.assertRaises(RuntimeError):
            run_main(RuntimeError('boom'))


def core_http_error():
    from urllib3.exceptions import MaxRetryError
    return MaxRetryError(None, '/apis/kubeflow.org/v1/namespaces/me/pytorchjobs', 'Connection refused')


class RestrictedAccessTests(unittest.TestCase):
    SNAP = {'nodes': [
        {'name': 'gpu-node-open', 'product': 'Tesla-V100-SXM2-16GB', 'restricted': [], 'sharing': '', 'gpus': 8},
        {'name': 'gpu-node-restricted', 'product': 'NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition',
         'restricted': ['example.com/gpu-access=enabled'], 'sharing': '', 'gpus': 8}]}
    RTX = 'NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition'

    def check(self, types, node=None, tol=(), exclude=()):
        with mock.patch.object(cluster_mod(), 'load_snapshot', return_value=self.SNAP), \
             mock.patch.object(cli, '_my_tolerations', return_value=list(tol)), \
             mock.patch.object(core, 'current_namespace', return_value='me'):
            cli._check_access(types, node, list(exclude), 1)

    def test_restricted_type_without_access_is_refused_before_submitting(self):
        with self.assertRaisesRegex(ValueError, r'(?s)no access.*gpu-node-restricted.*administrator'):
            self.check([self.RTX])
        with self.assertRaisesRegex(ValueError, 'no access'):
            self.check([], node='gpu-node-restricted')

    def test_allowed_or_mixed_requests_pass(self):
        self.check([self.RTX], tol=[{'key': 'example.com/gpu-access', 'operator': 'Equal', 'value': 'enabled', 'effect': 'NoSchedule'}])
        self.check(['v100', self.RTX])                       # V100 is usable, so the job can start
        self.check([])                                       # any GPU
        with self.assertRaisesRegex(ValueError, 'no access'):
            self.check(['v100', self.RTX], exclude=['gpu-node-open'])

    def test_unknown_tolerations_skip_the_check(self):
        with mock.patch.object(cluster_mod(), 'load_snapshot', return_value=self.SNAP), \
             mock.patch.object(cli, '_my_tolerations', return_value=None):
            cli._check_access([self.RTX], None, [], 1)

    def test_notebook_tolerations_count(self):
        p = notebook_pod()
        p.spec.tolerations = [client.V1Toleration(key='example.com/gpu-access', operator='Equal', value='enabled', effect='NoSchedule'),
                              client.V1Toleration(key='node.kubernetes.io/not-ready', operator='Exists', effect='NoExecute')]
        c, o = mock.Mock(), mock.Mock()
        c.read_namespaced_pod.return_value = p
        o.list_namespaced_custom_object.return_value = {'items': []}
        with mock.patch.object(core.client, 'CoreV1Api', return_value=c), mock.patch.object(core.client, 'CustomObjectsApi', return_value=o), \
             mock.patch.object(core, 'current_namespace', return_value='me'), mock.patch.object(core, 'current_notebook', return_value='nb-0'):
            tol = cli._my_tolerations()
        self.assertIn({'key': 'example.com/gpu-access', 'operator': 'Equal', 'value': 'enabled', 'effect': 'NoSchedule'}, tol)
        self.assertFalse(any((t.get('key') or '').startswith('node.kubernetes.io/') for t in tol))


def cluster_mod():
    from kjob import cluster
    return cluster


if __name__ == '__main__':
    unittest.main()
