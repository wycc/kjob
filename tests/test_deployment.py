"""Deployment regressions: real shell failures plus isolated Kubernetes lifecycle cases."""
import argparse
import io
import json
import os
import shlex
import subprocess
import tempfile
import queue
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from kubernetes import client
import urllib3
from kjob import cli, cluster, core
from kjob._output import decode_pod_output
from test_audit_fixes import fixture
from test_regressions import handle, pod, load_template, manifest_from
from test_kjob import pod_spec


class LogStorageTests(unittest.TestCase):
    def test_progress_is_line_framed_for_the_runtime_but_saved_bytes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            log, term = Path(d)/'job.log', Path(d)/'termination.json'
            command = "bash -c 'printf \"\\r10%%\\r20%%\\r\\ncomplete\\n\"'"
            shell = core.job_shell('test', command, d, str(log), 'abc123', str(term), live_progress=True)
            p = subprocess.run(['bash','-c',shell], capture_output=True, timeout=10,
                               env={**os.environ,'TMPDIR':d})
            self.assertEqual(p.returncode, 0)
            self.assertIn(b'\r10%\r20%\r\ncomplete\n', log.read_bytes())
            pmeta = pod()
            pmeta.metadata.annotations = {'kjob/log-stream': 'abc123'}
            self.assertIn('\r10%\r20%\r\ncomplete\n', decode_pod_output(p.stdout.decode(), pmeta))

    def shell(self, directory, log, command):
        term = Path(directory) / 'termination.json'
        shell = core.job_shell('test', command, directory, str(log), 'abc123', str(term))
        p = subprocess.run(['bash', '-c', shell], capture_output=True, text=True, timeout=10,
                           env={**os.environ, 'TMPDIR': directory})
        return p, json.loads(term.read_text())

    def test_enospc_is_not_success_and_kubelet_gets_independent_result(self):
        for script_rc in (0, 7):
            with self.subTest(script_rc=script_rc), tempfile.TemporaryDirectory() as d:
                p, result = self.shell(d, '/dev/full', f"bash -c 'echo output; exit {script_rc}'")
                self.assertEqual(p.returncode, script_rc or 74)
                self.assertEqual(result, {'kjob': 1, 'run_id': 'abc123', 'log_saved': False,
                                          'script_exit_code': script_rc})
                self.assertIn('log write failed:', p.stdout)
                self.assertIn('output', p.stdout)

    def test_unwritable_log_never_starts_user_script(self):
        with tempfile.TemporaryDirectory() as d:
            blocked = Path(d) / 'blocked'
            blocked.write_text('file')
            marker = Path(d) / 'should-not-run'
            p, result = self.shell(d, blocked / 'test.log', 'touch ' + shlex.quote(str(marker)))
            self.assertEqual(p.returncode, 74)
            self.assertFalse(marker.exists())
            self.assertFalse(result['log_saved'])
            self.assertIsNone(result['script_exit_code'])

    def test_success_and_script_failure_keep_complete_unicode_output_with_spaces(self):
        for rc in (0, 9):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory(prefix='kjob spaces ') as d:
                log = Path(d) / 'logs with spaces' / 'test.log'
                p, result = self.shell(d, log, f"bash -c 'printf \"測試結果\\n\"; exit {rc}'")
                self.assertEqual(p.returncode, rc)
                self.assertTrue(result['log_saved'])
                self.assertEqual(result['script_exit_code'], rc)
                self.assertIn('測試結果', log.read_text())
                self.assertEqual(core.log_result(str(log), 'abc123')[2], rc)


class MonitoringTests(unittest.TestCase):
    def test_http_progress_is_visible_while_the_connection_is_still_open(self):
        for chunked in (True, False, 'slow-chunk'):
            with self.subTest(chunked=chunked):
                release = threading.Event()
                class Handler(BaseHTTPRequestHandler):
                    protocol_version = 'HTTP/1.1'
                    def log_message(self, *args):
                        pass
                    def do_GET(self):
                        self.send_response(200)
                        self.send_header('Transfer-Encoding' if chunked else 'Connection',
                                         'chunked' if chunked else 'close')
                        self.end_headers()
                        self.wfile.write(b'8\r\n1%\r' if chunked == 'slow-chunk'
                                         else b'3\r\n1%\r\r\n' if chunked else b'1%\r')
                        self.wfile.flush()
                        release.wait(5)
                        self.wfile.write(b'done\n\r\n0\r\n\r\n' if chunked == 'slow-chunk'
                                         else b'5\r\ndone\n\r\n0\r\n\r\n' if chunked else b'done\n')
                        self.wfile.flush()
                        self.close_connection = True
                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
                serving.start()
                pool = urllib3.HTTPConnectionPool('127.0.0.1', server.server_port, timeout=3)
                response = pool.request('GET', '/', preload_content=False)
                j = handle()
                j.core.read_namespaced_pod.return_value = pod()
                j.core.read_namespaced_pod_log.return_value = response
                received = queue.Queue()
                def consume():
                    try:
                        for update in j.stream_logs():
                            received.put(update)
                    except Exception as error:
                        received.put(error)
                consumer = threading.Thread(target=consume, daemon=True)
                consumer.start()
                try:
                    self.assertEqual(received.get(timeout=2), '1%\r')
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                    consumer.join(4)
                    server.shutdown()
                    server.server_close()
                    pool.close()
                    serving.join(1)
                self.assertFalse(consumer.is_alive())
                tail = []
                while not received.empty():
                    tail.append(received.get_nowait())
                self.assertTrue(all(isinstance(x, str) for x in tail), tail)
                self.assertEqual(''.join(x if isinstance(x, core.LogFragment) else x+'\n' for x in tail), 'done\n')

    def test_short_carriage_return_updates_arrive_before_the_stream_ends(self):
        j = handle()
        j.core.read_namespaced_pod.return_value = pod()
        response = j.core.read_namespaced_pod_log.return_value
        def chunks():
            yield '進度 1\r'.encode()
            self.assertEqual(output.getvalue(), '進度 1\r')
            yield b'progress 2\r'
            self.assertEqual(output.getvalue(), '進度 1\rprogress 2\r')
            yield b'done\n'
        response.stream.return_value = chunks()
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.state = mock.Mock(return_value='Succeeded')
        j.exit_code = mock.Mock(return_value=0)
        j.stop_reason = mock.Mock(return_value=None)
        output = io.StringIO()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertEqual(output.getvalue(), '進度 1\rprogress 2\rdone\n')

    def test_tqdm_leading_carriage_return_does_not_wait_for_the_next_update(self):
        j = self.finished(code=0, log_saved=True)
        del j.stream_logs  # exercise the real reader and follow renderer
        output = io.StringIO()
        def chunks():
            yield b'\r10%'
            self.assertEqual(output.getvalue(), '\r10%')
            yield b'\r20'
            self.assertEqual(output.getvalue(), '\r10%\r20')
            yield b'%\n'
        j.core.read_namespaced_pod_log.return_value.stream.return_value = chunks()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertEqual(output.getvalue(), '\r10%\r20%\n')

    def test_klogs_follow_preserves_carriage_returns(self):
        j = handle()
        j.state = mock.Mock(return_value='Running')
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.stream_logs = mock.Mock(return_value=iter(['10%\r', '100%\r', '']))
        output = io.StringIO()
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', output):
            self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=True)), 0)
        self.assertEqual(output.getvalue(), '10%\r100%\r\n')

    def test_crlf_and_split_utf8_remain_intact(self):
        j = handle()
        j.core.read_namespaced_pod.return_value = pod()
        raw = '測試\r\nprogress\rfinal\n'.encode()
        j.core.read_namespaced_pod_log.return_value.stream.return_value = (bytes([b]) for b in raw)
        updates = list(j.stream_logs())
        text = ''.join(x if isinstance(x, core.LogFragment) or x.endswith('\r') else x+'\n' for x in updates)
        self.assertEqual(text, raw.decode())

    def test_newline_free_progress_does_not_buffer_the_entire_job_output(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod()
        text = '進度' * (core.LOG_FRAGMENT_CHARS + 17)
        raw = text.encode()
        # Chunks deliberately split UTF-8 characters; no replacement characters or data loss.
        j.core.read_namespaced_pod_log.return_value.stream.return_value = (
            raw[i:i+1024] for i in range(0, len(raw), 1024))
        fragments = list(j.stream_logs())
        self.assertEqual(''.join(fragments), text)
        self.assertLessEqual(max(map(len, fragments)), core.LOG_FRAGMENT_CHARS)
        j.core.read_namespaced_pod_log.return_value.close.assert_called_once()

    def finished(self, code=74, log_saved=False, state='Failed'):
        j = handle(gpus=0)
        j.uid, j.run_id = 'old', 'abc123'
        p = pod(phase='Failed' if code else 'Succeeded', code=code)
        p.status.container_statuses[0].state.terminated.message = json.dumps(
            {'kjob': 1, 'run_id': 'abc123', 'log_saved': log_saved, 'script_exit_code': 0})
        j.core.read_namespaced_pod.return_value = p
        j.wait_for_pod = mock.Mock(return_value=p.status.phase)
        j.state = mock.Mock(return_value=state)
        j.stream_logs = mock.Mock(return_value=iter(['[kjob now] exit code 0 run abc123']))
        j.submit, j.delete = mock.Mock(), mock.Mock()
        return j

    def test_ksrun_preserves_failed_log_pod_and_nonzero_exit(self):
        j, messages = self.finished(), []
        self.assertEqual(j.run(out=io.StringIO(), report=messages.append), 74)
        j.delete.assert_not_called()
        self.assertIn('keeping the finished job and Pod', messages[-1])
        self.assertIn('--ttl', messages[-1])
        self.assertEqual(core.read_events()[-1]['run_id'], 'abc123')

    def test_kwait_also_preserves_failed_log_pod(self):
        j = self.finished()
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(cli.cmd_wait(argparse.Namespace(name=j.spec.name, keep=False)), 74)
        j.delete.assert_not_called()

    def test_normal_completion_still_cleans_up(self):
        j = self.finished(code=0, log_saved=True)
        self.assertEqual(j.run(out=io.StringIO(), report=lambda _: None), 0)
        j.delete.assert_called_once()

    def test_main_exit_does_not_wait_for_a_stalled_operator(self):
        for code in (0, 7):
            j = self.finished(code=code, log_saved=True, state='Running')
            messages = []
            with self.subTest(code=code), mock.patch.object(core.time, 'sleep') as sleep:
                self.assertEqual(j.run(out=io.StringIO(), report=messages.append), code)
            sleep.assert_not_called()
            self.assertFalse(any('operator status' in message for message in messages))
            j.delete.assert_called_once()

    def test_kwait_cleans_up_when_main_finished_but_operator_still_running(self):
        j = self.finished(code=0, log_saved=True, state='Running')
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(cli.cmd_wait(argparse.Namespace(name=j.spec.name, keep=False)), 0)
        j.delete.assert_called_once()

    def test_unconfirmed_completion_returns_error_without_cancelling_job(self):
        j = self.finished(state='Running')
        j.core.read_namespaced_pod.return_value = pod()  # main has no termination status
        with mock.patch.object(core, 'STATUS_GRACE_SECONDS', 0):
            self.assertEqual(j.run(out=io.StringIO(), report=lambda _: None), 1)
        j.delete.assert_not_called()

    def test_retry_follow_does_not_keep_an_old_monitoring_error(self):
        j = self.finished(state='Running')
        p = pod()
        j.core.read_namespaced_pod.return_value = p
        with mock.patch.object(core, 'STATUS_GRACE_SECONDS', 0):
            self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 1)
        p.status.container_statuses = self.finished(code=0, log_saved=True).core.read_namespaced_pod.return_value.status.container_statuses
        j.state.return_value = 'Succeeded'
        self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 0)
        self.assertTrue(j.cleanup_finished(report=lambda _: None))
        j.delete.assert_called_once()

    def test_wrong_run_metadata_is_not_used(self):
        j = self.finished()
        j.run_id = 'different'
        self.assertEqual(j.exit_code(), 74)
        self.assertFalse(getattr(j, 'log_failed', False))

    def test_missing_container_diagnostic_goes_to_stderr_only_after_eviction(self):
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        for reason in ('Evicted', None):
            with self.subTest(reason=reason):
                j = self.finished(code=143, log_saved=True)
                j.core.read_namespaced_pod.return_value.status.reason = reason
                j.stream_logs = mock.Mock(return_value=iter(['saved checkpoint', diagnostic]))
                out, messages = io.StringIO(), []
                with mock.patch.object(core, 'EVICTION_GRACE_SECONDS', 0):
                    self.assertEqual(j.follow(out=out, report=messages.append), 143)
                self.assertIn('saved checkpoint', out.getvalue())
                if reason:
                    self.assertNotIn('rpc error:', out.getvalue())
                    self.assertTrue(any('log stream unavailable after eviction' in m for m in messages))
                    self.assertIn('Evicted', messages[-1])
                else:
                    self.assertIn(diagnostic, out.getvalue())

    def test_diagnostic_is_filtered_when_the_evicted_reason_arrives_after_it(self):
        # Seen on a live cluster: Docker's message ends the stream in the same second kubelet evicts the Pod,
        # before the Pod's status says Evicted.
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        j = self.finished(code=137, log_saved=True)
        status = j.core.read_namespaced_pod.return_value.status
        status.reason = None
        j.stream_logs = mock.Mock(return_value=iter(['dd finished', diagnostic]))
        out, messages = io.StringIO(), []
        with mock.patch.object(core.time, 'sleep', lambda _: setattr(status, 'reason', 'Evicted')):
            self.assertEqual(j.follow(out=out, report=messages.append), 137)
        self.assertEqual(out.getvalue(), 'dd finished\n')
        self.assertTrue(any('log stream unavailable after eviction' in m and diagnostic in m for m in messages))

    def test_diagnostic_followed_by_more_output_is_the_scripts_own(self):
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        j = self.finished(code=0, log_saved=True)
        j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
        j.stream_logs = mock.Mock(return_value=iter(['a', diagnostic, 'b']))
        out = io.StringIO()
        j.follow(out=out, report=lambda _: None)
        self.assertEqual(out.getvalue(), f'a\n{diagnostic}\nb\n')

    def test_klogs_keeps_eviction_diagnostics_separate_from_output(self):
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        for follow in (True, False):
            with self.subTest(follow=follow):
                j = self.finished(code=143, log_saved=True)
                j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
                j.stream_logs = mock.Mock(return_value=iter(['checkpoint saved', diagnostic]))
                j.core.read_namespaced_pod_log.return_value = 'checkpoint saved\n'+diagnostic+'\n'
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(cli, '_job_for', return_value=j), \
                        mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
                    self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=follow)), 0)
                self.assertEqual(out.getvalue(), 'checkpoint saved\n')
                self.assertIn(diagnostic, err.getvalue())

    def test_evicted_pod_with_unavailable_logs_still_reports_failure(self):
        for status in (400, 404):
            for code in (0, 143):
                with self.subTest(status=status, code=code):
                    j = self.finished(code=code, log_saved=True)
                    j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
                    j.stream_logs = mock.Mock(side_effect=core.ApiException(status=status))
                    messages = []
                    self.assertEqual(j.follow(out=io.StringIO(), report=messages.append), code or 1)
                    self.assertIn('Evicted', messages[-1])
                    self.assertTrue(any('log stream unavailable after eviction' in m for m in messages))
        j = self.finished(code=143, log_saved=True)
        j.stream_logs = mock.Mock(side_effect=core.ApiException(status=400))
        with self.assertRaises(core.ApiException), mock.patch.object(core, 'EVICTION_GRACE_SECONDS', 0):
            j.follow(out=io.StringIO(), report=lambda _: None)

    def test_log_request_refused_before_the_evicted_reason_arrives(self):
        j = self.finished(code=137, log_saved=True)
        status = j.core.read_namespaced_pod.return_value.status
        status.reason = None
        j.stream_logs = mock.Mock(side_effect=core.ApiException(status=400))
        messages = []
        with mock.patch.object(core.time, 'sleep', lambda _: setattr(status, 'reason', 'Evicted')):
            self.assertEqual(j.follow(out=io.StringIO(), report=messages.append), 137)
        self.assertTrue(any('log stream unavailable after eviction' in m for m in messages))
        self.assertIn('Evicted', messages[-1])

    def test_no_wait_for_an_evicted_reason_when_the_pod_has_not_started(self):
        j = self.finished(code=137, log_saved=True)
        j.core.read_namespaced_pod.return_value.status.phase = 'Pending'
        with mock.patch.object(core.time, 'sleep', side_effect=AssertionError('waited')):
            self.assertFalse(j.is_evicted_log_exception(core.ApiException(status=400)))

    def test_klogs_recovers_saved_bytes_when_evicted_container_is_gone(self):
        for follow in (False, True):
            for exists in (False, True):
                with self.subTest(follow=follow, exists=exists), tempfile.TemporaryDirectory() as d:
                    log = Path(d)/'saved.log'
                    payload = b'\r10%\r100%\nCRLF\r\n'
                    if exists:
                        log.write_bytes(payload)
                    j = self.finished(code=143, log_saved=True)
                    j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
                    j.stream_logs = mock.Mock(side_effect=core.ApiException(status=400))
                    j.core.read_namespaced_pod_log.side_effect = core.ApiException(status=400)
                    j.custom.get_namespaced_custom_object.return_value = {'metadata': {'annotations': {'kjob/log': str(log)}}}
                    out, err = io.StringIO(), io.StringIO()
                    with mock.patch.object(cli, '_job_for', return_value=j), mock.patch.object(core, 'read_history', return_value=[]), \
                            mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
                        rc = cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=follow, log_dir=d))
                    self.assertEqual(rc, 0 if exists else 1)
                    self.assertEqual(out.getvalue(), payload.decode() if exists else '')
                    self.assertIn('after eviction', err.getvalue())
                    self.assertNotIn('has not started', err.getvalue())

    def test_log_error_accounting_is_bound_to_run_not_reused_name(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'same.log'
            log.write_text('[kjob 2026-01-01T00:00:01Z] start same on node n run aaa\n'
                           '[kjob 2026-01-01T00:00:02Z] exit code 0 run aaa\n'
                           '[kjob 2026-01-01T00:01:01Z] start same on node n run bbb\n'
                           '[kjob 2026-01-01T00:01:02Z] exit code 0 run bbb\n')
            history = [{'name': 'same', 'submitted': '2026-01-01T00:00:00Z', 'log': str(log), 'gpus': 0,
                        'run_id': rid} for rid in ('aaa', 'bbb')]
            events = [{'event': 'log-error', 'name': 'same', 'run_id': 'aaa', 'exit_code': 74}]
            out = io.StringIO()
            with mock.patch.object(core, 'read_history', return_value=history), \
                    mock.patch.object(core, 'read_events', return_value=events), \
                    mock.patch.object(core, 'list_jobs', return_value=[]), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=0))
            rows = out.getvalue().splitlines()[1:]
            self.assertEqual([row.split()[5:7] for row in rows], [['LOG_ERROR', '74'], ['COMPLETED', '0']])

    def test_eviction_is_accounted_after_the_job_is_deleted(self):
        # The log alone reads like -t or kcancel (a SIGTERM line, then the script's 143).
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'sc.log'
            log.write_text('[kjob 2026-01-01T00:00:01Z] start sc on node n run aaa\n'
                           '[kjob 2026-01-01T00:00:30Z] stopping: SIGTERM received run aaa\n'
                           '[kjob 2026-01-01T00:00:30Z] exit code 143 run aaa\n')
            history = [{'name': 'sc', 'submitted': '2026-01-01T00:00:00Z', 'log': str(log), 'gpus': 0, 'run_id': 'aaa'}]
            events = [{'event': 'evicted', 'name': 'sc', 'run_id': 'aaa', 'exit_code': 137}]
            out = io.StringIO()
            with mock.patch.object(core, 'read_history', return_value=history), \
                    mock.patch.object(core, 'read_events', return_value=events), \
                    mock.patch.object(core, 'list_jobs', return_value=[]), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=0))
            self.assertEqual(out.getvalue().splitlines()[1].split()[5:7], ['EVICTED', '137'])

    def running_then(self, j, *later):
        """Pod reads: the script's container still runs, then each status in `later`."""
        running = pod()
        running.status.container_statuses = [client.V1ContainerStatus(
            name='pytorch', image='test', image_id='test', ready=True, restart_count=0,
            state=client.V1ContainerState(running=client.V1ContainerStateRunning()))]
        finished = j.core.read_namespaced_pod.return_value
        j.core.read_namespaced_pod.side_effect = [running, running, *later] + [finished] * 20

    def test_stream_ended_while_the_script_runs_is_followed_again(self):
        # kubelet closes a stream idle for hours; the API server restarts. ksrun must not report 1 for a running job.
        for cut in ('ended', 'reset'):
            with self.subTest(cut=cut):
                j = self.finished(code=0, log_saved=True, state='Succeeded')
                self.running_then(j)
                def first():
                    yield 'epoch 1'
                    if cut == 'reset':
                        raise urllib3.exceptions.ProtocolError('Connection broken')
                j.stream_logs = mock.Mock(side_effect=[first(), iter(['epoch 2', '[kjob now] exit code 0 run abc123'])])
                out, messages = io.StringIO(), []
                self.assertEqual(j.follow(out=out, report=messages.append), 0)
                self.assertEqual(out.getvalue(), 'epoch 1\nepoch 2\n')
                self.assertEqual(j.stream_logs.call_count, 2)
                self.assertGreaterEqual(j.stream_logs.call_args_list[1].kwargs['since_seconds'], 1)
                self.assertTrue(any('following again' in m for m in messages))

    def test_no_second_follow_once_the_script_has_ended(self):
        for lines in (['done', '[kjob now] exit code 0 run abc123'], ['killed: no exit line']):
            with self.subTest(lines=lines):
                j = self.finished(code=0, log_saved=True, state='Succeeded')
                j.stream_logs = mock.Mock(return_value=iter(lines))   # the pod's container has terminated
                self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 0)
                j.stream_logs.assert_called_once()

    def test_kwait_records_an_eviction_killed_before_the_wrapper_reported(self):
        # Kubelet SIGKILLs an evicted container (exit 137): no termination message carries the run id, so kwait,
        # which starts from the name only, takes it from the CR.
        j = self.finished(code=137, log_saved=True)
        del j.run_id, j.state                       # as cli._job_for builds it
        j.core.read_namespaced_pod.return_value.status.container_statuses[0].state.terminated.message = None
        j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
        j.custom.get_namespaced_custom_object.return_value = {
            'metadata': {'uid': 'old', 'annotations': {'kjob/run-id': 'abc123'}}}
        j.stream_logs = mock.Mock(return_value=iter([]))
        with mock.patch.object(core, 'job_state', return_value='Failed'), \
                mock.patch.object(core, 'record_event') as rec:
            self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 137)
        rec.assert_called_once_with('evicted', j.spec.name, run_id='abc123', exit_code=137)

    def test_follow_records_the_eviction_for_kacct(self):
        # The recorded code is the one ksrun/kwait return: an evicted script that exited 0 still failed.
        for code, final in ((137, 137), (0, 1)):
            with self.subTest(code=code):
                j = self.finished(code=code, log_saved=True)
                j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
                j.stream_logs = mock.Mock(return_value=iter([]))
                with mock.patch.object(core, 'record_event') as rec:
                    self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), final)
                rec.assert_called_once_with('evicted', j.spec.name, run_id=j.run_id, exit_code=final)

    def accounting_result(self, j, state='Running'):
        obj = {'metadata': {'name': j.spec.name, 'uid': 'old', 'annotations': {'kjob/run-id': 'abc123'}}}
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'job.log'
            log.write_text('[kjob 2026-01-01T00:00:01Z] start job on node n run abc123\n'
                           '[kjob 2026-01-01T00:00:02Z] exit code 0 run abc123\n')
            hist = [{'name': j.spec.name, 'run_id': 'abc123', 'submitted': '2026-01-01T00:00:00Z',
                     'log': str(log), 'gpus': 0}]
            out = io.StringIO()
            with mock.patch.object(core, 'read_history', return_value=hist), \
                    mock.patch.object(core, 'read_events', return_value=[]), \
                    mock.patch.object(core, 'list_jobs', return_value=[obj]), \
                    mock.patch.object(core, 'job_state', return_value=state), \
                    mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=0))
            return out.getvalue().splitlines()[1].split()[5:7]

    def test_background_accounting_checks_wrapper_status_without_kwait(self):
        for code, saved, expected in [(74, False, ['LOG_ERROR', '74']), (0, True, ['COMPLETED', '0'])]:
            with self.subTest(code=code):
                self.assertEqual(self.accounting_result(self.finished(code=code, log_saved=saved)), expected)

    def test_evicted_script_exiting_zero_is_not_accounted_as_completed(self):
        j = self.finished(code=0, log_saved=True)
        j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
        self.assertEqual(self.accounting_result(j), ['EVICTED', '1'])

    def test_background_accounting_does_not_call_log_exit_a_finished_container(self):
        for state, expected in [('Created', 'PENDING'), ('Running', 'RUNNING'), ('Succeeded', 'UNCONFIRMED')]:
            j = self.finished()
            j.core.read_namespaced_pod.return_value = pod()  # no terminated main container yet
            with self.subTest(state=state):
                self.assertEqual(self.accounting_result(j, state), [expected, '-'])

    def test_background_accounting_api_failure_does_not_report_success(self):
        j = self.finished()
        j.core.read_namespaced_pod.side_effect = core.ApiException(status=503)
        self.assertEqual(self.accounting_result(j), ['UNCONFIRMED', '-'])

    def test_background_accounting_does_not_read_a_replacement_pod_result(self):
        j = self.finished()
        j.core.read_namespaced_pod.return_value = pod(uid='new', phase='Succeeded', code=0)
        self.assertEqual(self.accounting_result(j), ['RUNNING', '-'])  # CR state only; foreign exit 0 is ignored


class ResourceTests(unittest.TestCase):
    def test_restartable_init_overlap_and_order(self):
        def c(cpu, sidecar=False):
            return client.V1Container(name='c', resources=client.V1ResourceRequirements(requests={'cpu': str(cpu)}),
                                      restart_policy='Always' if sidecar else None)
        for init, expected in [([c(8, True)], 9.5), ([c(2, True), c(8), c(3, True)], 10.5),
                               ([c(8), c(2, True), c(3, True)], 8.5), ([c(8), c(9)], 9.5), ([], 1.5)]:
            p = client.V1Pod(spec=client.V1PodSpec(containers=[c(1)], init_containers=init, overhead={'cpu': '500m'}))
            with self.subTest(init=init):
                self.assertEqual(cluster._requests(p, 'cpu'), expected)

    def test_shm_option_does_not_resize_other_scratch_volumes(self):
        p = fixture()
        for name, medium, size in [('ram-scratch', 'Memory', '128Mi'), ('disk-scratch', '', '500Mi')]:
            p.spec.volumes.append(client.V1Volume(name=name, empty_dir=client.V1EmptyDirVolumeSource(
                medium=medium, size_limit=size)))
            p.spec.containers[0].volume_mounts.append(client.V1VolumeMount(name=name, mount_path='/'+name))
        t = load_template(p, {'items': []})
        m = pod_spec(manifest_from(t, shm='8Gi'))
        volumes = {v['name']: v for v in m['volumes']}
        self.assertEqual(volumes['ram-scratch']['emptyDir']['sizeLimit'], '128Mi')
        self.assertEqual(volumes['disk-scratch']['emptyDir']['sizeLimit'], '500Mi')
        self.assertEqual(volumes['shm']['emptyDir']['sizeLimit'], '8Gi')


class KinfoTests(unittest.TestCase):
    def kinfo(self, snapshot):
        t = mock.Mock(volumes=[], volume_mounts=[], skipped={}, poddefaults_known=True)
        out = io.StringIO()
        with mock.patch.object(core.NotebookTemplate, 'from_current_pod', return_value=t), \
                mock.patch.object(core, 'quota', return_value=[]), \
                mock.patch.object(cluster, 'load_snapshot', **snapshot), mock.patch('sys.stdout', out):
            cli.cmd_info(argparse.Namespace())
        return out.getvalue()

    def test_short_names_are_only_those_this_cluster_has(self):
        text = self.kinfo({'return_value': {'nodes': [{'product': 'Tesla-V100-SXM2-16GB'}, {'product': 'NVIDIA-RTX-A5000'}]}})
        self.assertIn('v100=Tesla-V100-SXM2-16GB', text)
        self.assertIn('a5000=NVIDIA-RTX-A5000', text)
        self.assertNotIn('a30', text)

    def test_all_short_names_when_the_cluster_view_is_unavailable(self):
        text = self.kinfo({'side_effect': cluster.ClusterViewUnavailable('no access')})
        self.assertIn('a30=NVIDIA-A30', text)


class PlainPodLogTests(unittest.TestCase):
    """The default pod log stays plain text for kubectl logs, the Kubeflow UI and older kjob; framing is opt-in."""

    def run_wrapper(self, d, **kw):
        command = "bash -c 'printf \"hello\\n\\r10%%\\r100%%\\n\"'"
        shell = core.job_shell('test', command, d, str(Path(d)/'job.log'), 'abc123', str(Path(d)/'term'), **kw)
        return subprocess.run(['bash', '-c', shell], capture_output=True, timeout=10, env={**os.environ, 'TMPDIR': d})

    def test_default_pod_log_is_the_scripts_own_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_wrapper(d)
            self.assertEqual(p.returncode, 0)
            self.assertNotIn(b'[kjob-output:', p.stdout)
            self.assertIn(b'\nhello\n\r10%\r100%\n[kjob ', p.stdout)
            self.assertEqual(p.stdout, (Path(d)/'job.log').read_bytes())

    def test_live_progress_frames_the_pod_log_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_wrapper(d, live_progress=True)
            self.assertIn(b'[kjob-output:abc123] ', p.stdout)
            self.assertIn(b'hello\n\r10%\r100%\n', (Path(d)/'job.log').read_bytes())

    def annotations(self, **kw):
        return manifest_from(load_template(fixture()), **kw)['spec']['pytorchReplicaSpecs']['Master']['template'] \
            ['metadata']['annotations']

    def test_only_live_progress_jobs_announce_a_framed_log(self):
        self.assertNotIn('kjob/log-stream', self.annotations())
        self.assertIn('kjob/log-stream', self.annotations(live_progress=True))

    def test_live_progress_option(self):
        with mock.patch.object(cli, 'resolve_gpu_types', return_value=([], None)), \
                mock.patch.object(cli, '_check_access'):
            for extra, expected in (([], False), (['--live-progress'], True)):
                a = cli.build_parser().parse_args(['submit', '-G', '0'] + extra + ['x.py'])
                self.assertIs(cli._spec(a).live_progress, expected)
