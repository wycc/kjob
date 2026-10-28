"""Exercise the real wrapper, transport decoder and CLI output together."""
import argparse
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from kjob import cli, core
from kjob._output import decode_pod_output, frame_prefix
from test_regressions import handle, pod


class Response:
    def __init__(self, data, size=7):
        self.data, self.size = data, size

    def stream(self, amt):
        for i in range(0, len(self.data), self.size):
            yield self.data[i:i+self.size]

    def close(self):
        pass


def follower(response, framed=True):
    j = handle(gpus=0)
    p = pod(phase='Succeeded', code=0)
    p.metadata.annotations = {'kjob/log-stream': 'abc123'} if framed else {}
    j.core.read_namespaced_pod.return_value = p
    j.core.read_namespaced_pod_log.return_value = response
    j.wait_for_pod = mock.Mock(return_value='Running')
    j.state = mock.Mock(return_value='Succeeded')
    return j


class OutputTests(unittest.TestCase):
    def test_wrapper_to_client_preserves_progress_windows_lines_and_literal_frames(self):
        payloads = ['\r10%\r20%\r100%\n', '10%\r20%\r100%\r\n',
                    'first\r\nsecond\r\n', '進度\r\n\r完成\n',
                    'x'*1023+'測試🔥\u2028next\n',
                    frame_prefix('abc123')+'"ordinary user text"\n',
                    'x'*(core.LOG_FRAGMENT_CHARS*2+31)+'\n']
        for payload in payloads:
            with self.subTest(payload=payload[:40]), tempfile.TemporaryDirectory() as d:
                log, term = Path(d)/'saved.log', Path(d)/'termination.json'
                source = Path(d)/'payload'
                source.write_bytes(payload.encode())
                shell = core.job_shell('test', 'cat '+shlex.quote(str(source)), d, str(log), 'abc123', str(term), live_progress=True)
                p = subprocess.run(['bash', '-c', shell], capture_output=True, timeout=10,
                                   env={**os.environ, 'TMPDIR': d})
                self.assertEqual(p.returncode, 0)
                original = log.read_bytes()
                self.assertIn(payload.encode(), original)
                expected = original.split(b'\n', 1)[0].decode()+'\n'+payload
                for size in (1, 7, 1024):
                    j = follower(Response(p.stdout, size))
                    out = io.StringIO()
                    self.assertEqual(j.follow(out=out, report=lambda _: None), 0)
                    self.assertEqual(out.getvalue(), expected)
                    # Non-following klogs must decode the same wire format once.
                    self.assertEqual(decode_pod_output(p.stdout.decode(), j._pod()), original.decode())
                j = follower(Response(p.stdout))
                j.core.read_namespaced_pod_log.return_value = p.stdout.decode()
                out = io.StringIO()
                with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', out):
                    self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=False)), 0)
                self.assertEqual(out.getvalue(), original.decode())
                j = follower(Response(p.stdout))
                out = io.StringIO()
                with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', out):
                    self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=True)), 0)
                self.assertEqual(out.getvalue(), original.decode())

    def test_real_running_wrapper_displays_progress_before_the_script_can_finish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            script, release = root/'train.py', root/'release'
            script.write_text("import time\nfrom pathlib import Path\n"
                              "print('\\r10%',end='',flush=True)\n"
                              f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
                              "print('\\r100%',flush=True)\n")
            shell = core.job_shell('test', shlex.join([sys.executable, str(script)]), d,
                                   str(root/'saved.log'), 'abc123', str(root/'termination.json'), live_progress=True)
            p = subprocess.Popen(['bash', '-c', shell], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env={**os.environ, 'TMPDIR': d})
            class PipeResponse:
                def stream(self, amt):
                    while True:
                        data = os.read(p.stdout.fileno(), amt)
                        if not data:
                            return
                        yield data
                def close(self):
                    pass
            visible = threading.Event()
            class Output(io.StringIO):
                def write(self, text):
                    n = super().write(text)
                    if '10%' in self.getvalue(): visible.set()
                    return n
            j, out, result = follower(PipeResponse()), Output(), []
            worker = threading.Thread(target=lambda: result.append(j.follow(out=out, report=lambda _: None)), daemon=True)
            worker.start()
            try:
                self.assertTrue(visible.wait(5), out.getvalue())
                self.assertIsNone(p.poll())
            finally:
                release.touch()
                worker.join(5)
                p.wait(timeout=5)
                p.stdout.close(); p.stderr.close()
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [0])
            self.assertTrue(out.getvalue().endswith('\r10%\r100%\n'))

    def restricted_path(self, root):
        directory = root/'bin'
        directory.mkdir()
        for name in ('bash', 'cat', 'date', 'mkdir', 'rm', 'setsid', 'tee'):
            target = shutil.which(name)
            if target: (directory/name).symlink_to(target)
        return str(directory)

    def test_missing_or_unusable_python_falls_back_before_starting_script_once(self):
        for available in (True, False):
            for script_rc in (0, 7):
                for unusable in ('missing', 'not-executable', 'broken'):
                    with self.subTest(available=available, rc=script_rc, unusable=unusable), tempfile.TemporaryDirectory() as d:
                        root = Path(d)
                        interpreter = root/'display-python'
                        if unusable != 'missing':
                            interpreter.write_text('#!/bin/bash\nexit 23\n')
                            interpreter.chmod(0o755 if unusable == 'broken' else 0o644)
                        command = "bash -c 'printf x >> runs; printf \"HELLO\\n\"; exit "+str(script_rc)+"'"
                        with mock.patch.object(core.sys, 'executable', str(interpreter)):
                            shell = core.job_shell('test', command, d, str(root/'saved.log'), 'abc123', str(root/'termination.json'), live_progress=True)
                        env = {**os.environ, 'TMPDIR': d}
                        if not available: env['PATH'] = self.restricted_path(root)
                        p = subprocess.run(['/bin/bash', '-c', shell], capture_output=True, timeout=10, env=env)
                        self.assertEqual(p.returncode, script_rc)
                        self.assertEqual((root/'runs').read_text(), 'x')
                        self.assertIn(b'HELLO\n', (root/'saved.log').read_bytes())
                        self.assertNotIn(b'log write failed', p.stdout)
                        result = json.loads((root/'termination.json').read_text())
                        self.assertTrue(result['log_saved'])
                        self.assertEqual(result['script_exit_code'], script_rc)
                        if not available:
                            self.assertIn(b'display Python unavailable', p.stdout)
                            self.assertIn(b'HELLO', p.stdout)

    def test_formatter_failure_after_preflight_does_not_kill_training_or_blame_storage(self):
        for rc in (0, 7):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                interpreter = root/'display-python'
                flag = root/'checked'
                interpreter.write_text('#!/bin/bash\nif [ -f '+shlex.quote(str(flag))+' ]; then read -r -n 1; exit 23; fi\n'
                                       ': > '+shlex.quote(str(flag))+'\nexec '+shlex.quote(sys.executable)+' "$@"\n')
                interpreter.chmod(0o755)
                payload = 'x'*32768+'\n'+frame_prefix('abc123')+'"literal frame text"\n'
                script = root/'train.sh'
                script.write_text('printf x >> runs\nprintf %s '+shlex.quote(payload)+'\nexit '+str(rc)+'\n')
                with mock.patch.object(core.sys, 'executable', str(interpreter)):
                    shell = core.job_shell('test', 'bash '+shlex.quote(str(script)), d,
                                           str(root/'saved.log'), 'abc123', str(root/'termination.json'), live_progress=True)
                p = subprocess.run(['bash', '-c', shell], capture_output=True, timeout=10,
                                   env={**os.environ, 'TMPDIR': d})
                self.assertEqual(p.returncode, rc)
                self.assertEqual((root/'runs').read_text(), 'x')
                self.assertIn(payload.encode(), (root/'saved.log').read_bytes())
                self.assertTrue(json.loads((root/'termination.json').read_text())['log_saved'])
                self.assertIn(b'display formatter failed;', p.stdout)
                self.assertNotIn(b'log write failed:', p.stdout)
                decoded = decode_pod_output(p.stdout.decode(), follower(Response(b''))._pod())
                self.assertIn(frame_prefix('abc123')+'"literal frame text"\n', decoded)

    def test_plain_legacy_logs_and_foreign_or_malformed_frames_are_preserved(self):
        text = frame_prefix('other')+'"text"\n'+frame_prefix('abc123')+'invalid json\nplain\r\n'
        for framed in (True, False):
            j = follower(Response(text.encode()), framed)
            self.assertEqual(decode_pod_output(text, j._pod()), text)
            out = io.StringIO()
            self.assertEqual(j.follow(out=out, report=lambda _: None), 0)
            self.assertEqual(out.getvalue(), text)
