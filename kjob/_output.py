"""Lossless output framing for line-buffered container logs."""

from __future__ import annotations

import codecs
import json
import re


def frame_prefix(run_id: str) -> str:
    return f"[kjob-output:{run_id}] "


def pod_prefix(pod) -> str | None:
    annotations = getattr(getattr(pod, "metadata", None), "annotations", None)
    run_id = annotations.get("kjob/log-stream") if isinstance(annotations, dict) else None
    return frame_prefix(run_id) if isinstance(run_id, str) and run_id else None


def decode_frame(line: str, prefix: str) -> str:
    if line.startswith(prefix):
        try:
            value = json.loads(line[len(prefix):])
            if isinstance(value, str):
                return value
        except ValueError:
            pass
    return line  # plain fallback output, runtime diagnostics, or an incomplete frame


class PodOutputDecoder:
    def __init__(self, pod):
        self.prefix = pod_prefix(pod)

    def record(self, line: str) -> str:
        if re.match(r'\[kjob [^\]]*\] display (?:Python unavailable|formatter failed);', line):
            self.prefix = None
        return decode_frame(line, self.prefix) if self.prefix else line


def decode_pod_output(text: str, pod) -> str:
    decoder = PodOutputDecoder(pod)
    lines = text.split('\n')
    return ''.join(decoder.record(line+'\n') for line in lines[:-1]) + decoder.record(lines[-1])


def pod_output_chunks(response, pod):
    """Decode bounded transport records before interpreting user CR/LF characters."""
    prefix = pod_prefix(pod)
    records = PodOutputDecoder(pod)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    chunks = (iter(lambda: response.read1(1024), b'') if hasattr(type(response), 'read1')
              else response.stream(1))
    wire = ''
    for chunk in chunks:
        text = decoder.decode(chunk)
        if not prefix:
            yield text
            continue
        wire += text
        while '\n' in wire:
            line, wire = wire.split('\n', 1)
            yield records.record(line+'\n')
        # Frames are at most a few KiB. Plain fallback output can be newline-free.
        if len(wire) >= 64 * 1024:
            yield wire
            wire = ''
    tail = decoder.decode(b'', final=True)
    if prefix:
        yield records.record(wire+tail)
    else:
        yield tail


# Kept standalone: the job need not have kjob installed or importable.
DISPLAY_PROGRAM = """import codecs, json, os, sys
if sys.version_info < (3, 8):
    raise SystemExit(1)
prefix = sys.argv[1]
decoder = codecs.getincrementaldecoder('utf-8')('replace')
def emit(text):
    if not text:
        return
    data = (prefix + json.dumps(text, ensure_ascii=False) + '\\n').encode()
    while data:
        data = data[os.write(1, data):]
while True:
    data = os.read(0, 1024)
    if not data:
        break
    emit(decoder.decode(data))
emit(decoder.decode(b'', final=True))
"""
