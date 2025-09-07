from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Dict, Iterator, Optional


def file_tail(path: str, poll_sec: float = 0.2) -> Iterator[Dict[str, float]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_sec)
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def udp_json_stream(host: str, port: int) -> Iterator[Dict[str, float]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            yield json.loads(data.decode("utf-8", errors="ignore"))
        except socket.timeout:
            continue
        except Exception:
            continue
