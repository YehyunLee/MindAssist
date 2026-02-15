"""Local UDP stream helpers for EEG data sharing between processes."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional


UDP_HOST = "127.0.0.1"
UDP_PORT = 8765


class UDPBroadcaster:
    """Sends small JSON EEG payloads over localhost UDP."""

    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: Dict[str, Any]) -> None:
        msg = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(msg, self.addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class UDPSubscriber:
    """Receives EEG payload JSON datagrams from localhost UDP."""

    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT, timeout: float = 0.1):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.sock.bind((host, port))

    def recv(self) -> Optional[Dict[str, Any]]:
        try:
            raw, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return None
        except OSError:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
