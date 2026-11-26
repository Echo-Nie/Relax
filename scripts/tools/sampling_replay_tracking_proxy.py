# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Expose only configured ClearML HTTPS endpoints to an owned private netns."""

import argparse
import os
import select
import socket
import socketserver
from pathlib import Path
from urllib.parse import urlsplit

from scripts.tools.sampling_replay_isolated import checked_state


def clearml_hosts() -> set[str]:
    from clearml.backend_api.session import Session

    urls = (Session.get_api_server_host(), Session.get_app_server_host(), Session.get_files_server_host())
    hosts = set()
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.port not in (None, 443) or not parsed.hostname:
            raise ValueError("Tracking bridge requires configured HTTPS endpoints on port 443")
        hosts.add(parsed.hostname)
    return hosts


class ClearMLTunnel(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(30)
        line = self.rfile.readline(4097).decode("ascii", errors="replace").strip().split()
        if len(line) != 3 or line[0] != "CONNECT" or line[1] not in self.server.allowed_targets:
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        for _ in range(64):
            header = self.rfile.readline(8193)
            if header == b"\r\n":
                break
            if not header or len(header) > 8192:
                return
        else:
            return
        host, port = line[1].rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=30) as upstream:
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()
            while True:
                readable, _, _ = select.select([self.connection, upstream], [], [], 90)
                if not readable:
                    return
                for source in readable:
                    payload = source.recv(65536)
                    if not payload:
                        return
                    target = upstream if source is self.connection else self.connection
                    target.sendall(payload)


class TrackingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True


def serve(instance: Path, port: int) -> None:
    state = checked_state(instance)
    hosts = clearml_hosts()
    parent = os.open("/proc/self/ns/net", os.O_RDONLY)
    target = os.open(f"/proc/{state['pid']}/ns/net", os.O_RDONLY)
    try:
        os.setns(target, os.CLONE_NEWNET)
        server = TrackingServer(("127.0.0.1", port), ClearMLTunnel)
    finally:
        os.setns(parent, os.CLONE_NEWNET)
        os.close(target)
        os.close(parent)
    server.allowed_targets = {f"{host}:443" for host in hosts}
    # The listening socket belongs to the private namespace; outbound sockets
    # are opened in its parent. No shared interface, routes or firewall changes.
    print(f"ClearML-only bridge ready on private 127.0.0.1:{port}", flush=True)
    with server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    serve(args.instance, args.port)
