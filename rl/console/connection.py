"""Short SSH requests to the job manager. Credentials are used only for login."""

import base64
import gzip
import hashlib
import hmac
import io
import json
import re
import socket
import uuid
from pathlib import Path, PurePosixPath

import paramiko

HERE = Path(__file__).resolve().parent
REMOTE = PurePosixPath(".local/share/a2-fab-rl")


def whole_number(value, label):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    digits = str(value).strip()
    if not re.fullmatch(r"[0-9]{1,6}", digits):
        raise ValueError(f"{label}必须是非负整数。")
    return int(digits)


def address(host, port):
    host = str(host or "").strip().strip("[]")
    if not host or len(host) > 253 or not re.fullmatch(r"[a-zA-Z0-9.:_-]+", host):
        raise ValueError("请输入服务器 IP 或主机名。")
    port = whole_number(port, "SSH 端口")
    if not 1 <= port <= 65535:
        raise ValueError("SSH 端口必须是 1 到 65535 的整数。")
    return host, port


def fingerprint(key):
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def inspect_key(host, port):
    with socket.create_connection(address(host, port), timeout=8) as sock:
        with paramiko.Transport(sock) as transport:
            transport.start_client(timeout=8)
            return fingerprint(transport.get_remote_server_key())


class VerifiedHost(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected):
        self.expected = expected

    def missing_host_key(self, client, hostname, key):
        if not self.expected or not hmac.compare_digest(fingerprint(key), self.expected):
            raise ValueError("未知主机或指纹不一致。请先读取并核对 SHA256 主机指纹。")


class Connection:
    def __init__(self, host, port, user, password, expected):
        self.host, self.port = address(host, port)
        self.user = str(user or "").strip()
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", self.user):
            raise ValueError("请输入 SSH 用户名。")
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(VerifiedHost(str(expected or "").strip()))
        try:
            self.client.connect(
                self.host,
                port=self.port,
                username=self.user,
                password=password or None,
                timeout=10,
                banner_timeout=10,
                auth_timeout=12,
                allow_agent=not password,
                look_for_keys=not password,
            )
            self.client.get_transport().set_keepalive(30)
            contents = (HERE / "worker.py").read_bytes()
            name = "worker-" + hashlib.sha256(contents).hexdigest() + ".py"
            self.worker = REMOTE / "helpers" / name
            with self.client.open_sftp() as sftp:
                sftp.get_channel().settimeout(30)
                path = PurePosixPath(".")
                for part in (*REMOTE.parts, "helpers"):
                    path /= part
                    self.mkdir(sftp, path)
                self.put(sftp, self.worker, contents, reuse=True)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def mkdir(sftp, path):
        try:
            sftp.stat(str(path))
        except FileNotFoundError:
            try:
                sftp.mkdir(str(path), mode=0o700)
            except OSError:
                sftp.stat(str(path))  # Another connection may have created it.

    @staticmethod
    def put(sftp, target, contents, callback=None, reuse=False):
        if reuse:
            try:
                if sftp.stat(str(target)).st_size == len(contents):
                    return
            except FileNotFoundError:
                pass
        tmp = str(target) + "." + uuid.uuid4().hex + ".tmp"
        try:
            sftp.putfo(io.BytesIO(contents), tmp, file_size=len(contents), callback=callback, confirm=True)
            sftp.chmod(tmp, 0o600)
            sftp.posix_rename(tmp, str(target))
        finally:
            try:
                sftp.remove(tmp)
            except FileNotFoundError:
                pass

    def install_demo(self, callback=None):
        with gzip.open(HERE / "assets/demo-policy.pt.gz", "rb") as f:
            content = f.read()
        with self.client.open_sftp() as sftp:
            sftp.get_channel().settimeout(30)
            self.mkdir(sftp, REMOTE / "models")
            self.put(sftp, REMOTE / "models/demo.pt", content, callback)

    def call(self, action, **options):
        transport = self.client.get_transport()
        if transport is None or not transport.is_active():
            raise ConnectionError("SSH 连接已断开，请重新连接以查看远程任务。")
        with transport.open_session(timeout=15) as channel:
            channel.settimeout(30)
            channel.set_combine_stderr(True)
            channel.exec_command(f'exec python3 "$HOME/{self.worker}"')
            channel.sendall(json.dumps(dict(action=action, **options)).encode())
            channel.shutdown_write()
            with channel.makefile("rb") as stream:
                raw = stream.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("远程响应超过限制。")
            try:
                result = json.loads(raw)
            except ValueError:
                raise RuntimeError(
                    raw.decode(errors="replace")[-700:] or "远程任务管理器未返回结果。"
                ) from None
            if channel.recv_exit_status() or "error" in result:
                raise RuntimeError(result.get("error", "远程控制命令失败。"))
            return result

    def close(self):
        self.client.close()
