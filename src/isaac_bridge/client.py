"""Client that connects the MCP server to the Isaac Sim bridge over TCP socket.

The bridge_server.py runs inside Isaac Sim and listens on localhost:55123.
This client sends JSON commands and receives JSON responses.
"""

import json
import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 55123
CONNECT_TIMEOUT = 3.0
RECV_TIMEOUT = 15.0


class IsaacBridgeClient:
    """TCP client that communicates with the Isaac Sim bridge server."""

    def __init__(self, host: str = BRIDGE_HOST, port: int = BRIDGE_PORT) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._buf = b""

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Attempt to connect to the Isaac Sim bridge server."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(CONNECT_TIMEOUT)
            self._sock.connect((self._host, self._port))
            self._sock.settimeout(RECV_TIMEOUT)
            self._connected = True
            logger.info("Connected to Isaac Sim bridge at %s:%d", self._host, self._port)

            # Verify with ping
            resp = self.send_command("ping")
            if resp and resp.get("success"):
                logger.info("Isaac Sim bridge confirmed — sim_time=%.2f", resp.get("sim_time", 0))
                return True
            else:
                logger.warning("Bridge ping failed")
                self.disconnect()
                return False
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            logger.info("Isaac Sim bridge not available at %s:%d (%s)", self._host, self._port, e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the socket connection."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False

    def send_command(self, method: str, params: Optional[dict] = None) -> dict:
        """Send a command to the bridge and return the response."""
        if not self._sock:
            return {"success": False, "error": "Not connected to Isaac Sim bridge"}

        cmd = {"method": method, "params": params or {}}
        try:
            msg = json.dumps(cmd) + "\n"
            self._sock.sendall(msg.encode("utf-8"))

            # Read response line
            return self._read_response()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error("Bridge connection lost: %s", e)
            self._connected = False
            return {"success": False, "error": f"Connection lost: {e}"}

    def _read_response(self) -> dict:
        """Read a single newline-delimited JSON response."""
        while b"\n" not in self._buf:
            try:
                data = self._sock.recv(65536)
                if not data:
                    self._connected = False
                    return {"success": False, "error": "Connection closed"}
                self._buf += data
            except socket.timeout:
                return {"success": False, "error": "Response timeout"}

        line, self._buf = self._buf.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid response: {e}"}

    # ── Convenience methods matching ROS2Client interface ────────────────

    def get_joint_positions(self) -> dict:
        return self.send_command("get_joint_positions")

    def get_end_effector_pose(self) -> dict:
        return self.send_command("get_end_effector_pose")

    def get_robot_state(self) -> dict:
        return self.send_command("get_robot_state")

    def move_to_joint_positions(self, joint_values: list[float]) -> dict:
        return self.send_command("move_to_joint_positions", {"joint_values": joint_values})

    def move_to_pose(self, x: float, y: float, z: float) -> dict:
        return self.send_command("move_to_pose", {"x": x, "y": y, "z": z})

    def move_to_named_position(self, position_name: str) -> dict:
        return self.send_command("move_to_named_position", {"position_name": position_name})

    def stop_robot(self) -> dict:
        return self.send_command("stop_robot")

    def open_gripper(self) -> dict:
        return self.send_command("open_gripper")

    def close_gripper(self) -> dict:
        return self.send_command("close_gripper")
