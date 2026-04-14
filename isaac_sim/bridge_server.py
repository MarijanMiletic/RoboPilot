"""Isaac Sim Bridge Server — runs INSIDE Isaac Sim, receives commands over a socket.

This script starts a UR10 simulation and listens on localhost:54321 for JSON
commands from the MCP server. Commands include joint moves, IK targets, gripper
control, and state queries.

Launch with Isaac Sim's Python or your venv with isaacsim installed:
    python bridge_server.py

Or with the batch script:
    run_bridge.bat

Protocol: newline-delimited JSON over TCP socket.
    Request:  {"method": "get_joint_positions", "params": {}}
    Response: {"success": true, "result": {...}}
"""

import json
import logging
import os
import socket
import sys
import threading
import time

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRIDGE] %(levelname)s: %(message)s")
logger = logging.getLogger("bridge_server")

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 54321


class IsaacBridgeServer:
    """Runs UR10 simulation in Isaac Sim and accepts control commands over TCP."""

    def __init__(self, headless: bool = False) -> None:
        print("[BRIDGE] Starting Isaac Sim...", flush=True)
        from isaacsim import SimulationApp

        self.simulation_app = SimulationApp({"headless": headless})
        print("[BRIDGE] SimulationApp created, warming up...", flush=True)

        # Let Kit process a few frames so the renderer settles
        for i in range(5):
            self.simulation_app.update()
        print("[BRIDGE] Warmup done, importing modules...", flush=True)

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators.examples.universal_robots import UR10
        from isaacsim.robot.manipulators.examples.universal_robots.kinematics_solver import (
            KinematicsSolver,
        )
        print("[BRIDGE] Imports OK", flush=True)

        self.World = World
        self.UR10 = UR10
        self.DynamicCuboid = DynamicCuboid
        self.FixedCuboid = FixedCuboid
        self.VisualCuboid = VisualCuboid
        self.KinematicsSolver = KinematicsSolver
        self.ArticulationAction = ArticulationAction
        self.euler_angles_to_quat = euler_angles_to_quat

        self._setup_scene()

        # Command queue: commands from socket → sim loop
        self._command_queue: list[dict] = []
        self._response_map: dict[str, dict] = {}
        self._lock = threading.Lock()

        # Target state for smooth interpolation
        self._target_joint_positions: np.ndarray | None = None
        self._target_ee_position: np.ndarray | None = None
        self._target_ee_orientation: np.ndarray | None = None
        self._use_ik = False
        self._gripper_open = True
        self._is_moving = False
        self._emergency_stop = False

    def _setup_scene(self) -> None:
        """Build the simulation scene with UR10 and workspace objects."""
        print("[BRIDGE] Creating World...", flush=True)
        self.world = self.World(stage_units_in_meters=1.0)
        self.simulation_app.update()

        print("[BRIDGE] Adding ground plane...", flush=True)
        self.world.scene.add_default_ground_plane()
        self.simulation_app.update()

        # ── Robot mounting table (with collision so robot doesn't fall through) ─
        print("[BRIDGE] Adding robot table...", flush=True)
        self.world.scene.add(
            self.FixedCuboid(
                prim_path="/World/RobotTable",
                name="robot_table",
                position=np.array([0.0, 0.0, 0.4]),
                scale=np.array([0.6, 0.6, 0.8]),
                color=np.array([0.3, 0.3, 0.35]),
            )
        )
        self.simulation_app.update()

        print("[BRIDGE] Adding UR10 robot (this may take a moment)...", flush=True)
        self.ur10 = self.world.scene.add(
            self.UR10(
                prim_path="/World/UR10",
                name="my_ur10",
                position=np.array([0.0, 0.0, 0.8]),
                attach_gripper=True,
            )
        )
        print("[BRIDGE] UR10 added!", flush=True)
        self.simulation_app.update()

        print("[BRIDGE] Adding workspace objects...", flush=True)
        # Input conveyor — to the right of robot (FixedCuboid so workpiece doesn't fall through)
        self.world.scene.add(
            self.FixedCuboid(
                prim_path="/World/InputConveyor",
                name="input_conveyor",
                position=np.array([0.9, 0.0, 0.35]),
                scale=np.array([0.5, 0.4, 0.7]),
                color=np.array([0.2, 0.2, 0.25]),
            )
        )

        # Output conveyor — to the left of robot
        self.world.scene.add(
            self.FixedCuboid(
                prim_path="/World/OutputConveyor",
                name="output_conveyor",
                position=np.array([-0.9, 0.0, 0.35]),
                scale=np.array([0.5, 0.4, 0.7]),
                color=np.array([0.2, 0.2, 0.25]),
            )
        )

        # Sample workpiece — on top of input conveyor
        self.workpiece = self.world.scene.add(
            self.DynamicCuboid(
                prim_path="/World/Workpiece",
                name="workpiece",
                position=np.array([0.9, 0.0, 0.75]),
                scale=np.array([0.04, 0.04, 0.03]),
                color=np.array([1.0, 0.5, 0.0]),
                mass=0.15,
            )
        )
        self.simulation_app.update()

        print("[BRIDGE] Resetting world (initializing physics)...", flush=True)
        self.world.reset()
        # Step a few frames so articulations initialize properly
        for _ in range(10):
            self.world.step(render=True)
        print("[BRIDGE] Physics initialized", flush=True)

        # Set robot to upright "home" pose (all-zeros = arm stretched flat)
        print("[BRIDGE] Setting home pose...", flush=True)
        home_joints = np.array([0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0])
        action = self.ArticulationAction(joint_positions=home_joints)
        self.ur10.get_articulation_controller().apply_action(action)
        for _ in range(30):
            self.world.step(render=True)
        print("[BRIDGE] Robot in home position", flush=True)

        print("[BRIDGE] Setting up IK solver...", flush=True)
        self.ik_solver = self.KinematicsSolver(
            robot_articulation=self.ur10,
            attach_gripper=True,
        )

        # ── Lighting ──────────────────────────────────────────────────────
        print("[BRIDGE] Adding lights...", flush=True)
        from pxr import UsdLux, Gf, UsdGeom

        stage = self.world.stage

        # Dome light for ambient illumination
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(1000.0)

        # Distant light for directional shadows
        distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant.GetIntensityAttr().Set(3000.0)
        distant.GetAngleAttr().Set(1.0)
        xform = UsdGeom.Xformable(distant.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

        self.simulation_app.update()

        # ── Camera ───────────────────────────────────────────────────────
        print("[BRIDGE] Setting viewport camera...", flush=True)
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=np.array([2.5, 2.0, 2.0]),
            target=np.array([0.0, 0.0, 0.8]),
        )
        self.simulation_app.update()

        print("[BRIDGE] Scene ready — UR10 with gripper, conveyors, workpiece", flush=True)

    # ── Command handlers ─────────────────────────────────────────────────

    def _handle_command(self, cmd: dict) -> dict:
        """Dispatch a command and return a response dict."""
        method = cmd.get("method", "")
        params = cmd.get("params", {})

        try:
            if method == "get_joint_positions":
                return self._cmd_get_joint_positions()
            elif method == "get_end_effector_pose":
                return self._cmd_get_ee_pose()
            elif method == "get_robot_state":
                return self._cmd_get_robot_state()
            elif method == "move_to_joint_positions":
                return self._cmd_move_joints(params)
            elif method == "move_to_pose":
                return self._cmd_move_to_pose(params)
            elif method == "move_to_named_position":
                return self._cmd_move_named(params)
            elif method == "stop_robot":
                return self._cmd_stop()
            elif method == "open_gripper":
                self.ur10.gripper.open()
                self._gripper_open = True
                return {"success": True, "gripper": "open"}
            elif method == "close_gripper":
                self.ur10.gripper.close()
                self._gripper_open = False
                return {"success": True, "gripper": "closed"}
            elif method == "ping":
                sim_time = float(self.world.current_time)
                return {"success": True, "message": "pong", "sim_time": sim_time}
            else:
                return {"success": False, "error": f"Unknown method: {method}"}
        except Exception as e:
            logger.error("Command %s failed: %s", method, e)
            return {"success": False, "error": str(e)}

    def _cmd_get_joint_positions(self) -> dict:
        positions = self.ur10.get_joint_positions()
        names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        return {
            "success": True,
            "joint_names": names,
            "positions_rad": [round(float(p), 4) for p in positions[:6]],
            "positions_deg": [round(float(np.degrees(p)), 2) for p in positions[:6]],
        }

    def _cmd_get_ee_pose(self) -> dict:
        ee_pos, ee_rot = self.ik_solver.compute_end_effector_pose()
        position = {
            "x": round(float(ee_pos[0]), 4),
            "y": round(float(ee_pos[1]), 4),
            "z": round(float(ee_pos[2]), 4),
        }
        orientation = {
            "qx": round(float(ee_rot[1]), 4),
            "qy": round(float(ee_rot[2]), 4),
            "qz": round(float(ee_rot[3]), 4),
            "qw": round(float(ee_rot[0]), 4),
        }
        return {
            "success": True,
            "position": position,
            "orientation": orientation,
        }

    def _cmd_get_robot_state(self) -> dict:
        joints = self._cmd_get_joint_positions()
        ee = self._cmd_get_ee_pose()
        return {
            "success": True,
            "robot_model": "UR10",
            "controller_status": (
                "stopped" if self._emergency_stop
                else ("moving" if self._is_moving else "idle")
            ),
            "is_moving": self._is_moving,
            "joint_state": joints,
            "end_effector_pose": ee,
            "gripper": {"is_open": self._gripper_open, "model": "Surface Gripper"},
            "sim_time": float(self.world.current_time),
        }

    def _cmd_move_joints(self, params: dict) -> dict:
        joint_values = params.get("joint_values", [])
        if len(joint_values) != 6:
            return {"success": False, "error": "Expected 6 joint values"}
        self._target_joint_positions = np.array(joint_values, dtype=np.float64)
        self._use_ik = False
        self._is_moving = True
        self._emergency_stop = False
        return {"success": True, "move_type": "joint_space", "target": joint_values}

    def _cmd_move_to_pose(self, params: dict) -> dict:
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        z = params.get("z", 0.0)
        self._target_ee_position = np.array([x, y, z])
        # Default: gripper pointing down
        self._target_ee_orientation = self.euler_angles_to_quat(np.array([0, np.pi, 0]))
        self._use_ik = True
        self._is_moving = True
        self._emergency_stop = False
        return {"success": True, "move_type": "cartesian", "target": {"x": x, "y": y, "z": z}}

    def _cmd_move_named(self, params: dict) -> dict:
        name = params.get("position_name", "")
        named = {
            "home": [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0],
            "ready": [0.0, -1.2, 1.4, -1.7708, -1.5708, 0.0],
            "pick_approach": [1.0472, -1.0472, 1.5708, -2.0944, -1.5708, 0.0],
            "place_approach": [-1.0472, -1.0472, 1.5708, -2.0944, -1.5708, 0.0],
            "inspection": [0.0, -0.7854, 1.5708, -2.3562, -1.5708, 0.0],
        }
        if name not in named:
            available = list(named.keys())
            return {
                "success": False,
                "error": f"Unknown position: {name}. Available: {available}",
            }
        self._target_joint_positions = np.array(named[name], dtype=np.float64)
        self._use_ik = False
        self._is_moving = True
        self._emergency_stop = False
        return {
            "success": True,
            "move_type": "named",
            "named_position": name,
            "target": named[name],
        }

    def _cmd_stop(self) -> dict:
        self._emergency_stop = True
        self._is_moving = False
        self._target_joint_positions = None
        self._target_ee_position = None
        # Apply zero velocity by setting current positions as target
        current = self.ur10.get_joint_positions()
        action = self.ArticulationAction(joint_positions=current[:6])
        self.ur10.get_articulation_controller().apply_action(action)
        return {"success": True, "action": "emergency_stop", "all_velocities_zero": True}

    # ── Socket server ────────────────────────────────────────────────────

    def _start_socket_server(self) -> None:
        """Start TCP socket server in a background thread."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((BRIDGE_HOST, BRIDGE_PORT))
        server.listen(5)
        server.settimeout(1.0)
        logger.info("Bridge socket listening on %s:%d", BRIDGE_HOST, BRIDGE_PORT)

        def accept_loop():
            while self.simulation_app.is_running():
                try:
                    client, addr = server.accept()
                    logger.info("Client connected: %s", addr)
                    t = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error("Accept error: %s", e)

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()

    def _handle_client(self, client: socket.socket) -> None:
        """Handle a single client connection — read JSON lines, dispatch, reply."""
        buf = b""
        client.settimeout(1.0)
        while self.simulation_app.is_running():
            try:
                data = client.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        resp = {"success": False, "error": f"Invalid JSON: {e}"}
                        client.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                        continue

                    # Process command in sim thread via queue
                    cmd_id = f"{time.time():.6f}"
                    cmd["_id"] = cmd_id
                    with self._lock:
                        self._command_queue.append(cmd)

                    # Wait for response (up to 10s)
                    deadline = time.time() + 10.0
                    resp = None
                    while time.time() < deadline:
                        with self._lock:
                            if cmd_id in self._response_map:
                                resp = self._response_map.pop(cmd_id)
                                break
                        time.sleep(0.01)

                    if resp is None:
                        resp = {"success": False, "error": "Command timed out"}

                    client.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except socket.timeout:
                continue
            except Exception as e:
                logger.error("Client error: %s", e)
                break
        client.close()

    # ── Main simulation loop ─────────────────────────────────────────────

    def run(self) -> None:
        """Main loop: steps physics, processes commands, applies motion targets."""
        self._start_socket_server()

        print("=" * 60, flush=True)
        print("ISAAC SIM BRIDGE SERVER RUNNING", flush=True)
        print(f"UR10 robot ready — waiting for MCP commands on port {BRIDGE_PORT}", flush=True)
        print("=" * 60, flush=True)

        while self.simulation_app.is_running():
            self.world.step(render=True)

            if not self.world.is_playing():
                continue

            if self.world.current_time_step_index == 0:
                self.world.reset()

            # Process queued commands
            with self._lock:
                commands = list(self._command_queue)
                self._command_queue.clear()

            for cmd in commands:
                cmd_id = cmd.pop("_id", "")
                resp = self._handle_command(cmd)
                with self._lock:
                    self._response_map[cmd_id] = resp

            # Apply motion targets
            if not self._emergency_stop and self._is_moving:
                if self._use_ik and self._target_ee_position is not None:
                    action, success = self.ik_solver.compute_inverse_kinematics(
                        target_position=self._target_ee_position,
                        target_orientation=self._target_ee_orientation,
                    )
                    if success:
                        self.ur10.get_articulation_controller().apply_action(action)
                    # Check if arrived
                    ee_pos, _ = self.ik_solver.compute_end_effector_pose()
                    dist = np.linalg.norm(self._target_ee_position - ee_pos)
                    if dist < 0.005:
                        self._is_moving = False

                elif self._target_joint_positions is not None:
                    current = self.ur10.get_joint_positions()[:6]
                    # Smooth interpolation toward target
                    alpha = 0.05  # Smoothing factor
                    interp = current + alpha * (self._target_joint_positions - current)
                    action = self.ArticulationAction(joint_positions=interp)
                    self.ur10.get_articulation_controller().apply_action(action)
                    # Check if arrived
                    dist = np.linalg.norm(self._target_joint_positions - current)
                    if dist < 0.01:
                        self._is_moving = False

        logger.info("Shutting down bridge server...")
        self.simulation_app.close()


if __name__ == "__main__":
    headless = "--headless" in sys.argv
    server = IsaacBridgeServer(headless=headless)
    server.run()
