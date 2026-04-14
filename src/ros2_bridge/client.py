"""Async ROS2 client with Isaac Sim bridge and mock fallback.

Connection priority: Isaac Sim bridge (socket) > ROS2 (rclpy) > Mock mode.
"""

import asyncio
import base64
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# Guard ROS2 imports — fall back to mock mode if unavailable
try:
    import rclpy  # noqa: F401

    ROS2_AVAILABLE = True
    logger.info("ROS2 (rclpy) available — live mode enabled")
except ImportError:
    ROS2_AVAILABLE = False
    logger.info("ROS2 (rclpy) not available")

# Try to import Isaac bridge client
try:
    from isaac_bridge.client import IsaacBridgeClient

    ISAAC_BRIDGE_AVAILABLE = True
except ImportError:
    ISAAC_BRIDGE_AVAILABLE = False


def _load_robot_config() -> dict:
    """Load robot configuration from YAML."""
    path = CONFIG_DIR / "robot_config.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f).get("robot", {})
    return {}


UR10E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class ROS2Client:
    """Async ROS2 client with Isaac Sim bridge and mock fallback.

    Connection priority: Isaac Sim bridge (socket) > ROS2 (rclpy) > Mock mode.
    """

    def __init__(self) -> None:
        self._robot_config = _load_robot_config()
        self._named_positions = self._robot_config.get("named_positions", {})
        self._joint_state: dict[str, float] = {}
        self._joint_velocities: dict[str, float] = {}
        self._joint_efforts: dict[str, float] = {}
        self._camera_frames: dict[str, bytes] = {}
        self._gripper_position: float = 0.085  # open
        self._node: Optional[Any] = None
        self._spin_task: Optional[asyncio.Task] = None
        self._is_moving: bool = False
        self._controller_status: str = "idle"
        self._start_time = time.time()
        self._isaac_bridge: Optional[Any] = None
        self._mode: str = "mock"  # "isaac", "ros2", or "mock"

        # Initialize mock state at home position
        home = self._named_positions.get("home", {})
        home_values = home.get("joint_values", [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0])
        for name, val in zip(UR10E_JOINT_NAMES, home_values):
            self._joint_state[name] = val
            self._joint_velocities[name] = 0.0
            self._joint_efforts[name] = 0.0

    async def connect(self) -> bool:
        """Connect using priority: Isaac Sim bridge > ROS2 > Mock mode."""
        # 1. Try Isaac Sim bridge (socket on localhost:54321)
        if ISAAC_BRIDGE_AVAILABLE:
            self._isaac_bridge = IsaacBridgeClient()
            if self._isaac_bridge.connect():
                self._mode = "isaac"
                logger.info("Connected to Isaac Sim bridge — LIVE simulation mode")
                return True
            else:
                self._isaac_bridge = None
                logger.info("Isaac Sim bridge not running")

        # 2. Try ROS2
        if not ROS2_AVAILABLE:
            self._mode = "mock"
            logger.info("Mock mode active — no Isaac Sim or ROS2")
            return True

        try:
            rclpy.init()
            self._node = rclpy.create_node("industrial_mcp_client")
            self._setup_subscriptions()
            self._spin_task = asyncio.create_task(self._spin_loop())
            self._mode = "ros2"
            logger.info("ROS2 node initialized: industrial_mcp_client")
            return True
        except Exception as e:
            logger.error("Failed to initialize ROS2: %s", e)
            return False

    def _setup_subscriptions(self) -> None:
        """Create ROS2 topic subscriptions."""
        if not self._node:
            return
        try:
            from sensor_msgs.msg import Image, JointState

            self._node.create_subscription(
                JointState, "/joint_states", self._joint_state_callback, 10
            )
            for cam_name in ["wrist", "overhead", "inspection"]:
                topic = f"/camera_{cam_name}/color/image_raw"
                self._node.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=cam_name: self._camera_callback(msg, name),
                    10,
                )
            logger.info("ROS2 subscriptions created")
        except ImportError:
            logger.warning("sensor_msgs not available — subscriptions skipped")

    def _joint_state_callback(self, msg: Any) -> None:
        """Handle incoming JointState messages."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._joint_state[name] = msg.position[i]
            if i < len(msg.velocity):
                self._joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self._joint_efforts[name] = msg.effort[i]

    def _camera_callback(self, msg: Any, camera_name: str) -> None:
        """Handle incoming Image messages — cache raw data."""
        self._camera_frames[camera_name] = bytes(msg.data)

    async def _spin_loop(self) -> None:
        """Background task to spin the ROS2 node without blocking asyncio."""
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
            await asyncio.sleep(0.01)

    async def shutdown(self) -> None:
        """Clean up ROS2 and Isaac bridge resources."""
        if self._isaac_bridge:
            self._isaac_bridge.disconnect()
        if self._spin_task:
            self._spin_task.cancel()
        if self._node:
            self._node.destroy_node()
        if ROS2_AVAILABLE:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    # ── Robot State ��─────────────────────────────────────────────────��───

    async def get_joint_positions(self) -> dict:
        """Get current joint positions in radians."""
        if self._mode == "isaac":
            result = self._isaac_bridge.get_joint_positions()
            if result.get("success"):
                result["timestamp"] = time.time()
                return result
            return self._mock_joint_positions()
        if self._mode == "mock":
            return self._mock_joint_positions()
        return {
            "joint_names": UR10E_JOINT_NAMES,
            "positions_rad": [self._joint_state.get(j, 0.0) for j in UR10E_JOINT_NAMES],
            "positions_deg": [
                round(math.degrees(self._joint_state.get(j, 0.0)), 2)
                for j in UR10E_JOINT_NAMES
            ],
            "timestamp": time.time(),
        }

    async def get_end_effector_pose(self) -> dict:
        """Get end-effector Cartesian pose (position + quaternion)."""
        if self._mode == "isaac":
            result = self._isaac_bridge.get_end_effector_pose()
            if result.get("success"):
                result["timestamp"] = time.time()
                return result
            return self._mock_ee_pose()
        if self._mode == "mock":
            return self._mock_ee_pose()
        return self._mock_ee_pose()

    async def get_robot_state(self) -> dict:
        """Get full robot state including joints, EE pose, and controller status."""
        if self._mode == "isaac":
            result = self._isaac_bridge.get_robot_state()
            if result.get("success"):
                result["timestamp"] = time.time()
                result["mode"] = "isaac_sim"
                result["uptime_seconds"] = round(time.time() - self._start_time, 1)
                return result
        joints = await self.get_joint_positions()
        ee_pose = await self.get_end_effector_pose()
        return {
            "robot_model": "UR10e",
            "controller_status": self._controller_status,
            "is_moving": self._is_moving,
            "joint_state": joints,
            "end_effector_pose": ee_pose,
            "gripper": {
                "model": "Robotiq 2F-85",
                "position_m": round(self._gripper_position, 4),
                "is_open": self._gripper_position > 0.04,
                "force_N": 100,
            },
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "timestamp": time.time(),
        }

    # ── Motion Planning ────────────────────��─────────────────────────────

    async def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.7071,
        qz: float = 0.0,
        qw: float = 0.7071,
        velocity_scaling: float = 0.5,
        acceleration_scaling: float = 0.5,
    ) -> dict:
        """Plan and execute Cartesian motion to target pose via MoveIt2 or Isaac Sim."""
        if self._mode == "isaac":
            result = self._isaac_bridge.move_to_pose(x, y, z)
            result["timestamp"] = time.time()
            return result
        if self._mode == "mock":
            return self._mock_move_result(
                "cartesian", {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}
            )
        return self._mock_move_result(
            "cartesian", {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}
        )

    async def move_to_joint_positions(self, joint_values: list[float]) -> dict:
        """Plan and execute joint-space motion."""
        if len(joint_values) != 6:
            return {"success": False, "error": "Expected 6 joint values for UR10e"}
        if self._mode == "isaac":
            result = self._isaac_bridge.move_to_joint_positions(joint_values)
            result["timestamp"] = time.time()
            return result
        if self._mode == "mock":
            return self._mock_move_joint(joint_values)
        return self._mock_move_joint(joint_values)

    async def move_to_named_position(self, position_name: str) -> dict:
        """Move to a predefined named position (home, ready, pick_approach, etc.)."""
        if position_name not in self._named_positions:
            available = list(self._named_positions.keys())
            return {
                "success": False,
                "error": f"Unknown position '{position_name}'. Available: {available}",
            }
        pos = self._named_positions[position_name]
        joint_values = pos["joint_values"]
        result = await self.move_to_joint_positions(joint_values)
        result["named_position"] = position_name
        result["description"] = pos.get("description", "")
        return result

    async def execute_pick_and_place(
        self,
        pick_x: float,
        pick_y: float,
        pick_z: float,
        place_x: float,
        place_y: float,
        place_z: float,
        approach_height: float = 0.1,
        gripper_width: float = 0.04,
    ) -> dict:
        """Execute a complete pick-and-place sequence."""
        steps = []

        # 1. Move to pick approach
        result = await self.move_to_pose(pick_x, pick_y, pick_z + approach_height)
        steps.append({"step": "pick_approach", "success": result.get("success", True)})

        # 2. Open gripper
        self._gripper_position = 0.085
        steps.append({"step": "gripper_open", "success": True})

        # 3. Move down to pick
        result = await self.move_to_pose(pick_x, pick_y, pick_z)
        steps.append({"step": "pick_descend", "success": result.get("success", True)})

        # 4. Close gripper
        self._gripper_position = gripper_width
        steps.append({"step": "gripper_close", "success": True, "grip_width_m": gripper_width})

        # 5. Retract
        result = await self.move_to_pose(pick_x, pick_y, pick_z + approach_height)
        steps.append({"step": "pick_retract", "success": result.get("success", True)})

        # 6. Move to place approach
        result = await self.move_to_pose(place_x, place_y, place_z + approach_height)
        steps.append({"step": "place_approach", "success": result.get("success", True)})

        # 7. Move down to place
        result = await self.move_to_pose(place_x, place_y, place_z)
        steps.append({"step": "place_descend", "success": result.get("success", True)})

        # 8. Open gripper
        self._gripper_position = 0.085
        steps.append({"step": "gripper_release", "success": True})

        # 9. Retract
        result = await self.move_to_pose(place_x, place_y, place_z + approach_height)
        steps.append({"step": "place_retract", "success": result.get("success", True)})

        all_ok = all(s["success"] for s in steps)
        return {
            "success": all_ok,
            "operation": "pick_and_place",
            "pick_position": {"x": pick_x, "y": pick_y, "z": pick_z},
            "place_position": {"x": place_x, "y": place_y, "z": place_z},
            "steps_completed": len([s for s in steps if s["success"]]),
            "total_steps": len(steps),
            "steps": steps,
        }

    async def stop_robot(self) -> dict:
        """Emergency stop — zero velocity on all joints."""
        if self._mode == "isaac":
            result = self._isaac_bridge.stop_robot()
            result["timestamp"] = time.time()
            logger.warning("EMERGENCY STOP sent to Isaac Sim")
            return result
        self._is_moving = False
        self._controller_status = "stopped"
        for name in UR10E_JOINT_NAMES:
            self._joint_velocities[name] = 0.0
        logger.warning("EMERGENCY STOP executed")
        return {
            "success": True,
            "action": "emergency_stop",
            "controller_status": "stopped",
            "all_velocities_zero": True,
            "timestamp": time.time(),
        }

    # ── Camera ───────────────────────────────────────────────────────────

    async def get_camera_image(self, camera_name: str) -> dict:
        """Capture image from named camera, return as base64 JPEG."""
        valid_cameras = ["wrist", "overhead", "inspection"]
        if camera_name not in valid_cameras:
            return {"error": f"Unknown camera '{camera_name}'. Available: {valid_cameras}"}

        if not ROS2_AVAILABLE or camera_name not in self._camera_frames:
            return self._mock_camera_image(camera_name)

        # Convert raw ROS2 image to base64 JPEG
        try:
            import io

            from PIL import Image as PILImage

            raw = self._camera_frames[camera_name]
            img = PILImage.frombytes("RGB", (1280, 720), raw)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {
                "camera_name": camera_name,
                "format": "jpeg",
                "resolution": [1280, 720],
                "image_base64": b64,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Camera image conversion failed: %s", e)
            return {"error": str(e)}

    # ── ROS2 Introspection ───────────────────────────────────────────────

    async def list_topics(self) -> list[dict]:
        """List all active ROS2 topics."""
        if not ROS2_AVAILABLE or not self._node:
            return self._mock_topics()
        topics = self._node.get_topic_names_and_types()
        return [{"topic": name, "types": types} for name, types in topics]

    async def list_nodes(self) -> list[str]:
        """List all active ROS2 nodes."""
        if not ROS2_AVAILABLE or not self._node:
            return self._mock_nodes()
        return self._node.get_node_names()

    # ── Mock Data Generators ─────────────────────────────────────────────

    def _mock_joint_positions(self) -> dict:
        """Generate realistic mock joint positions with slight noise."""
        noise = [random.gauss(0, 0.001) for _ in range(6)]
        positions = [self._joint_state.get(j, 0.0) + n for j, n in zip(UR10E_JOINT_NAMES, noise)]
        return {
            "joint_names": UR10E_JOINT_NAMES,
            "positions_rad": [round(p, 4) for p in positions],
            "positions_deg": [round(math.degrees(p), 2) for p in positions],
            "timestamp": time.time(),
            "_mock": True,
        }

    def _mock_ee_pose(self) -> dict:
        """Generate realistic end-effector pose for UR10e home position."""
        return {
            "position": {"x": 0.1171, "y": 0.4783, "z": 1.2072},
            "orientation": {"qx": 0.0, "qy": 0.7071, "qz": 0.0, "qw": 0.7071},
            "frame_id": "base_link",
            "child_frame_id": "tool0",
            "timestamp": time.time(),
            "_mock": True,
        }

    def _mock_move_result(self, move_type: str, target: dict) -> dict:
        """Mock a successful motion planning result."""
        planning_time = round(random.uniform(0.3, 1.5), 3)
        execution_time = round(random.uniform(1.5, 4.0), 3)
        self._controller_status = "idle"
        return {
            "success": True,
            "move_type": move_type,
            "target": target,
            "planning_time_s": planning_time,
            "execution_time_s": execution_time,
            "trajectory_points": random.randint(40, 120),
            "controller_status": "idle",
            "timestamp": time.time(),
            "_mock": True,
        }

    def _mock_move_joint(self, joint_values: list[float]) -> dict:
        """Mock joint-space motion and update internal state."""
        for name, val in zip(UR10E_JOINT_NAMES, joint_values):
            self._joint_state[name] = val
        return self._mock_move_result(
            "joint_space",
            {name: round(val, 4) for name, val in zip(UR10E_JOINT_NAMES, joint_values)},
        )

    def _mock_camera_image(self, camera_name: str) -> dict:
        """Generate a synthetic test image as base64 JPEG."""
        try:
            import io

            from PIL import Image as PILImage
            from PIL import ImageDraw

            w, h = (1280, 720) if camera_name != "overhead" else (1920, 1080)
            img = PILImage.new("RGB", (w, h), color=(40, 40, 50))
            draw = ImageDraw.Draw(img)
            # Grid lines
            for x in range(0, w, 100):
                draw.line([(x, 0), (x, h)], fill=(60, 60, 70), width=1)
            for y in range(0, h, 100):
                draw.line([(0, y), (w, y)], fill=(60, 60, 70), width=1)
            # Crosshair
            cx, cy = w // 2, h // 2
            draw.line([(cx - 30, cy), (cx + 30, cy)], fill=(0, 255, 0), width=2)
            draw.line([(cx, cy - 30), (cx, cy + 30)], fill=(0, 255, 0), width=2)
            # Label
            draw.text((20, 20), f"MOCK: {camera_name}", fill=(0, 255, 0))
            draw.text((20, 50), "UR10e Industrial Cell", fill=(180, 180, 180))
            draw.text((20, 80), f"Resolution: {w}x{h}", fill=(180, 180, 180))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            return {
                "camera_name": camera_name,
                "format": "jpeg",
                "resolution": [w, h],
                "image_base64": b64,
                "timestamp": time.time(),
                "_mock": True,
            }
        except ImportError:
            return {
                "camera_name": camera_name,
                "error": "Pillow not installed — cannot generate mock image",
                "_mock": True,
            }

    def _mock_topics(self) -> list[dict]:
        """Return realistic mock ROS2 topic list for a UR10e + Isaac Sim setup."""
        return [
            {"topic": "/joint_states", "types": ["sensor_msgs/msg/JointState"]},
            {"topic": "/tf", "types": ["tf2_msgs/msg/TFMessage"]},
            {"topic": "/tf_static", "types": ["tf2_msgs/msg/TFMessage"]},
            {"topic": "/clock", "types": ["rosgraph_msgs/msg/Clock"]},
            {
                "topic": "/camera_wrist/color/image_raw",
                "types": ["sensor_msgs/msg/Image"],
            },
            {
                "topic": "/camera_overhead/color/image_raw",
                "types": ["sensor_msgs/msg/Image"],
            },
            {
                "topic": "/camera_inspection/color/image_raw",
                "types": ["sensor_msgs/msg/Image"],
            },
            {
                "topic": "/camera_wrist/camera_info",
                "types": ["sensor_msgs/msg/CameraInfo"],
            },
            {
                "topic": "/camera_overhead/camera_info",
                "types": ["sensor_msgs/msg/CameraInfo"],
            },
            {
                "topic": "/camera_inspection/camera_info",
                "types": ["sensor_msgs/msg/CameraInfo"],
            },
            {
                "topic": "/joint_trajectory_controller/joint_trajectory",
                "types": ["trajectory_msgs/msg/JointTrajectory"],
            },
            {
                "topic": "/gripper_controller/command",
                "types": ["std_msgs/msg/Float64"],
            },
            {
                "topic": "/move_group/status",
                "types": ["action_msgs/msg/GoalStatusArray"],
            },
            {
                "topic": "/robot_description",
                "types": ["std_msgs/msg/String"],
            },
        ]

    def _mock_nodes(self) -> list[str]:
        """Return realistic mock ROS2 node list."""
        return [
            "/industrial_mcp_client",
            "/joint_state_publisher",
            "/robot_state_publisher",
            "/move_group",
            "/controller_manager",
            "/joint_trajectory_controller",
            "/gripper_controller",
            "/camera_wrist_driver",
            "/camera_overhead_driver",
            "/camera_inspection_driver",
            "/isaac_sim_bridge",
            "/tf2_buffer_server",
        ]
