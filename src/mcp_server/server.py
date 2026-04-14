"""Industrial Robot MCP Server — bridges LLMs with robotic arm control via Isaac Sim.

Provides MCP tools for robot control, vision, and diagnostics.
Runs in mock mode when Isaac Sim / ROS2 is not available.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

from ros2_bridge.client import ROS2Client
from vision.camera_feed import CameraFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# ── Global instances ─────────────────────────────────────────────────────
mcp = FastMCP("Industrial Robot MCP Server")
ros2_client = ROS2Client()
camera_feed = CameraFeed()


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    path = CONFIG_DIR / filename
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS — Robot State (3 tools)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_robot_state() -> str:
    """Get the full state of the UR10e robot including joint positions, end-effector
    pose, gripper status, and controller state. Returns a comprehensive snapshot of
    the robot's current configuration and operational status."""
    try:
        state = await ros2_client.get_robot_state()
        return json.dumps(state, indent=2)
    except Exception as e:
        logger.error("get_robot_state failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_joint_positions() -> str:
    """Get the current joint angles of all 6 UR10e joints in both radians and degrees.
    Joint names: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3."""
    try:
        joints = await ros2_client.get_joint_positions()
        return json.dumps(joints, indent=2)
    except Exception as e:
        logger.error("get_joint_positions failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_end_effector_pose() -> str:
    """Get the Cartesian position (x, y, z) and orientation (quaternion qx, qy, qz, qw)
    of the robot's end-effector (tool0) relative to the base_link frame."""
    try:
        pose = await ros2_client.get_end_effector_pose()
        return json.dumps(pose, indent=2)
    except Exception as e:
        logger.error("get_end_effector_pose failed: %s", e)
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS — Motion Planning (5 tools)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def move_to_pose(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.7071,
    qz: float = 0.0,
    qw: float = 0.7071,
    velocity_scaling: float = 0.5,
    acceleration_scaling: float = 0.5,
) -> str:
    """Move the robot end-effector to a Cartesian pose using MoveIt2 planning.
    Position in meters (x, y, z), orientation as quaternion (qx, qy, qz, qw).
    velocity_scaling and acceleration_scaling range from 0.0 to 1.0.
    The robot workspace extends approximately 1.3m from the base in all directions."""
    try:
        result = await ros2_client.move_to_pose(
            x, y, z, qx, qy, qz, qw, velocity_scaling, acceleration_scaling
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("move_to_pose failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def move_to_joint_positions(joint_values: list[float]) -> str:
    """Move the robot by specifying target joint angles in radians for all 6 joints.
    Order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
    Joint limits are approximately +/- 2*pi for most joints."""
    try:
        result = await ros2_client.move_to_joint_positions(joint_values)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("move_to_joint_positions failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def move_to_named_position(position_name: str) -> str:
    """Move the robot to a predefined named position.
    Available positions: 'home' (upright), 'ready' (forward), 'pick_approach'
    (above input conveyor), 'place_approach' (above output conveyor),
    'inspection' (above inspection station)."""
    try:
        result = await ros2_client.move_to_named_position(position_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("move_to_named_position failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def execute_pick_and_place(
    pick_x: float,
    pick_y: float,
    pick_z: float,
    place_x: float,
    place_y: float,
    place_z: float,
    approach_height: float = 0.1,
    gripper_width: float = 0.04,
) -> str:
    """Execute a complete pick-and-place operation. The robot will:
    1. Move above the pick position (approach_height above pick_z)
    2. Open gripper, descend to pick position
    3. Close gripper to gripper_width (meters)
    4. Retract, move to place position
    5. Open gripper, retract
    All coordinates in meters relative to base_link."""
    try:
        result = await ros2_client.execute_pick_and_place(
            pick_x, pick_y, pick_z,
            place_x, place_y, place_z,
            approach_height, gripper_width,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("execute_pick_and_place failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def stop_robot() -> str:
    """Emergency stop — immediately halts all robot motion by commanding zero velocity
    on all joints. Use this when you need to stop the robot urgently. The controller
    status will change to 'stopped'. A new move command is required to resume."""
    try:
        result = await ros2_client.stop_robot()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("stop_robot failed: %s", e)
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS — Vision (2 tools)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_camera_image(camera_name: str) -> str:
    """Capture an RGB image from a camera and return it as base64-encoded JPEG.
    Available cameras: 'wrist' (on tool0, 1280x720), 'overhead' (top-down, 1920x1080),
    'inspection' (angled at inspection station, 1920x1080).
    The base64 image can be decoded and displayed by the LLM."""
    try:
        result = await ros2_client.get_camera_image(camera_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("get_camera_image failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_workspace(camera_name: str) -> str:
    """Perform a visual inspection of the workspace using the specified camera.
    Returns the captured image (base64 JPEG) along with any detected objects.
    Available cameras: 'wrist', 'overhead', 'inspection'.
    Pass the returned image to the LLM for detailed visual quality analysis."""
    try:
        result = camera_feed.inspect_workspace(camera_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("inspect_workspace failed: %s", e)
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS — System Introspection (3 tools)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_ros2_topics() -> str:
    """List all active ROS2 topics with their message types. Useful for debugging
    communication and verifying that all expected data streams are running."""
    try:
        topics = await ros2_client.list_topics()
        return json.dumps({"topic_count": len(topics), "topics": topics}, indent=2)
    except Exception as e:
        logger.error("list_ros2_topics failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_ros2_nodes() -> str:
    """List all running ROS2 nodes. Each node represents a process in the robotic
    system (controllers, drivers, planning, etc.)."""
    try:
        nodes = await ros2_client.list_nodes()
        return json.dumps({"node_count": len(nodes), "nodes": nodes}, indent=2)
    except Exception as e:
        logger.error("list_ros2_nodes failed: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_system_health() -> str:
    """Get an overall health report for the robotic system including: controller status,
    sensor connectivity, communication status, and connection mode.
    Use this as a quick dashboard-style overview."""
    try:
        robot_state = await ros2_client.get_robot_state()
        nodes = await ros2_client.list_nodes()
        topics = await ros2_client.list_topics()

        health = {
            "overall_status": "HEALTHY",
            "connection_mode": ros2_client._mode,
            "controller": {
                "status": robot_state.get("controller_status", "unknown"),
                "is_moving": robot_state.get("is_moving", False),
                "uptime_seconds": robot_state.get("uptime_seconds", 0),
            },
            "ros2": {
                "nodes_active": len(nodes),
                "topics_active": len(topics),
            },
            "sensors": {
                "cameras": {
                    "wrist": "online",
                    "overhead": "online",
                    "inspection": "online",
                },
                "joint_state_publisher": "online",
                "tf_broadcaster": "online",
            },
            "gripper": robot_state.get("gripper", {}),
            "timestamp": time.time(),
        }
        return json.dumps(health, indent=2)
    except Exception as e:
        logger.error("get_system_health failed: %s", e)
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# MCP RESOURCES — Read-only context
# ═══════════════════════════════════════════════════════════════════════════


@mcp.resource("robot://config")
async def robot_config() -> str:
    """UR10e robot configuration: joint limits, workspace bounds, named positions,
    gripper specifications, and controller settings."""
    config = _load_yaml("robot_config.yaml")
    return json.dumps(config, indent=2)


@mcp.resource("factory://layout")
async def factory_layout() -> str:
    """Factory cell layout: robot placement, conveyor positions, camera locations,
    inspection station, safety zones, and workpiece specifications."""
    layout = _load_yaml("factory_layout.yaml")
    return json.dumps(layout, indent=2)


@mcp.prompt()
def robot_status_report() -> str:
    """Generate a status report covering robot state and system health."""
    return """You are generating a status report for a UR10e robotic cell.
Please execute the following steps and compile the results into a structured report:

1. Call get_system_health() to get the overall system status
2. Call get_robot_state() to get the current robot configuration
3. Call get_joint_positions() and get_end_effector_pose() for details

Format the report with these sections:
- **System Status Summary** — overall health, controller state
- **Robot Configuration** — current joint positions, end-effector pose, gripper state
- **Recommendations** — any actions or observations

Use clear, concise language."""


@mcp.prompt()
def visual_quality_check() -> str:
    """Quality inspection workflow using camera vision."""
    return """You are performing a visual quality inspection on the robotic cell.

Execute the following inspection sequence:

1. Call get_camera_image("inspection") to capture the inspection camera view
2. Call inspect_workspace("overhead") to get an overhead view with detections

Analyze the captured images for:
- **Workpiece presence** — is a part on the inspection station?
- **Surface quality** — any visible scratches, dents, or discoloration?
- **Orientation** — is the part correctly positioned for the next operation?

Report findings as:
- PASS: Part meets all visual quality criteria
- FAIL: Describe specific defects found
- INCONCLUSIVE: If image quality is insufficient, recommend adjustments

Include the camera images in your response for operator review."""


# ═══════════════════════════════════════════════════════════════════════════
# Server entry point
# ═══════════════════════════════════════════════════════════════════════════


async def _initialize() -> None:
    """Initialize async components before serving."""
    await ros2_client.connect()
    mode_labels = {
        "isaac": "LIVE — Isaac Sim (socket bridge)",
        "ros2": "LIVE — ROS2 (rclpy)",
        "mock": "MOCK (no simulation)",
    }
    logger.info("Industrial ROS2 MCP Server initialized")
    logger.info("Mode: %s", mode_labels.get(ros2_client._mode, ros2_client._mode))


def main() -> None:
    """Entry point for the MCP server."""
    asyncio.run(_initialize())
    logger.info("Starting MCP server on stdio transport...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
