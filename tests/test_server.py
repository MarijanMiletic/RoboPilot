"""Tests for the MCP server tools — validates mock mode responses."""

import json
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcp_server.server import (
    execute_pick_and_place,
    get_camera_image,
    get_end_effector_pose,
    get_joint_positions,
    get_robot_state,
    get_system_health,
    inspect_workspace,
    list_ros2_nodes,
    list_ros2_topics,
    move_to_joint_positions,
    move_to_named_position,
    move_to_pose,
    stop_robot,
)


def _parse(result: str) -> dict:
    """Parse JSON string result from an MCP tool."""
    data = json.loads(result)
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    return data


# ── Robot State Tools ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_robot_state():
    result = _parse(await get_robot_state())
    assert result["robot_model"] == "UR10e"
    assert "joint_state" in result
    assert "end_effector_pose" in result
    assert "gripper" in result
    assert result["controller_status"] in ("idle", "moving", "stopped")


@pytest.mark.asyncio
async def test_get_joint_positions():
    result = _parse(await get_joint_positions())
    assert len(result["joint_names"]) == 6
    assert len(result["positions_rad"]) == 6
    assert len(result["positions_deg"]) == 6
    assert result["joint_names"][0] == "shoulder_pan_joint"


@pytest.mark.asyncio
async def test_get_end_effector_pose():
    result = _parse(await get_end_effector_pose())
    assert "position" in result
    assert "orientation" in result
    pos = result["position"]
    assert all(k in pos for k in ("x", "y", "z"))


# ── Motion Planning Tools ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_to_pose():
    result = _parse(await move_to_pose(0.5, 0.3, 1.0))
    assert result["success"] is True
    assert result["move_type"] == "cartesian"


@pytest.mark.asyncio
async def test_move_to_joint_positions():
    joints = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]
    result = _parse(await move_to_joint_positions(joints))
    assert result["success"] is True


@pytest.mark.asyncio
async def test_move_to_joint_positions_wrong_count():
    result = _parse(await move_to_joint_positions([0.0, 0.0]))
    assert result["success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_move_to_named_position_home():
    result = _parse(await move_to_named_position("home"))
    assert result["success"] is True
    assert result["named_position"] == "home"


@pytest.mark.asyncio
async def test_move_to_named_position_invalid():
    result = _parse(await move_to_named_position("nonexistent"))
    assert result["success"] is False


@pytest.mark.asyncio
async def test_execute_pick_and_place():
    result = _parse(await execute_pick_and_place(
        pick_x=0.5, pick_y=0.0, pick_z=0.78,
        place_x=-0.5, place_y=0.0, place_z=0.78,
    ))
    assert result["success"] is True
    assert result["operation"] == "pick_and_place"
    assert result["total_steps"] == 9


@pytest.mark.asyncio
async def test_stop_robot():
    result = _parse(await stop_robot())
    assert result["success"] is True
    assert result["action"] == "emergency_stop"
    assert result["all_velocities_zero"] is True


# ── Vision Tools ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_camera_image():
    result = _parse(await get_camera_image("wrist"))
    assert result["camera_name"] == "wrist"
    assert result["format"] == "jpeg"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_get_camera_image_invalid():
    result = _parse(await get_camera_image("nonexistent"))
    assert "error" in result


@pytest.mark.asyncio
async def test_inspect_workspace():
    result = _parse(await inspect_workspace("overhead"))
    assert result["camera_name"] == "overhead"
    assert "detections" in result


# ── System Introspection Tools ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_ros2_topics():
    result = _parse(await list_ros2_topics())
    assert result["topic_count"] > 0
    topics = [t["topic"] for t in result["topics"]]
    assert "/joint_states" in topics


@pytest.mark.asyncio
async def test_list_ros2_nodes():
    result = _parse(await list_ros2_nodes())
    assert result["node_count"] > 0
    assert "/move_group" in result["nodes"]


@pytest.mark.asyncio
async def test_get_system_health():
    result = _parse(await get_system_health())
    assert result["overall_status"] == "HEALTHY"
    assert "controller" in result
    assert "ros2" in result
    assert "sensors" in result
