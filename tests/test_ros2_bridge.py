"""Tests for the ROS2 bridge client — validates mock mode functionality."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ros2_bridge.client import UR10E_JOINT_NAMES, ROS2Client


@pytest.fixture
def client():
    """Fresh ROS2Client in mock mode."""
    return ROS2Client()


@pytest.mark.asyncio
async def test_connect(client):
    """Connection should succeed in mock mode."""
    assert await client.connect() is True


@pytest.mark.asyncio
async def test_get_joint_positions(client):
    result = await client.get_joint_positions()
    assert len(result["joint_names"]) == 6
    assert result["joint_names"] == UR10E_JOINT_NAMES
    assert len(result["positions_rad"]) == 6


@pytest.mark.asyncio
async def test_get_end_effector_pose(client):
    result = await client.get_end_effector_pose()
    assert "position" in result
    assert "orientation" in result
    pos = result["position"]
    assert all(k in pos for k in ("x", "y", "z"))


@pytest.mark.asyncio
async def test_get_robot_state(client):
    result = await client.get_robot_state()
    assert result["robot_model"] == "UR10e"
    assert "gripper" in result
    assert result["gripper"]["model"] == "Robotiq 2F-85"


@pytest.mark.asyncio
async def test_move_to_pose(client):
    result = await client.move_to_pose(0.5, 0.0, 1.0)
    assert result["success"] is True
    assert result["move_type"] == "cartesian"


@pytest.mark.asyncio
async def test_move_to_joint_positions(client):
    joints = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]
    result = await client.move_to_joint_positions(joints)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_move_to_joint_positions_wrong_count(client):
    result = await client.move_to_joint_positions([0.0])
    assert result["success"] is False


@pytest.mark.asyncio
async def test_move_to_named_position(client):
    result = await client.move_to_named_position("home")
    assert result["success"] is True
    assert result["named_position"] == "home"


@pytest.mark.asyncio
async def test_move_to_named_position_invalid(client):
    result = await client.move_to_named_position("does_not_exist")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_stop_robot(client):
    result = await client.stop_robot()
    assert result["success"] is True
    assert result["all_velocities_zero"] is True


@pytest.mark.asyncio
async def test_pick_and_place(client):
    result = await client.execute_pick_and_place(
        pick_x=0.5, pick_y=0.0, pick_z=0.78,
        place_x=-0.5, place_y=0.0, place_z=0.78,
    )
    assert result["success"] is True
    assert result["total_steps"] == 9
    assert result["steps_completed"] == 9


@pytest.mark.asyncio
async def test_get_camera_image(client):
    result = await client.get_camera_image("wrist")
    assert result["camera_name"] == "wrist"
    assert "image_base64" in result or "error" in result


@pytest.mark.asyncio
async def test_get_camera_image_invalid(client):
    result = await client.get_camera_image("nonexistent")
    assert "error" in result


@pytest.mark.asyncio
async def test_list_topics(client):
    topics = await client.list_topics()
    assert len(topics) > 0
    topic_names = [t["topic"] for t in topics]
    assert "/joint_states" in topic_names


@pytest.mark.asyncio
async def test_list_nodes(client):
    nodes = await client.list_nodes()
    assert len(nodes) > 0
    assert "/move_group" in nodes
