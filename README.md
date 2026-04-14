# Industrial Robot MCP Server

[MCP](https://modelcontextprotocol.io/) server that bridges LLMs with a UR10e robotic arm via NVIDIA Isaac Sim. Control the robot, capture camera feeds, and run diagnostics — all through natural language.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Desktop / LLM                  │
│                     (MCP Client)                         │
└──────────────────────┬──────────────────────────────────┘
                       │ stdio (MCP Protocol)
┌──────────────────────▼──────────────────────────────────┐
│               Industrial Robot MCP Server                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  MCP Tools  │  │  Resources   │  │   Prompts     │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │                                                │
│  ┌──────▼──────┐                    ┌───────────────┐  │
│  │ ROS2 Bridge │                    │ Camera Feed   │  │
│  │  (client)   │                    │  (Vision)     │  │
│  └──────┬──────┘                    └───────────────┘  │
└─────────┼───────────────────────────────────────────────┘
          │ TCP Socket (localhost:54321)
┌─────────▼───────────────────────────────────────────────┐
│              NVIDIA Isaac Sim 4.5 Bridge                 │
│  ┌─────────┐  ┌───────────┐  ┌────────┐               │
│  │  UR10e  │  │ Conveyors │  │Cameras │               │
│  │  Robot  │  │ (In/Out)  │  │  (x3)  │               │
│  └─────────┘  └───────────┘  └────────┘               │
└─────────────────────────────────────────────────────────┘
```

## MCP Tools (13 total)

### Robot State
| Tool | Description |
|------|-------------|
| `get_robot_state()` | Full robot state: joints, EE pose, gripper, controller |
| `get_joint_positions()` | Joint angles in radians and degrees |
| `get_end_effector_pose()` | Cartesian pose (position + quaternion) |

### Motion Planning
| Tool | Description |
|------|-------------|
| `move_to_pose(x,y,z,qx,qy,qz,qw)` | Cartesian motion planning |
| `move_to_joint_positions(joints)` | Joint-space motion |
| `move_to_named_position(name)` | Predefined positions: home, ready, pick_approach, place_approach, inspection |
| `execute_pick_and_place(...)` | Full 9-step pick-and-place sequence |
| `stop_robot()` | Emergency stop — zero velocity all joints |

### Vision
| Tool | Description |
|------|-------------|
| `get_camera_image(camera)` | Capture RGB image as base64 JPEG |
| `inspect_workspace(camera)` | Object detection on workspace |

### System Introspection
| Tool | Description |
|------|-------------|
| `list_ros2_topics()` | All active topics with message types |
| `list_ros2_nodes()` | All running nodes |
| `get_system_health()` | Dashboard: controller, sensors, connection mode |

## Quick Start (Windows)

### One-Click Setup

```bash
setup.bat
```

This creates a Python 3.10 venv, installs Isaac Sim, PyTorch (CUDA), and all dependencies.

### Run the Full Pipeline

**Terminal 1** — Start Isaac Sim with UR10 robot:
```bash
run_isaac.bat
```

**Terminal 2** — Open Claude Desktop. The MCP server starts automatically.

Talk to the robot in natural language:
- "Move the robot to the home position"
- "What are the current joint positions?"
- "Pick up the object at x=0.5 y=0 z=0.78 and place it at x=-0.5 y=0 z=0.78"

### Mock Mode (no Isaac Sim)

```bash
pip install -e ".[dev]"
pytest
PYTHONPATH=src python -m mcp_server.server
```

### Docker

```bash
cd docker
docker compose up mcp-server
```

## Connection Modes

The server automatically selects the best available backend:

1. **Isaac Sim** (TCP bridge on port 54321) — full simulation with physics
2. **ROS2** (rclpy) — connect to real or simulated robot via ROS2
3. **Mock** — synthetic data, no external dependencies

## Prerequisites

| Component | Version | Required |
|-----------|---------|----------|
| Python | 3.10+ | Yes |
| FastMCP | 2.0+ | Yes |
| NVIDIA Isaac Sim | 4.5 | Optional (mock mode available) |
| NVIDIA GPU | RTX series | For Isaac Sim |
| ROS2 Humble | Humble | Optional |

## Project Structure

```
industrial-ros2-mcp/
├── src/
│   ├── mcp_server/server.py         # MCP server — 13 tools, 2 resources, 2 prompts
│   ├── ros2_bridge/client.py        # Async client (Isaac/ROS2/mock fallback)
│   ├── isaac_bridge/client.py       # TCP socket client for Isaac Sim bridge
│   └── vision/camera_feed.py        # Camera capture and workspace inspection
├── config/
│   ├── factory_layout.yaml          # Factory cell layout definition
│   └── robot_config.yaml            # UR10e specs, named positions, gripper
├── isaac_sim/
│   └── bridge_server.py             # Isaac Sim bridge (UR10 + TCP socket server)
├── docker/
│   ├── docker-compose.yaml
│   └── Dockerfile.mcp
├── tests/                           # pytest suite — mock mode validation
├── setup.bat                        # One-click Windows setup
├── run_isaac.bat                    # Launch Isaac Sim bridge
├── pyproject.toml
└── README.md
```

## Tech Stack

- **Robot**: Universal Robots UR10e (simulated)
- **Simulation**: NVIDIA Isaac Sim 4.5
- **Protocol**: MCP (Model Context Protocol) via FastMCP
- **Bridge**: TCP socket (newline-delimited JSON)

## License

MIT
