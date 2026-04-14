"""Isaac Sim Factory Cell Scene Builder.

Programmatically builds the industrial robotic cell scene with:
- UR10e robot on a pedestal
- Input/output conveyor belts
- Inspection station
- 3 cameras (wrist, overhead, inspection)
- Safety zone markers
- Proper lighting

Run from Isaac Sim's Script Editor or via:
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh factory_cell.py
"""

import logging
import sys

logger = logging.getLogger(__name__)

# Guard Isaac Sim imports — script may be loaded for inspection outside of Sim
try:
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.stage_utils import add_reference_to_stage
    from pxr import Gf, UsdGeom, UsdLux

    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    logger.info("Isaac Sim modules not available — scene builder will not execute")


# ── Scene parameters (from config/factory_layout.yaml) ──────────────────

CELL_SIZE = (8.0, 6.0, 3.5)  # length, width, height

ROBOT_BASE_POS = (0.0, 0.0, 0.85)
PEDESTAL_SIZE = (0.4, 0.4, 0.85)

CONVEYOR_INPUT = {
    "start": (2.0, 0.0, 0.75),
    "end": (0.8, 0.0, 0.75),
    "width": 0.5,
    "length": 1.2,
}
CONVEYOR_OUTPUT = {
    "start": (-0.8, 0.0, 0.75),
    "end": (-2.0, 0.0, 0.75),
    "width": 0.5,
    "length": 1.2,
}

INSPECTION_POS = (0.0, -1.0, 0.75)
INSPECTION_SIZE = (0.6, 0.6, 0.05)

# UR10e URDF/USD nucleus path (adjust to your Isaac Sim installation)
UR10E_USD_PATH = "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"

CAMERAS = {
    "wrist": {
        "parent": "/World/ur10e/tool0",
        "translation": (0.0, 0.0, 0.05),
        "rotation": (0.0, 0.0, 0.0),
        "focal_length": 18.0,
        "resolution": (1280, 720),
    },
    "overhead": {
        "parent": "/World",
        "translation": (0.0, 0.0, 2.5),
        "rotation": (-90.0, 0.0, 0.0),
        "focal_length": 12.0,
        "resolution": (1920, 1080),
    },
    "inspection": {
        "parent": "/World",
        "translation": (0.3, -1.0, 1.5),
        "rotation": (-45.0, 0.0, 0.0),
        "focal_length": 24.0,
        "resolution": (1920, 1080),
    },
}


def build_factory_cell() -> None:
    """Build the complete factory cell scene in Isaac Sim."""
    if not ISAAC_AVAILABLE:
        logger.error("Isaac Sim not available. Run this inside Isaac Sim's Python environment.")
        return

    logger.info("Building factory cell scene...")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    _create_ground_plane(world)
    _create_pedestal(world)
    _create_robot(stage)
    _create_conveyors(world)
    _create_inspection_station(world)
    _create_cameras(stage)
    _create_safety_markers(world)
    _create_lighting(stage)
    _create_sample_workpieces(world)

    _setup_ros2_bridge(stage)

    world.reset()
    logger.info("Factory cell scene built successfully")


def _create_ground_plane(world: "World") -> None:
    """Add a ground plane to the scene."""
    world.scene.add_default_ground_plane()
    logger.info("Ground plane added")


def _create_pedestal(world: "World") -> None:
    """Create the robot pedestal/table."""
    world.scene.add(
        FixedCuboid(
            prim_path="/World/pedestal",
            name="robot_pedestal",
            position=(0.0, 0.0, PEDESTAL_SIZE[2] / 2),
            scale=(PEDESTAL_SIZE[0], PEDESTAL_SIZE[1], PEDESTAL_SIZE[2]),
            color=(0.3, 0.3, 0.35),
        )
    )
    logger.info("Pedestal created at origin, height=%.2fm", PEDESTAL_SIZE[2])


def _create_robot(stage) -> None:
    """Load the UR10e robot USD from the Nucleus server."""
    robot_prim_path = "/World/ur10e"
    try:
        add_reference_to_stage(
            usd_path=UR10E_USD_PATH,
            prim_path=robot_prim_path,
        )
        # Position robot on top of pedestal
        robot_xform = XFormPrim(prim_path=robot_prim_path)
        robot_xform.set_world_pose(
            position=ROBOT_BASE_POS,
            orientation=(1.0, 0.0, 0.0, 0.0),  # wxyz identity
        )
        logger.info("UR10e loaded at %s", ROBOT_BASE_POS)
    except Exception as e:
        logger.error(
            "Failed to load UR10e from %s: %s. "
            "Ensure Isaac Sim Nucleus assets are available.",
            UR10E_USD_PATH,
            e,
        )


def _create_conveyors(world: "World") -> None:
    """Create input and output conveyor belts as visual cuboids."""
    for name, conv in [("input_conveyor", CONVEYOR_INPUT), ("output_conveyor", CONVEYOR_OUTPUT)]:
        cx = (conv["start"][0] + conv["end"][0]) / 2
        cy = conv["start"][1]
        cz = conv["start"][2] / 2
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=(cx, cy, cz),
                scale=(conv["length"], conv["width"], conv["start"][2]),
                color=(0.15, 0.15, 0.2),
            )
        )
        # Belt surface (thin visual layer on top)
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/{name}_belt",
                name=f"{name}_belt",
                position=(cx, cy, conv["start"][2]),
                scale=(conv["length"], conv["width"] - 0.05, 0.02),
                color=(0.1, 0.1, 0.12),
            )
        )
    logger.info("Conveyors created")


def _create_inspection_station(world: "World") -> None:
    """Create the quality inspection platform."""
    world.scene.add(
        FixedCuboid(
            prim_path="/World/inspection_station",
            name="inspection_station",
            position=(INSPECTION_POS[0], INSPECTION_POS[1], INSPECTION_POS[2] / 2),
            scale=(INSPECTION_SIZE[0], INSPECTION_SIZE[1], INSPECTION_POS[2]),
            color=(0.5, 0.5, 0.55),
        )
    )
    # Inspection surface
    world.scene.add(
        VisualCuboid(
            prim_path="/World/inspection_surface",
            name="inspection_surface",
            position=(INSPECTION_POS[0], INSPECTION_POS[1], INSPECTION_POS[2]),
            scale=(INSPECTION_SIZE[0], INSPECTION_SIZE[1], INSPECTION_SIZE[2]),
            color=(0.85, 0.85, 0.9),
        )
    )
    logger.info("Inspection station created at %s", INSPECTION_POS)


def _create_cameras(stage) -> None:
    """Create camera prims for wrist, overhead, and inspection cameras."""
    for cam_name, cfg in CAMERAS.items():
        prim_path = f"{cfg['parent']}/camera_{cam_name}"
        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.GetFocalLengthAttr().Set(cfg["focal_length"])
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*cfg["translation"]))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*cfg["rotation"]))

        logger.info("Camera '%s' created at %s", cam_name, prim_path)


def _create_safety_markers(world: "World") -> None:
    """Create visual safety zone markers (light curtain indicators)."""
    safety_positions = [
        (1.5, 0.0, 1.0),   # Right boundary
        (-1.5, 0.0, 1.0),  # Left boundary
    ]
    for i, pos in enumerate(safety_positions):
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/safety_marker_{i}",
                name=f"safety_marker_{i}",
                position=pos,
                scale=(0.02, 4.0, 2.0),
                color=(1.0, 0.2, 0.0),  # Orange-red safety color
            )
        )
    logger.info("Safety markers created")


def _create_lighting(stage) -> None:
    """Set up scene lighting for camera image quality."""
    # Main overhead dome light
    dome_light = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome_light.GetIntensityAttr().Set(500.0)

    # Directional key light
    dist_light = UsdLux.DistantLight.Define(stage, "/World/key_light")
    dist_light.GetIntensityAttr().Set(2000.0)
    xform = UsdGeom.Xformable(dist_light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    # Inspection ring light (spot light above inspection station)
    spot = UsdLux.SphereLight.Define(stage, "/World/inspection_light")
    spot.GetIntensityAttr().Set(3000.0)
    spot.GetRadiusAttr().Set(0.1)
    xform = UsdGeom.Xformable(spot.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.0, 1.3))

    logger.info("Lighting configured")


def _create_sample_workpieces(world: "World") -> None:
    """Add sample workpieces on the input conveyor for demonstration."""
    workpiece_positions = [
        (1.6, 0.0, 0.78),
        (1.3, 0.0, 0.78),
        (1.0, 0.0, 0.78),
    ]
    for i, pos in enumerate(workpiece_positions):
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/workpiece_{i}",
                name=f"workpiece_{i}",
                position=pos,
                scale=(0.05, 0.05, 0.03),
                mass=0.15,
                color=(0.6, 0.6, 0.65),
            )
        )
    logger.info("Sample workpieces placed on input conveyor")


def _setup_ros2_bridge(stage) -> None:
    """Configure ROS2 bridge OmniGraph for topic publishing.

    This sets up the Action Graph nodes for:
    - Joint State Publisher (/joint_states)
    - Camera publishers (/camera_*/color/image_raw)
    - Clock Publisher (/clock)
    - TF Publisher (/tf)

    Note: Requires omni.isaac.ros2_bridge extension to be enabled.
    """
    try:
        import omni.graph.core as og

        # Create the ROS2 action graph
        og.Controller.edit(
            {"graph_path": "/World/ROS2_Bridge", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                    ("CameraWrist", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("CameraOverhead", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("CameraInspection", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "CameraWrist.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "CameraOverhead.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "CameraInspection.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishJointState.inputs:targetPrim", "/World/ur10e"),
                    ("PublishJointState.inputs:topicName", "joint_states"),
                    ("PublishTF.inputs:targetPrims", ["/World/ur10e"]),
                    ("CameraWrist.inputs:cameraPrim", "/World/ur10e/tool0/camera_wrist"),
                    ("CameraWrist.inputs:topicName", "camera_wrist/color/image_raw"),
                    ("CameraWrist.inputs:type", "rgb"),
                    ("CameraOverhead.inputs:cameraPrim", "/World/camera_overhead"),
                    ("CameraOverhead.inputs:topicName", "camera_overhead/color/image_raw"),
                    ("CameraOverhead.inputs:type", "rgb"),
                    (
                        "CameraInspection.inputs:cameraPrim",
                        "/World/camera_inspection",
                    ),
                    (
                        "CameraInspection.inputs:topicName",
                        "camera_inspection/color/image_raw",
                    ),
                    ("CameraInspection.inputs:type", "rgb"),
                ],
            },
        )
        logger.info("ROS2 Bridge OmniGraph configured")
    except Exception as e:
        logger.warning(
            "ROS2 Bridge setup skipped: %s. Enable omni.isaac.ros2_bridge extension.", e
        )


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not ISAAC_AVAILABLE:
        print(
            "ERROR: This script must be run inside NVIDIA Isaac Sim's Python environment.\n"
            "Use: ~/.local/share/ov/pkg/isaac-sim-*/python.sh factory_cell.py"
        )
        sys.exit(1)
    build_factory_cell()
