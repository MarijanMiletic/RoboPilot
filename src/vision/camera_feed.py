"""Camera capture and basic vision processing for Isaac Sim cameras."""

import base64
import io
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw
except ImportError:
    PILImage = None
    logger.warning("Pillow not available — camera feed will return raw data only")


class CameraFeed:
    """Manages camera image capture, encoding, and basic workspace inspection."""

    CAMERA_CONFIGS = {
        "wrist": {"resolution": (1280, 720), "frame_id": "tool0"},
        "overhead": {"resolution": (1920, 1080), "frame_id": "overhead_camera_link"},
        "inspection": {"resolution": (1920, 1080), "frame_id": "inspection_camera_link"},
    }

    def __init__(self) -> None:
        self._latest_frames: dict[str, np.ndarray] = {}

    def update_frame(self, camera_name: str, frame: np.ndarray) -> None:
        """Store latest frame from ROS2 subscription or Isaac Sim."""
        self._latest_frames[camera_name] = frame

    def get_available_cameras(self) -> list[str]:
        """Return list of configured camera names."""
        return list(self.CAMERA_CONFIGS.keys())

    def capture_image(self, camera_name: str) -> dict:
        """Capture and encode an image from the specified camera.

        Returns a dict with base64-encoded JPEG image and metadata.
        """
        if camera_name not in self.CAMERA_CONFIGS:
            return {
                "error": f"Unknown camera '{camera_name}'",
                "available": list(self.CAMERA_CONFIGS.keys()),
            }

        config = self.CAMERA_CONFIGS[camera_name]

        if camera_name in self._latest_frames:
            frame = self._latest_frames[camera_name]
            return self._encode_frame(camera_name, frame, config)

        # No live frame — generate a synthetic placeholder
        return self._generate_synthetic_image(camera_name, config)

    def inspect_workspace(self, camera_name: str) -> dict:
        """Perform workspace inspection using the specified camera.

        Returns image and any detected objects/anomalies.
        In mock mode, returns a synthetic image with simulated detections.
        """
        image_result = self.capture_image(camera_name)
        if "error" in image_result:
            return image_result

        # In Phase 3, real object detection would run here.
        # For now, return the image with mock detection metadata.
        detections = self._mock_detections(camera_name)

        return {
            **image_result,
            "inspection_type": "workspace_scan",
            "detections": detections,
            "recommendation": (
                "Image captured for VLM analysis. "
                "Pass this image to the LLM for visual inspection."
            ),
        }

    def _encode_frame(self, camera_name: str, frame: np.ndarray, config: dict) -> dict:
        """Encode a numpy frame as base64 JPEG."""
        if PILImage is None:
            return {"error": "Pillow not installed — cannot encode image"}

        try:
            img = PILImage.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            h, w = frame.shape[:2]
            return {
                "camera_name": camera_name,
                "format": "jpeg",
                "resolution": [w, h],
                "frame_id": config["frame_id"],
                "image_base64": b64,
                "size_bytes": buf.tell(),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Failed to encode frame from %s: %s", camera_name, e)
            return {"error": str(e)}

    def _generate_synthetic_image(self, camera_name: str, config: dict) -> dict:
        """Generate a synthetic camera image for mock/demo mode."""
        if PILImage is None:
            return {
                "camera_name": camera_name,
                "error": "Pillow not installed",
                "_mock": True,
            }

        w, h = config["resolution"]
        img = PILImage.new("RGB", (w, h), color=(35, 35, 45))
        draw = ImageDraw.Draw(img)

        # Grid
        for x in range(0, w, 80):
            draw.line([(x, 0), (x, h)], fill=(50, 50, 60), width=1)
        for y in range(0, h, 80):
            draw.line([(0, y), (w, y)], fill=(50, 50, 60), width=1)

        # Crosshair at center
        cx, cy = w // 2, h // 2
        draw.line([(cx - 40, cy), (cx + 40, cy)], fill=(0, 200, 0), width=2)
        draw.line([(cx, cy - 40), (cx, cy + 40)], fill=(0, 200, 0), width=2)
        draw.ellipse(
            [(cx - 60, cy - 60), (cx + 60, cy + 60)], outline=(0, 200, 0), width=1
        )

        # Simulated workpiece (small rectangle)
        if camera_name in ("overhead", "inspection"):
            wx, wy = cx - 25, cy - 15
            draw.rectangle([(wx, wy), (wx + 50, wy + 30)], fill=(160, 160, 170))
            draw.rectangle([(wx, wy), (wx + 50, wy + 30)], outline=(200, 200, 210), width=2)

        # HUD overlay
        draw.text((15, 15), f"CAM: {camera_name.upper()}", fill=(0, 255, 0))
        draw.text((15, 40), f"RES: {w}x{h}", fill=(160, 160, 160))
        draw.text((15, 60), f"FRAME: {config['frame_id']}", fill=(160, 160, 160))
        draw.text((15, 80), "MODE: SYNTHETIC", fill=(255, 200, 0))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "camera_name": camera_name,
            "format": "jpeg",
            "resolution": [w, h],
            "frame_id": config["frame_id"],
            "image_base64": b64,
            "size_bytes": buf.tell(),
            "timestamp": time.time(),
            "_mock": True,
        }

    def _mock_detections(self, camera_name: str) -> list[dict]:
        """Return simulated object detections based on camera position."""
        if camera_name == "overhead":
            return [
                {
                    "label": "workpiece",
                    "confidence": 0.96,
                    "bbox": [580, 480, 660, 540],
                    "class_id": 0,
                },
                {
                    "label": "conveyor_belt",
                    "confidence": 0.99,
                    "bbox": [200, 400, 1720, 680],
                    "class_id": 1,
                },
            ]
        elif camera_name == "inspection":
            return [
                {
                    "label": "workpiece",
                    "confidence": 0.98,
                    "bbox": [750, 400, 1170, 680],
                    "class_id": 0,
                },
            ]
        else:  # wrist
            return [
                {
                    "label": "workpiece",
                    "confidence": 0.91,
                    "bbox": [500, 350, 780, 550],
                    "class_id": 0,
                },
            ]
