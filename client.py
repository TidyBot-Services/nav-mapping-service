"""
TidyBot Nav Mapping Service — Python Client SDK

Usage:
    from client import NavMappingClient

    client = NavMappingClient("http://<backend-host>:8004")

    # Check health
    print(client.health())

    # Update map with depth image
    client.update("depth.png", pose=(1.0, 2.0, 0.5),
                  intrinsics=(fx, fy, cx, cy))

    # Get map as PNG bytes
    png_data = client.get_map(format="png")

    # Plan a path
    path = client.plan(start=(0, 0), goal=(2.0, 3.0))

    # Get exploration frontiers
    frontiers = client.get_frontiers()
"""

import base64
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Union


class NavMappingClient:
    """Client SDK for the TidyBot Nav Mapping Service."""

    def __init__(self, base_url: str = "http://localhost:8004", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check service health."""
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _encode_depth(self, image) -> str:
        """Encode depth image to base64 PNG."""
        if isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        elif isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".png", image)
            return base64.b64encode(buf.tobytes()).decode()
        return image

    def update(
        self,
        depth_image,
        pose: tuple,
        intrinsics: tuple,
        max_range: float = 5.0,
    ) -> dict:
        """
        Update the occupancy grid with a depth frame.

        Args:
            depth_image: Path, bytes, numpy array (uint16 mm), or base64 string.
            pose: (x, y, theta) robot pose in meters/radians.
            intrinsics: (fx, fy, cx, cy) camera intrinsics in pixels.
            max_range: Maximum depth range in meters.

        Returns:
            dict with update status and timing.
        """
        payload = {
            "depth_image": self._encode_depth(depth_image),
            "pose": {"x": pose[0], "y": pose[1], "theta": pose[2]},
            "intrinsics": {"fx": intrinsics[0], "fy": intrinsics[1],
                          "cx": intrinsics[2], "cy": intrinsics[3]},
            "max_range": max_range,
        }
        r = requests.post(f"{self.base_url}/update", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_map(self, format: str = "png") -> Union[bytes, dict]:
        """
        Retrieve current occupancy grid.

        Args:
            format: 'png' (returns bytes), 'json' (returns dict with grid), or 'info' (returns stats).
        """
        r = requests.get(f"{self.base_url}/map", params={"format": format}, timeout=self.timeout)
        r.raise_for_status()
        if format == "png":
            return r.content
        return r.json()

    def plan(
        self,
        start: tuple,
        goal: tuple,
        use_inflation: bool = True,
    ) -> dict:
        """
        Plan a collision-free path from start to goal.

        Args:
            start: (x, y) in world meters.
            goal: (x, y) in world meters.
            use_inflation: Whether to inflate obstacles by robot radius.

        Returns:
            dict with path_grid, path_world, length_m, planning_ms.
        """
        payload = {
            "start": list(start),
            "goal": list(goal),
            "use_inflation": use_inflation,
        }
        r = requests.post(f"{self.base_url}/plan", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_frontiers(self) -> dict:
        """
        Get frontier cells for exploration.

        Returns:
            dict with frontiers list, count, and detection_ms.
        """
        r = requests.get(f"{self.base_url}/frontiers", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self) -> dict:
        """Reset the occupancy grid."""
        r = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = NavMappingClient()
    print("Health:", client.health())
