"""
TidyBot Nav Mapping Service — Python Client SDK

Usage:
    from services.nav_mapping.client import NavMappingClient

    client = NavMappingClient()
    client.update(depth_bytes, pose=(x, y, theta), intrinsics=(fx, fy, cx, cy))
    png = client.get_map(format="png")
    path = client.plan(start=(0, 0), goal=(2.0, 3.0))
"""

import base64
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional, Union

import numpy as np


class NavMappingClient:
    """Client SDK for the TidyBot Nav Mapping Service."""

    def __init__(self, host: str = "http://158.130.109.188:8004", timeout: float = 30.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str, params: Optional[dict] = None):
        url = f"{self.host}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in ct:
                return json.loads(raw)
            return raw  # binary (e.g. PNG)

    def health(self) -> dict:
        """Check service health."""
        return self._get("/health")

    @staticmethod
    def _encode_depth(image) -> str:
        """Encode depth image to base64."""
        if isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".png", image)
            return base64.b64encode(buf.tobytes()).decode()
        elif isinstance(image, (str, Path)):
            p = Path(image)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode()
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
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
            Dict with update status and timing.
        """
        payload = {
            "depth_image": self._encode_depth(depth_image),
            "pose": {"x": pose[0], "y": pose[1], "theta": pose[2]},
            "intrinsics": {"fx": intrinsics[0], "fy": intrinsics[1],
                          "cx": intrinsics[2], "cy": intrinsics[3]},
            "max_range": max_range,
        }
        return self._post("/update", payload)

    def get_map(self, format: str = "png") -> Union[bytes, dict]:
        """
        Retrieve current occupancy grid.

        Args:
            format: 'png' (returns bytes), 'json' (returns dict with grid), or 'info' (returns stats).
        """
        return self._get("/map", params={"format": format})

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
        return self._post("/plan", payload)

    def get_frontiers(self) -> dict:
        """Get frontier cells for exploration."""
        return self._get("/frontiers")

    def reset(self) -> dict:
        """Reset the occupancy grid."""
        return self._post("/reset", {})


if __name__ == "__main__":
    client = NavMappingClient()
    print("Health:", client.health())
