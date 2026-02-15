"""
Navigation & Occupancy Mapping Service — TidyBot Backend
Maintains a 2D occupancy grid from depth images + robot pose.
Provides path planning (A*) and frontier detection for exploration.
"""

import base64
import heapq
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ─── Configuration ────────────────────────────────────────────────
MAP_SIZE = int(os.getenv("MAP_SIZE", 512))           # grid cells per side
RESOLUTION = float(os.getenv("MAP_RESOLUTION", 0.05))  # meters per cell
ROBOT_RADIUS = float(os.getenv("ROBOT_RADIUS", 0.15))  # meters, for inflation
MAP_PERSIST_PATH = os.getenv("MAP_PERSIST_PATH", "map_state.npz")
FREE = 0
UNKNOWN = 127
OCCUPIED = 255

# ─── Global State ─────────────────────────────────────────────────
occupancy_grid: np.ndarray = None  # uint8 HxW
origin = np.array([0.0, 0.0])     # world coords of grid cell (0,0)


def init_map():
    global occupancy_grid, origin
    if os.path.exists(MAP_PERSIST_PATH):
        data = np.load(MAP_PERSIST_PATH)
        occupancy_grid = data["grid"]
        origin = data["origin"]
        print(f"Loaded persisted map {occupancy_grid.shape} from {MAP_PERSIST_PATH}")
    else:
        occupancy_grid = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, dtype=np.uint8)
        origin = np.array([-MAP_SIZE * RESOLUTION / 2, -MAP_SIZE * RESOLUTION / 2])
        print(f"Initialized new {MAP_SIZE}x{MAP_SIZE} map, resolution={RESOLUTION}m/cell")


def save_map():
    np.savez_compressed(MAP_PERSIST_PATH, grid=occupancy_grid, origin=origin)


def world_to_grid(x: float, y: float):
    gx = int((x - origin[0]) / RESOLUTION)
    gy = int((y - origin[1]) / RESOLUTION)
    return gx, gy


def grid_to_world(gx: int, gy: int):
    x = gx * RESOLUTION + origin[0]
    y = gy * RESOLUTION + origin[1]
    return x, y


def in_bounds(gx: int, gy: int) -> bool:
    return 0 <= gx < occupancy_grid.shape[1] and 0 <= gy < occupancy_grid.shape[0]


# ─── Depth Projection ────────────────────────────────────────────
def project_depth_to_grid(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    pose_x: float, pose_y: float, pose_theta: float,
    max_range: float = 5.0,
    min_range: float = 0.1,
):
    """Project depth image (H,W uint16 mm) to 2D occupancy grid updates."""
    global occupancy_grid

    h, w = depth.shape
    # Subsample for speed
    step = max(1, min(h, w) // 120)
    vs, us = np.mgrid[0:h:step, 0:w:step]
    vs = vs.flatten()
    us = us.flatten()
    zs = depth[vs, us].astype(np.float64) / 1000.0  # mm → m

    valid = (zs > min_range) & (zs < max_range)
    us, vs, zs = us[valid], vs[valid], zs[valid]

    # Camera frame: x right, y down, z forward
    cam_x = (us - cx) * zs / fx
    cam_z = zs  # forward

    # Project to 2D robot frame (forward = x_robot, left = y_robot)
    cos_t, sin_t = np.cos(pose_theta), np.sin(pose_theta)
    world_x = pose_x + cos_t * cam_z - sin_t * cam_x
    world_y = pose_y + sin_t * cam_z + cos_t * cam_x

    # Mark occupied cells
    for wx, wy in zip(world_x, world_y):
        gx, gy = world_to_grid(wx, wy)
        if in_bounds(gx, gy):
            occupancy_grid[gy, gx] = OCCUPIED

    # Raycasting: mark free cells along rays from robot to each hit
    robot_gx, robot_gy = world_to_grid(pose_x, pose_y)
    # Subsample rays for speed
    ray_step = max(1, len(world_x) // 500)
    for wx, wy in zip(world_x[::ray_step], world_y[::ray_step]):
        gx, gy = world_to_grid(wx, wy)
        for pt in bresenham(robot_gx, robot_gy, gx, gy):
            px, py = pt
            if in_bounds(px, py) and (px, py) != (gx, gy):
                if occupancy_grid[py, px] != OCCUPIED:
                    occupancy_grid[py, px] = FREE


def bresenham(x0, y0, x1, y1):
    """Bresenham's line algorithm yielding (x,y) tuples."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


# ─── Obstacle Inflation ──────────────────────────────────────────
def get_inflated_grid() -> np.ndarray:
    """Return occupancy grid with obstacles inflated by robot radius."""
    inflate_cells = max(1, int(ROBOT_RADIUS / RESOLUTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_cells+1, 2*inflate_cells+1))
    occ_binary = (occupancy_grid == OCCUPIED).astype(np.uint8)
    inflated = cv2.dilate(occ_binary, kernel)
    result = occupancy_grid.copy()
    result[inflated == 1] = OCCUPIED
    return result


# ─── A* Path Planning ────────────────────────────────────────────
def astar(start, goal, grid):
    """A* on 8-connected grid. start/goal are (gx,gy). Returns list of (gx,gy) or None."""
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sy, sx] == OCCUPIED or grid[gy, gx] == OCCUPIED:
        return None

    SQRT2 = 1.414
    neighbors = [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-1,-1,SQRT2),(1,-1,SQRT2),(-1,1,SQRT2),(1,1,SQRT2)]

    def heuristic(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1])) + (1.414-1)*min(abs(a[0]-b[0]), abs(a[1]-b[1]))

    open_set = [(heuristic(start, goal), 0, sx, sy)]
    g_score = {(sx, sy): 0}
    came_from = {}
    closed = set()

    while open_set:
        f, g, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in closed:
            continue
        if (cx, cy) == (gx, gy):
            # Reconstruct
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path
        closed.add((cx, cy))
        for dx, dy, cost in neighbors:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in closed and grid[ny, nx] != OCCUPIED:
                ng = g + cost
                if ng < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (ng + heuristic((nx,ny), goal), ng, nx, ny))
    return None


# ─── Frontier Detection ──────────────────────────────────────────
def detect_frontiers() -> list:
    """Find frontier cells: free cells adjacent to unknown cells."""
    grid = occupancy_grid
    h, w = grid.shape
    free_mask = (grid == FREE)
    unknown_mask = (grid == UNKNOWN)

    # Dilate unknown by 1 pixel
    kernel = np.ones((3,3), np.uint8)
    unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8), kernel)

    frontier_mask = free_mask & (unknown_dilated == 1)
    ys, xs = np.where(frontier_mask)

    # Cluster frontiers
    if len(xs) == 0:
        return []

    # Return centroids of connected components
    frontier_img = frontier_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(frontier_img)
    frontiers = []
    for label_id in range(1, num_labels):
        pts = np.argwhere(labels == label_id)  # (y, x)
        if len(pts) < 3:
            continue
        cy, cx = pts.mean(axis=0)
        wx, wy = grid_to_world(int(cx), int(cy))
        frontiers.append({
            "grid": [int(cx), int(cy)],
            "world": [round(wx, 3), round(wy, 3)],
            "size": len(pts),
        })
    frontiers.sort(key=lambda f: f["size"], reverse=True)
    return frontiers


# ─── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_map()
    yield
    save_map()
    print("Map saved on shutdown.")


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot Nav Mapping Service",
    description="Occupancy grid mapping, A* path planning, and frontier detection for robot navigation.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length x (pixels)")
    fy: float = Field(..., description="Focal length y (pixels)")
    cx: float = Field(..., description="Principal point x (pixels)")
    cy: float = Field(..., description="Principal point y (pixels)")

class RobotPose(BaseModel):
    x: float = Field(..., description="Robot x position (meters)")
    y: float = Field(..., description="Robot y position (meters)")
    theta: float = Field(..., description="Robot heading (radians)")

class UpdateRequest(BaseModel):
    depth_image: str = Field(..., description="Base64-encoded uint16 PNG depth image (mm)")
    pose: RobotPose
    intrinsics: CameraIntrinsics
    max_range: float = Field(5.0, description="Max depth range in meters")

class PlanRequest(BaseModel):
    start: list[float] = Field(..., description="Start position [x, y] in world meters")
    goal: list[float] = Field(..., description="Goal position [x, y] in world meters")
    use_inflation: bool = Field(True, description="Use obstacle inflation for planning")

class PlanResponse(BaseModel):
    path_grid: list[list[int]] = Field(..., description="Path as grid coordinates [[gx,gy],...]")
    path_world: list[list[float]] = Field(..., description="Path as world coordinates [[x,y],...]")
    length_m: float
    planning_ms: float

class MapInfoResponse(BaseModel):
    shape: list[int]
    resolution: float
    origin: list[float]
    robot_radius: float
    free_cells: int
    occupied_cells: int
    unknown_cells: int

class HealthResponse(BaseModel):
    status: str
    map_shape: list[int]
    resolution: float
    robot_radius: float


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        map_shape=list(occupancy_grid.shape),
        resolution=RESOLUTION,
        robot_radius=ROBOT_RADIUS,
    )


@app.post("/update")
async def update_map(req: UpdateRequest):
    """Submit a depth frame + pose to update the occupancy grid."""
    try:
        img_bytes = base64.b64decode(req.depth_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        depth = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("Could not decode depth image")
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid depth image: {e}")

    t0 = time.perf_counter()
    project_depth_to_grid(
        depth,
        req.intrinsics.fx, req.intrinsics.fy,
        req.intrinsics.cx, req.intrinsics.cy,
        req.pose.x, req.pose.y, req.pose.theta,
        max_range=req.max_range,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    # Periodically persist
    save_map()

    return {
        "status": "ok",
        "update_ms": round(elapsed, 2),
        "map_shape": list(occupancy_grid.shape),
    }


@app.get("/map")
async def get_map(format: str = Query("png", description="Response format: 'png', 'json', or 'info'")):
    """Retrieve current occupancy grid."""
    if format == "info":
        return MapInfoResponse(
            shape=list(occupancy_grid.shape),
            resolution=RESOLUTION,
            origin=origin.tolist(),
            robot_radius=ROBOT_RADIUS,
            free_cells=int((occupancy_grid == FREE).sum()),
            occupied_cells=int((occupancy_grid == OCCUPIED).sum()),
            unknown_cells=int((occupancy_grid == UNKNOWN).sum()),
        )
    elif format == "json":
        return {
            "grid": occupancy_grid.tolist(),
            "resolution": RESOLUTION,
            "origin": origin.tolist(),
        }
    else:
        _, buf = cv2.imencode(".png", occupancy_grid)
        return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/plan", response_model=PlanResponse)
async def plan_path(req: PlanRequest):
    """Plan a collision-free path from start to goal using A*."""
    grid = get_inflated_grid() if req.use_inflation else occupancy_grid

    sx, sy = world_to_grid(req.start[0], req.start[1])
    gx, gy = world_to_grid(req.goal[0], req.goal[1])

    t0 = time.perf_counter()
    path = astar((sx, sy), (gx, gy), grid)
    elapsed = (time.perf_counter() - t0) * 1000

    if path is None:
        raise HTTPException(status_code=404, detail="No path found between start and goal")

    path_world = []
    length = 0.0
    for i, (px, py) in enumerate(path):
        wx, wy = grid_to_world(px, py)
        path_world.append([round(wx, 4), round(wy, 4)])
        if i > 0:
            prev = path_world[i-1]
            length += ((wx - prev[0])**2 + (wy - prev[1])**2) ** 0.5

    return PlanResponse(
        path_grid=[[p[0], p[1]] for p in path],
        path_world=path_world,
        length_m=round(length, 4),
        planning_ms=round(elapsed, 2),
    )


@app.get("/frontiers")
async def get_frontiers():
    """Return frontier cells for exploration."""
    t0 = time.perf_counter()
    frontiers = detect_frontiers()
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "frontiers": frontiers,
        "count": len(frontiers),
        "detection_ms": round(elapsed, 2),
    }


@app.post("/reset")
async def reset_map():
    """Reset the occupancy grid to unknown."""
    global occupancy_grid
    occupancy_grid = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, dtype=np.uint8)
    if os.path.exists(MAP_PERSIST_PATH):
        os.remove(MAP_PERSIST_PATH)
    return {"status": "ok", "message": "Map reset to unknown"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
