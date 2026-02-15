# Navigation & Occupancy Mapping Service

TidyBot backend service that maintains a 2D occupancy grid map from depth images and robot pose data. Provides A* path planning and frontier detection for autonomous exploration.

**Server:** `158.130.109.188:8004`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/update` | Submit depth frame + pose to update occupancy grid |
| GET | `/map` | Retrieve occupancy grid (PNG, JSON, or info) |
| POST | `/plan` | A* path planning between two world positions |
| GET | `/frontiers` | Detect frontier cells for exploration |
| POST | `/reset` | Reset map to unknown |

## Quick Start

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python main.py  # starts on port 8004
```

## Client SDK

```python
from client import NavMappingClient

client = NavMappingClient("http://158.130.109.188:8004")

# Update map with depth image
client.update(
    depth_image="depth.png",           # uint16 PNG (mm)
    pose=(1.0, 2.0, 0.5),             # x, y, theta
    intrinsics=(525, 525, 320, 240),   # fx, fy, cx, cy
)

# Get map
info = client.get_map(format="info")
png = client.get_map(format="png")

# Plan path
result = client.plan(start=(0, 0), goal=(2.0, 3.0))
print(result["path_world"], result["length_m"])

# Get frontiers
frontiers = client.get_frontiers()
```

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAP_SIZE` | 512 | Grid cells per side |
| `MAP_RESOLUTION` | 0.05 | Meters per cell |
| `ROBOT_RADIUS` | 0.15 | Robot radius for obstacle inflation (m) |
| `MAP_PERSIST_PATH` | `map_state.npz` | Path for map persistence |

## Depth Image Format

- **Format:** uint16 PNG
- **Units:** millimeters
- **Encoding:** Base64 for API transport
