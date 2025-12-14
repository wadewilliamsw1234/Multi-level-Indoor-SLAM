# Foxglove Visualization Setup

This guide explains how to visualize SLAM algorithms running on the ISEC dataset using Foxglove Studio.

## What You'll See

### LeGO-LOAM Visualization
- **3D Point Cloud**: LiDAR data from Ouster OS-128 building up in real-time
- **Trajectory**: Robot path growing as SLAM processes data
- **Map Building**: Accumulated point cloud map with intensity coloring

### ORB-SLAM3 Visualization
- **Stereo Camera Feeds**: Left (cam1) and right (cam3) images
- **Trajectory**: Camera path in 2D/3D view
- **Position Plots**: X, Y, Z coordinates over time

## Quick Start

### 1. Install Foxglove Studio

Download from: https://foxglove.dev/download

Or install via:
```bash
# Ubuntu/Debian
sudo snap install foxglove-studio

# Or download AppImage from website
```

### 2. Build Visualization Docker Images

```bash
cd ~/Dev/ros1/slam-benchmark

# Build LeGO-LOAM with visualization
docker build -t slam-benchmark/lego-loam-viz:latest -f docker/Dockerfile.lego-loam-viz .

# Build ORB-SLAM3 with visualization
docker build -t slam-benchmark/orb-slam3-viz:latest -f docker/Dockerfile.orb-slam3-viz .
```

### 3. Run with Visualization

**LeGO-LOAM:**
```bash
docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    -p 9090:9090 \
    --network=host \
    slam-benchmark/lego-loam-viz:latest \
    /root/run_lego_loam_viz.sh 5th_floor
```

**ORB-SLAM3:**
```bash
docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    -v ~/Dev/ros1/slam-benchmark/config:/config:ro \
    -p 9090:9090 \
    --network=host \
    slam-benchmark/orb-slam3-viz:latest \
    /root/run_orb_slam3_viz.sh 5th_floor
```

### 4. Connect Foxglove

1. Open Foxglove Studio
2. Click "Open connection"
3. Select "Rosbridge (ROS 1 & 2)"
4. Enter URL: `ws://localhost:9090`
5. Click "Open"

### 5. Load Layout

1. In Foxglove, click the layout menu (top-left)
2. Select "Import from file"
3. Choose:
   - `config/foxglove/lego_loam_layout.json` for LeGO-LOAM
   - `config/foxglove/orb_slam3_layout.json` for ORB-SLAM3

### 6. Start Playback

Return to your terminal and press **Enter** when prompted to start bag playback.

## Using Make Commands

For convenience, use the Makefile targets:

```bash
# Build visualization images
make build-viz

# Run LeGO-LOAM with visualization
make viz-lego-loam FLOOR=5th_floor

# Run ORB-SLAM3 with visualization
make viz-orb-slam3 FLOOR=5th_floor
```

## Available ROS Topics

### LeGO-LOAM Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/ouster/points` | PointCloud2 | Raw LiDAR point cloud |
| `/laser_cloud_surround` | PointCloud2 | Accumulated map |
| `/aft_mapped_to_init` | Odometry | Current pose estimate |
| `/imu/data` | Imu | IMU measurements |

### ORB-SLAM3 Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/camera_array/cam1/image_raw` | Image | Left camera |
| `/camera_array/cam3/image_raw` | Image | Right camera |
| `/orb_slam3/camera_pose` | PoseStamped | Current camera pose |
| `/orb_slam3/trajectory` | Path | Full trajectory path |

## Customizing the Visualization

### Foxglove Panel Types
- **3D**: Point clouds, poses, trajectories
- **Image**: Camera feeds
- **Plot**: Time-series data
- **Raw Messages**: Debug message contents
- **Topic List**: Browse available topics

### Adjusting Point Cloud Display
1. Click on the 3D panel
2. In the left sidebar, find the topic (e.g., `/ouster/points`)
3. Adjust:
   - Point size: 1-5 for visibility
   - Color mode: intensity, height, or flat color
   - Color map: turbo, viridis, etc.

### Changing Camera View
- **Orbit**: Click and drag
- **Pan**: Shift + drag
- **Zoom**: Scroll wheel
- **Reset**: Double-click

## Recording Videos

### Using Foxglove Recording
1. Click the "Record" button in Foxglove
2. Run the visualization
3. Stop recording when done
4. Export as video

### Using Screen Capture
```bash
# Install OBS or SimpleScreenRecorder
sudo apt install simplescreenrecorder

# Or use built-in screen recording (GNOME)
# Press Ctrl+Alt+Shift+R to start/stop
```

## Troubleshooting

### Can't connect to ws://localhost:9090
- Ensure Docker container is running
- Check that port 9090 is exposed: `docker ps`
- Try `ws://127.0.0.1:9090` instead

### No topics showing up
- Wait for rosbridge to fully start (~5 seconds)
- Click "Refresh" in the topic list panel
- Check container logs for errors

### Point cloud not visible
- Zoom out (scroll wheel)
- Check the topic is enabled in the 3D panel settings
- Increase point size

### Images not showing
- Verify camera topics exist in topic list
- Check that image transport is working
- May need to wait for bag playback to start

## Presentation Tips

1. **Use Full Screen**: Press F11 in Foxglove for immersive view
2. **Pre-load Layout**: Import layout before the presentation
3. **Dark Theme**: Foxglove's dark theme looks great on projectors
4. **Slow Playback**: Use 0.25x or 0.5x playback for detailed explanations
5. **Pause at Key Moments**: Show loop closures, challenging sections

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Docker Container                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   roscore   │  │  SLAM Node  │  │ rosbag play │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│         │                │                │         │
│         └────────────────┼────────────────┘         │
│                          │                          │
│                  ┌───────────────┐                  │
│                  │   rosbridge   │                  │
│                  │   port 9090   │                  │
│                  └───────────────┘                  │
└─────────────────────────│───────────────────────────┘
                          │ WebSocket
                          ▼
                ┌───────────────────┐
                │  Foxglove Studio  │
                │  (Desktop/Web)    │
                └───────────────────┘
```
