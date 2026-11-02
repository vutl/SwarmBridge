"""Camera <-> ground geometry for CoPerception-UAV (town3 sample, nuScenes-style metadata).

Verified conventions (scripts/check_geometry.py):
  * calibrated_sensor holds the camera pose in the GLOBAL frame (translation ~ ego pose,
    rotation includes the UAV yaw), not sensor-to-ego as in real nuScenes.
  * Global frame is AirSim NED-like for poses: camera z ~ -42.6 means ~42.6 m above ground
    (ground plane z = 0). Vehicle boxes share the same x/y.
  * Camera body axes follow AirSim: x = optical axis, y = image right, z = image down.
  * Metadata intrinsics are for 1600x900; the PNGs are 800x450 -> f = 400, c = (400, 225).
  * The ground is NOT flat: ~11% of vehicles sit on elevated roads (box z up to ~8 m, z up).
    Images are reprojected onto z = 0, so anything elevated must be placed at its apparent
    position on that plane (`apparent_xy`) to stay aligned with the reprojected image.
"""
import numpy as np

W, H = 800, 450
F, CX, CY = 400.0, 400.0, 225.0


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_ground(u, v, cam):
    """u, v: arrays of pixel coords; cam: calibrated_sensor dict -> (x, y) on plane z = 0."""
    R, C = quat_to_R(cam["rotation"]), np.asarray(cam["translation"], float)
    rays = np.stack([np.ones_like(u, dtype=float), (u - CX) / F, (v - CY) / F], -1) @ R.T
    s = -C[2] / rays[..., 2]
    return C[0] + s * rays[..., 0], C[1] + s * rays[..., 1]


def ground_to_pixel(x, y, cam, z_up=0.0):
    """Points at height z_up (metres above the z = 0 ground, box convention) -> (u, v, in_front)."""
    R, C = quat_to_R(cam["rotation"]), np.asarray(cam["translation"], float)
    P = np.stack([x, y, -np.broadcast_to(np.asarray(z_up, float), np.shape(x))], -1) - C
    d = P @ R  # world -> camera body
    return CX + F * d[..., 1] / d[..., 0], CY + F * d[..., 2] / d[..., 0], d[..., 0] > 0


def apparent_xy(xy, z_up, cam):
    """Where a point at height z_up appears on the z = 0 plane seen from this camera:
    the camera ray through it, extended down to z = 0. xy: (..., 2)."""
    C = np.asarray(cam["translation"], float)
    h = -C[2]  # camera height above z = 0
    return C[:2] + (np.asarray(xy, float) - C[:2]) * (h / (h - np.asarray(z_up, float)))[..., None]
