"""UAV3D camera geometry (found empirically against _seg.png, see axis_search.py):
world -> camera:  d = R^T ((X - t) * [1, 1, -1]);  cv = (d_y, d_z, d_x);  uv = K cv / cv_z
where R = quaternion(calibrated_sensor.rotation) (w,x,y,z), t = calibrated_sensor.translation (ego_pose is identity)."""
import numpy as np

FLIP = np.array([1.0, 1.0, -1.0])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_to_cam(X, cal):
    """X: (N,3) world points -> (N,3) OpenCV camera coords (z = depth)."""
    R = quat_to_R(cal["rotation"])
    d = ((X - np.array(cal["translation"])) * FLIP) @ R
    return d[:, [1, 2, 0]]


def project(X, cal):
    P = world_to_cam(X, cal)
    K = np.array(cal["camera_intrinsic"])
    uv = (P @ K.T)[:, :2] / P[:, 2:3]
    return uv, P[:, 2]


def box_corners(a, z_is_bottom=False):
    w, l, h = a["size"]
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2 + (h / 2 if z_is_bottom else 0)
    return (quat_to_R(a["rotation"]) @ np.vstack([x, y, z])).T + np.array(a["translation"])
