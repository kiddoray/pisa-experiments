"""Camera motion helper utilities extracted from MOVI worker without side effects.

These functions are pure samplers; they require a RNG object with .rand and .uniform methods (NumPy RandomState or Generator).
"""
from __future__ import annotations
import numpy as np
import kubric as kb

__all__ = [
    "get_linear_camera_motion_start_end",
    "get_linear_lookat_motion_start_end",
]

def get_linear_camera_motion_start_end(rng, movement_speed: float,
                                       inner_radius: float = 3.,
                                       outer_radius: float = 4.,
                                       z_offset: float = 0.1):
    """Sample a linear path which starts and ends within a half-sphere shell.

    Returns (camera_start, camera_end) as np.ndarray of shape (3,).
    """
    while True:
        camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                     outer_radius,
                                                                     z_offset))
        direction = rng.rand(3) - 0.5
        # Avoid zero norm
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        movement = direction / norm * movement_speed
        camera_end = camera_start + movement
        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z_offset):
            return camera_start, camera_end


def get_linear_lookat_motion_start_end(rng, inner_radius: float = 1.0,
                                       outer_radius: float = 4.0):
    """Sample a linear path which goes through the workspace center.

    Returns (camera_start, camera_end) as np.ndarray of shape (3,).
    """
    while True:
        camera_through = np.array(
            kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
        )
        # sample one endpoint of the trajectory
        while True:
            camera_start = np.array(
                kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
            )
            if camera_start[-1] < inner_radius:
                break
        continuation = rng.rand(1) * 0.5
        camera_end = camera_through + continuation * (camera_through - camera_start)
        # Randomly swap for augmentation
        if rng.rand(1)[0] < 0.5:
            camera_start, camera_end = camera_end, camera_start
        return camera_start, camera_end
