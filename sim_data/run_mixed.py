import os
import time
import sys

# =============================================================================
# GPU SELECTION - MUST BE SET BEFORE IMPORTING BPY/BLENDER
# =============================================================================
# For Eevee: We need to set EGL device selection before Blender initializes
# For Cycles: Use KUBRIC_GPU_ID to select specific GPU within Blender
#
# KUBRIC_GPU_ID: Which GPU index Blender should use (for Cycles)
# CUDA_VISIBLE_DEVICES: Which GPUs are visible to the process
#
# For Eevee with EGL (headless), use __EGL_DEVICE_ID to select GPU
# This must be set before bpy is imported

# if "KUBRIC_GPU_ID" in os.environ:
#     gpu_id = os.environ["KUBRIC_GPU_ID"]
#     # Set CUDA_VISIBLE_DEVICES if not already set
#     if "CUDA_VISIBLE_DEVICES" not in os.environ:
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
#         print(f"[GPU Setup] Auto-setting CUDA_VISIBLE_DEVICES={gpu_id} based on KUBRIC_GPU_ID")
#     # Set EGL device ID for Eevee headless rendering
#     if "__EGL_DEVICE_ID" not in os.environ:
#         os.environ["__EGL_DEVICE_ID"] = gpu_id
#         print(f"[GPU Setup] Auto-setting __EGL_DEVICE_ID={gpu_id} for Eevee EGL rendering")

# Ensure Kubric uses GPU
if "KUBRIC_USE_GPU" not in os.environ:
    os.environ["KUBRIC_USE_GPU"] = "1"

import pyquaternion as pyquat
import bpy

# =============================================================================
# EEVEE GPU DEVICE SELECTION (after bpy import)
# =============================================================================
# def setup_eevee_gpu():
#     """Configure Eevee to use specific GPU device."""
#     try:
#         prefs = bpy.context.preferences.system
#         # Get available GPU devices for Eevee
#         # In Blender 4.x, Eevee uses the system's OpenGL/Vulkan device
#         # The device selection depends on the GPU backend
        
#         gpu_backend = getattr(prefs, 'gpu_backend', None)
#         print(f"[GPU Setup] Blender GPU backend: {gpu_backend}")
        
#         # Log available compute devices
#         cycles_prefs = bpy.context.preferences.addons.get('cycles')
#         if cycles_prefs:
#             cycles_prefs = cycles_prefs.preferences
#             cycles_prefs.get_devices()
#             print("[GPU Setup] Available GPU devices:")
#             for i, dev in enumerate(cycles_prefs.devices):
#                 print(f"  [{i}] {dev.name} (type={dev.type}, use={dev.use})")
#     except Exception as e:
#         print(f"[GPU Setup] Warning: Could not configure Eevee GPU: {e}")

# # Call setup after bpy import
# setup_eevee_gpu()

# print("Rendering Engine:", bpy.context.scene.render.engine)
from mathutils import Vector, Matrix, Quaternion, Euler
import hashlib
import string
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import sys
import sys; sys.path = ["kubric"] + sys.path
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.core import materials
import numpy as np
from camera_motion_utils import (
    get_linear_camera_motion_start_end,
    get_linear_lookat_motion_start_end,
)

from loguru import logger
import logging
from skimage import io 
import json 
import scipy.spatial.transform as transform
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R
from config import Configuration
import uuid
from math import radians
from kubric_utils import *
import copy
import signal
import tarfile
import shutil
# import moviepy.editor as mp
import subprocess
from PIL import Image

# --- JSON helpers -----------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy and Blender math types to serializable forms."""
    def default(self, o):
        # NumPy scalars and arrays
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()

        # Blender mathutils types
        try:
            if isinstance(o, (Vector, Euler)):
                return list(o)
            if isinstance(o, Quaternion):
                # Quaternion(w, x, y, z)
                return [o.w, o.x, o.y, o.z]
        except Exception:
            pass

        # Generic "has tolist" objects
        if hasattr(o, "tolist"):
            try:
                return o.tolist()
            except Exception:
                pass

        # Bytes -> utf-8 string
        if isinstance(o, (bytes, bytearray)):
            try:
                return o.decode("utf-8", errors="ignore")
            except Exception:
                return str(o)

        return super().default(o)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def time_limit(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Reset the alarm
            return result
        return wrapper
    return decorator

parser = kb.ArgumentParser()
parser.add_argument(
    "--output_dir", 
    type=str,    
    default="output"
)
parser.add_argument(
    "--scenario", 
    type=str, 
    default="fall"
)
parser.add_argument(
    "--stage1",
    type=str,
    default="fall",
    help="Scenario to run during the first half of 'combined' scenario, fall or throw"
)
parser.add_argument(
    "--stage2",
    type=str,
    default="center",
    help="Scenario to run during the second half of 'combined' scenario, objects move toward center or at random directions"
)

parser.add_argument(
    "--stage2_speed",
    type=float,
    default=30.0,
    help="Speed of objects during the second half of 'combined' scenario"
)

parser.add_argument(
    "--stage2_force",
    type=str,
    default='force',
    help="Force application mode during the second half of 'combined' scenario: 'force' or 'impulse'"
)
parser.add_argument(
    "--stage2_force_duration",
    type=float,
    default=0.3,
    help="Duration (s) of force application during the second half of 'combined' scenario"
)
parser.add_argument(
    "--combined_influx_count",
    type=int,
    default=3,
    help="Number of offscreen objects to enter the scene during the second half of 'combined' scenario"
)
parser.add_argument(
    "--combined_influx_speed",
    type=float,
    default=30.0,
    help="Speed (m/s) of offscreen objects entering the scene in 'combined' scenario"
)
parser.add_argument(
    "--combined_margin",
    type=float,
    default=0.1,
    help="Margin (m) beyond the dynamic spawn region to place offscreen entrants"
)

# Slope scenario parameters
parser.add_argument(
    "--slope_angle_deg",
    type=float,
    default=25.0,
    help="Slope angle in degrees (default: 25)"
)
parser.add_argument(
    "--slope_length",
    type=float,
    default=3.0,
    help="Length of the slope along the incline direction in meters (default: 3.0)"
)
parser.add_argument(
    "--slope_width",
    type=float,
    default=2.0,
    help="Width of the slope perpendicular to incline in meters (default: 2.0)"
)
parser.add_argument(
    "--slope_thickness",
    type=float,
    default=0.08,
    help="Thickness of the slope panel in meters (default: 0.08)"
)
parser.add_argument(
    "--slope_friction",
    type=float,
    default=0.3,
    help="Friction coefficient of the slope surface (default: 0.3, lower = more slippery)"
)
parser.add_argument(
    "--slope_restitution",
    type=float,
    default=0.1,
    help="Restitution (bounciness) of the slope surface (default: 0.1)"
)

parser.add_argument(
    "--video_id",
    type=str,
    default=str(uuid.uuid4())
)
parser.add_argument(
    "--scene_save_path",
    type=str,
    default=""
)
parser.add_argument(
    "--save_mp4",
    action="store_true",
    default=False
)
parser.add_argument(
    "--save_gif",
    action="store_true",
    default=False
)
parser.add_argument(
    "--tar",
    action="store_true",
    default=False
)
parser.add_argument(
    "--num_threads",
    type=int,
    default=None
)
parser.add_argument(
    "--objects_split",
    type=str,
    default="train",
    choices=["train", "test"]
)
parser.add_argument(
    "--backgrounds_split",
    type=str,
    default="train",
    choices=["train", "test"]
)
parser.add_argument(
    "--max_motion_blur",
    type=float,
    default=0.0
)
parser.add_argument(
    "--kubasic_assets",
    type=str,
    default="gs://kubric-public/assets/KuBasic/KuBasic.json"
)
parser.add_argument(
    "--hdri_assets", 
    type=str,
    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
)
parser.add_argument(
    "--gso_assets", 
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json"
)
parser.add_argument(
    "--gso_physics_lookup",
    type=str,
    default="",
    help="Optional path to precomputed GSO object physics lookup JSON; defaults to <gso_assets_basename>_physics.json if empty"
)
parser.add_argument(
    "--background_mode",
    type=str,
    choices=["dome", "world"],
    default="dome",
    help="How to apply the HDRI background: 'dome' (geometry with inside texture) or 'world' (World Environment texture in Cycles)"
)
parser.add_argument(
    "--save_state", 
    action="store_true"
)
parser.add_argument(
    "--focal_length", 
    type=float, 
    default=35.0
)
parser.add_argument(
    "--sensor_width", 
    type=float, 
    default=32
)
parser.add_argument(
    "--camera_motion",
    type=str,
    choices=["static", "fixed_random", "linear", "linear_pan", "orbit"],
    default="static",
    help="Camera movement mode: static (original), fixed_random (random pose looking at origin), linear (linear position interpolation look-at origin), linear_pan (linear position + linear look-at), orbit (circular motion around origin)"
)
parser.add_argument(
    "--max_camera_movement",
    type=float,
    default=4.0,
    help="Maximum linear movement distance for camera start->end when using linear or linear_pan modes"
)
parser.add_argument(
    "--camera_orbit_cycles",
    type=float,
    default=0.5,
    help="Number of full revolutions the camera completes during the animation in orbit mode (e.g. 0.5 = half circle)"
)
parser.add_argument(
    "--camera_orbit_radius_min",
    type=float,
    default=3.0,
    help="Minimum radius for orbit mode sampling"
)
parser.add_argument(
    "--camera_orbit_radius_max",
    type=float,
    default=5.0,
    help="Maximum radius for orbit mode sampling"
)
parser.add_argument(
    "--camera_orbit_height_min",
    type=float,
    default=0.6,
    help="Minimum camera height for orbit mode"
)
parser.add_argument(
    "--camera_orbit_height_max",
    type=float,
    default=0.8,
    help="Maximum camera height for orbit mode"
)
# multi-view rendering parameters
parser.add_argument(
    "--num_views",
    type=int,
    default=1,
    help="Number of camera viewpoints to render per scene (1 = current behavior)"
)
parser.add_argument(
    "--view_sampling",
    type=str,
    choices=["random", "circle", "vertical"],
    default="circle",
    help="Strategy to sample camera viewpoints around the scene center"
)
parser.add_argument(
    "--ground_mode",
    type=str,
    choices=["none", "shadow", "textured"],
    default="textured",
    help="Ground rendering mode: none (no ground plane), shadow (shadow catcher), textured (visible PBR-like plane)"
)
parser.add_argument(
    "--ground_size",
    type=float,
    default=10.0,
    help="Side length of square ground plane in meters (only if ground_mode != none)"
)
parser.add_argument(
    "--ground_color",
    type=str,
    default="#7f7f7f",
    help="Base color for textured ground as hex RGB (e.g. #808080)"
)
parser.add_argument(
    "--ground_texture",
    type=str,
    default="",
    help="Optional path to a ground albedo texture image; if provided, overrides ground_color"
)
parser.add_argument(
    "--ground_normal",
    type=str,
    default="",
    help="Optional path to a ground normal map image (Non-Color)"
)
parser.add_argument(
    "--ground_roughness",
    type=str,
    default="",
    help="Optional path to a ground roughness map image (Non-Color)"
)
parser.add_argument(
    "--ground_displacement",
    type=str,
    default="",
    help="Optional path to a ground height/displacement map; connected to Output Displacement in Cycles and as bump fallback in Eevee"
)
parser.add_argument(
    "--ground_texture_manifest",
    type=str,
    default="/mnt/solo/kiddoray/Texture_haven.json",
    help="Path to a Poly Haven texture manifest JSON; if present, textured ground will randomly pick an asset with Diffuse, NormalGL, Roughness, Displacement maps"
)
parser.add_argument(
    "--ground_use_dome_material",
    action="store_true",
    default=False,
    help="If set and ground_mode=textured, assign the same material as the dome to the ground plane"
)
parser.add_argument(
    "--enable_physics_ground",
    action="store_true",
    default=True,
    help="Add an invisible static collider at z=0 so objects can rest on the ground regardless of visual ground mode"
)
parser.add_argument(
    "--physics_ground_thickness",
    type=float,
    default=0.02,
    help="Thickness of the invisible physics ground cube (meters); top will align to z=0"
)
parser.add_argument(
    "--physics_ground_size",
    type=float,
    default=None,
    help="Side length of the invisible physics ground; defaults to ground_size if not set"
)

parser.add_argument(
    "--engine",
    type=str,
    choices=["cycles", "eevee", "eevee_next"],
    default=None,
    help="Render engine to use: 'cycles' or 'eevee'. If omitted, defaults to 'cycles'.",
)
parser.add_argument(
    "--flow_render_mode",
    type=str,
    choices=["auto", "same", "split"],
    default="auto",
    help="How to render optical flow: 'same' = together with main passes using chosen engine; 'split' = if engine=eevee, render main passes in Eevee then temporarily switch to Cycles for flow; 'auto' = behave like 'split' only when engine=eevee (ignored if engine=cycles)."
)
parser.add_argument(
    "--flow_samples",
    type=int,
    default=1,
    help="Samples per pixel to use for the optical flow Cycles pass when splitting (lower for speed)."
)
parser.add_argument(
    "--disable_flow",
    action="store_true",
    default=False,
    help="If set, do not request forward/backward optical flow layers at all."
)

# parser.set_defaults(resolution="1024x1024", frame_rate=16, frame_end=32)
parser.set_defaults(resolution="1024x1024")
args = parser.parse_args()

VELOCITY_RANGE = [(-2., -2., 0.), (2., 2., 0.)]
THROW_SPAWN_REGION = [(-3., -3., 0.5), (3., 3., 0.8)]
KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear",
                   "torus_knot", "sponge", "spot", "teapot", "suzanne")

class Intuitive:
    def __init__(self, args):
        (
            self.scene, 
            self.rng, 
            _, 
            self.scratch_dir, 
            self.renderer, 
            self.simulator
        ) = self.configure(args)

        self.video_id = args.video_id
        self.args = args
        self.base_dir = args.output_dir
        self.output_dir = os.path.join(args.output_dir, self.video_id)
        (
            self.kubasic, 
            self.gso, 
            self.hdri_source
        ) = self.load_scene(args)
        # Load object physics lookup (GSO) if available
        try:
            self._object_physics_lookup = {}
            self._load_object_physics_lookup()
        except Exception as e:
            logging.warning(f"Failed to load object physics lookup: {e}")
        self.cfg = Configuration.get_config_for_scenario(self.args.scenario)
        
        if self.cfg.min_num_dynamic_objects == self.cfg.max_num_dynamic_objects:
            self.num_dynamic_objects = self.cfg.min_num_dynamic_objects
        else:
            self.num_dynamic_objects = self.rng.randint(
                self.cfg.min_num_dynamic_objects,
                self.cfg.max_num_dynamic_objects,
            )
        self.cfg.dynamic_spawn_region[0][2] += self.cfg.min_drop_height
        self.metadata = {
            "object_split": args.objects_split,
            "background_split": args.backgrounds_split,
        }
        self.num_threads = args.num_threads
        # For 'combined' scenario: objects spawned mid-simulation (kept separate to avoid affecting camera placement)
        self.late_dynamic_objects = []
                
    
    def configure(self, args):
        scene, rng, output_dir, scratch_dir = kb.setup(args)
        simulator = PyBullet(
            scene, scratch_dir
        )
        renderer = Blender(
            scene, 
            scratch_dir,
            use_denoising=True, 
            samples_per_pixel=64,
            motion_blur=args.max_motion_blur
        )
        # Select engine: --engine takes precedence (default to cycles if omitted)
        eng = getattr(args, "engine", None)
        try:
            if eng is None:
                bpy.context.scene.render.engine = "BLENDER_EEVEE"
            elif eng.lower() == "cycles":
                bpy.context.scene.render.engine = "CYCLES"
            elif eng.lower() == "eevee_next":
                bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
            else:
                bpy.context.scene.render.engine = "BLENDER_EEVEE"
            logging.info(f"Render engine set to {bpy.context.scene.render.engine} (engine={eng})")
        except Exception as e:
            logging.warning(f"Failed to set render engine: {e}")
        return scene, rng, output_dir, scratch_dir, renderer, simulator

    

    def load_scene(self, args):
        hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
        kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
        gso = kb.AssetSource.from_manifest(args.gso_assets)
        return kubasic, gso, hdri_source

    def _load_object_physics_lookup(self):
        """Load precomputed GSO object physics lookup file.
        Derives path from args.gso_physics_lookup or <gso_assets_basename>_physics.json.
        Expects schema with key 'lookup'.
        """
        path_override = getattr(self.args, 'gso_physics_lookup', '') or ''
        if path_override:
            candidate = path_override
        else:
            base = os.path.splitext(os.path.basename(getattr(self.args, 'gso_assets', 'GSO.json')))[0]
            # Write next to output_dir or current CWD? Prefer same directory as run invocation (CWD)
            candidate = os.path.join(os.getcwd(), base + '_physics.json')
            if not os.path.exists(candidate):
                # Also try alongside manifest path if local
                gso_path = getattr(self.args, 'gso_assets', '')
                if gso_path and os.path.exists(gso_path):
                    root = os.path.splitext(gso_path)[0]
                    alt = root + '_physics.json'
                    if os.path.exists(alt):
                        candidate = alt
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._object_physics_lookup = data.get('lookup', {}) or {}
            logging.info(f"Loaded object physics lookup '{candidate}' (entries={len(self._object_physics_lookup)})")
        except FileNotFoundError:
            logging.warning(f"Object physics lookup file not found: {candidate}")
        except Exception as e:
            logging.warning(f"Failed reading object physics lookup '{candidate}': {e}")

    def _apply_object_physics(self, obj, asset_id: str):
        """Apply friction/restitution from lookup to a newly created object.
        Push changes to PyBullet dynamics if possible and record metadata.
        """
        try:
            rec = self._object_physics_lookup.get(asset_id)
            if not rec:
                return
            mu = float(rec.get('friction', 0.6))
            e = float(rec.get('restitution', 0.2))
            try:
                obj.friction = mu
                obj.restitution = e
            except Exception:
                pass
            # Push to physics backend
            try:
                body_id = obj.linked_objects[self.simulator]
                self.simulator._physics_client.changeDynamics(int(body_id), -1, lateralFriction=mu, restitution=e)
            except Exception:
                pass
            # Metadata for later analysis
            try:
                if not hasattr(obj, 'metadata') or obj.metadata is None:
                    obj.metadata = {}
                obj.metadata['physics'] = {
                    'friction': mu,
                    'restitution': e,
                    'category': rec.get('category'),
                    'tokens': rec.get('tokens'),
                }
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"Failed to apply object physics for {asset_id}: {e}")

    def _apply_initial_velocities(self, objs, velocity_mode: str = 'center'):
        """
        Ensure the current obj.velocity is applied to the PyBullet backend so the
        motion actually happens during the main simulation.
        """
        # Make sure all bodies are registered before setting velocities
        try:
            for _ in range(2):
                self.simulator._physics_client.stepSimulation()
        except Exception:
            pass

        for o in objs:
            # Fallback: if velocity wasn't set, sample one (XY plane)
            v = getattr(o, "velocity", None)
            try:
                if v is None:
                    if velocity_mode == 'center':
                        # Direction from object position to scene center
                        pos = getattr(o, "position", (0., 0., 0.))
                        dir_xy = (-pos[0], -pos[1])
                        n = (dir_xy[0]*dir_xy[0] + dir_xy[1]*dir_xy[1]) ** 0.5
                        if n > 1e-6:
                            dir_xy = (dir_xy[0] / n, dir_xy[1] / n)
                        else:
                            dir_xy = (1.0, 0.0)
                        speed = self.rng.uniform(VELOCITY_RANGE[0][0], VELOCITY_RANGE[1][0])
                        v = (dir_xy[0] * speed, dir_xy[1] * speed, 0.0)
                    else:
                        v = self.rng.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
                    try:
                        o.velocity = v
                    except Exception:
                        pass

                # Push to physics backend
                try:
                    obj_idx = o.linked_objects[self.simulator]
                except KeyError:
                    self.simulator._physics_client.stepSimulation()
                    obj_idx = o.linked_objects[self.simulator]
                self.simulator._physics_client.resetBaseVelocity(obj_idx, (float(v[0]), float(v[1]), 0.0), (0.0, 0.0, 0.0))
            except Exception as e:
                logging.warning(f"Failed to apply velocity for object {getattr(o, 'name', 'unknown')}: {e}")

        # Small tick to commit
        try:
            self.simulator._physics_client.stepSimulation()
        except Exception:
            pass

    def _accelerate_objects_with_force(self, objs, target_speed: float, dir_mode: str, duration_s: float, start_frame: int, force_mode: str) -> int:
        """Apply a short pulse of external force to accelerate objects realistically.

        - Uses PyBullet applyExternalForce in WORLD frame at the object's COM.
        - Force magnitude per object: F = m * (target_speed / duration_s)
        - Direction: from current position toward scene origin unless obj.velocity
          is already set (then use its XY direction).
        - Steps the simulator one frame at a time for the given duration.

                Extended modes:
                - Impulse mode (args.force_mode == 'impulse'): apply a single
                    linear momentum change Δp = m * target_speed * impulse_multiplier along the chosen
                    direction using PyBullet's applyImpulse (preferred) or applyExternalForce fallback.
                    After the impulse, step a number of settling frames (impulse_settle_frames) to allow
                    motion to evolve.

        Returns the number of frames stepped (may be 0 on failure).
        """
        if not objs:
            return 0
        try:
            import pybullet as p  # local import to avoid hard dependency at module import time
        except Exception as e:
            logging.warning(f"PyBullet import failed for external force mode: {e}; falling back to velocity reset.")
            try:
                self._apply_initial_velocities(objs, velocity_mode=dir_mode)
            except Exception:
                pass
            return 0

        try:
            frame_rate = float(getattr(self.args, 'frame_rate', 16))
            if frame_rate <= 0:
                frame_rate = 16.0
        except Exception:
            frame_rate = 16.0
        try:
            duration_s = float(duration_s)
        except Exception:
            duration_s = 0.3
        steps = max(1, int(round(duration_s * frame_rate)))

        # User knobs to strengthen external force
        try:
            force_multiplier = float(getattr(self.args, 'force_multiplier', 2.0))
        except Exception:
            force_multiplier = 2.0
        try:
            ensure_break_static = bool(getattr(self.args, 'force_min_overcome_friction', True))
        except Exception:
            ensure_break_static = True
        try:
            break_factor = float(getattr(self.args, 'force_break_static_factor', 1.1))
        except Exception:
            break_factor = 1.1
        # Gravity estimate (used to estimate friction threshold μ*m*g)
        try:
            g = float(getattr(self.scene, 'gravity', (0, 0, -9.81))[2])
            g = abs(g) if isinstance(g, (int, float)) else 9.81
        except Exception:
            g = 9.81

        try:
            impulse_multiplier = float(getattr(self.args, 'impulse_multiplier', 1.0))
        except Exception:
            impulse_multiplier = 1.0
        try:
            impulse_settle_frames = int(getattr(self.args, 'impulse_settle_frames', max(1, steps)))
        except Exception:
            impulse_settle_frames = max(1, steps)
        
        # If user explicitly sets impulse_mode, we override steps to 1 for application then settle
        if force_mode == 'impulse':
            steps = 1  # only one application phase; settling handled separately

        client = getattr(self.simulator, '_physics_client', None)
        if client is None:
            logging.warning("Physics client unavailable; cannot apply external forces.")
            try:
                self._apply_initial_velocities(objs, velocity_mode=dir_mode)
            except Exception:
                pass
            return 0

        # Pre-compute mass and force vector per object
        obj_info = []
        for o in objs:
            try:
                body_id = o.linked_objects[self.simulator]
            except Exception:
                # attempt a registration tick
                try:
                    client.stepSimulation()
                    body_id = o.linked_objects[self.simulator]
                except Exception:
                    continue

            try:
                dyn = client.getDynamicsInfo(int(body_id), -1)
                print(f"DYN INFO of object {body_id}:", dyn)
                mass = float(dyn[0]) if dyn and len(dyn) > 0 else 1.0
                lat_mu = float(dyn[1]) if dyn and len(dyn) > 1 else 0.6
                if mass <= 0:
                    mass = 1.0
            except Exception:
                mass = 1.0
                lat_mu = 0.6

            # Consider ground friction override if present
            try:
                mu_ground = float(getattr(self, '_ground_friction_override', lat_mu))
                mu_eff = max(lat_mu, mu_ground)
            except Exception:
                mu_eff = lat_mu

            # Determine desired push direction in XY (supports modes)
            dir_xy = None

            v = getattr(o, 'velocity', None)
            if v is not None and dir_xy is None:
                try:
                    vx, vy = float(v[0]), float(v[1])
                    n = (vx*vx + vy*vy) ** 0.5
                    if n > 1e-6:
                        dir_xy = (vx / n, vy / n)
                except Exception:
                    dir_xy = None

            # Direction mode (per object). Don't mutate the function arg across iterations.
            mode_for_obj = 'center'
            try:
                mode_for_obj = str(dir_mode).lower()
            except Exception:
                pass
            if mode_for_obj not in ('center', 'random', 'mixed'):
                mode_for_obj = 'center'

            if mode_for_obj == 'mixed':
                # Choose independently per object; keep the original dir_mode unchanged
                try:
                    mode_for_obj = str(self.rng.choice(['center', 'random']))
                except Exception:
                    import random as _py_rnd
                    mode_for_obj = 'random' if _py_rnd.random() < 0.5 else 'center'

            if mode_for_obj == 'random':
                try:
                    # Ensure scalar draw for broader RNG compatibility
                    th = self.rng.uniform(0.0, 2*np.pi)
                    try:
                        th = float(th)  # coerce ndarray/Decimal to float if needed
                    except Exception:
                        pass
                except Exception:
                    import random as _py_rnd
                    th = _py_rnd.random() * 2*np.pi
                dir_xy = (float(np.cos(th)), float(np.sin(th)))
            else:
                # 'center' mode: push toward origin in XY
                try:
                    x, y, _ = o.position
                    to_center = (-float(x), -float(y))
                    n = (to_center[0]**2 + to_center[1]**2) ** 0.5
                    if n < 1e-6:
                        dir_xy = (-1.0, 0.0)
                    else:
                        dir_xy = (to_center[0]/n, to_center[1]/n)
                except Exception:
                    dir_xy = (-1.0, 0.0)
            
            # Force magnitude to reach target speed over duration
            try:
                speed = float(self.rng.uniform(target_speed, 1.5*target_speed))
                a = float(speed) / max(1e-6, duration_s)
            except Exception:
                a = 2.0  # reasonable small acceleration
            # Base force from desired acceleration and user multiplier
            Fmag = mass * a * max(0.0, force_multiplier)
            # Ensure force is strong enough to overcome static friction if requested
            if ensure_break_static:
                F_fric_threshold = mu_eff * mass * g * max(1.0, break_factor)
                if Fmag < F_fric_threshold:
                    Fmag = F_fric_threshold
            Fx, Fy = (Fmag * dir_xy[0], Fmag * dir_xy[1])
            obj_info.append((o, int(body_id), Fx, Fy, dir_xy, mu_eff, mass, speed))
            print(f"Prepared force for body {body_id}: mass={mass:.3f}, mu_eff={mu_eff:.3f}, dir={dir_xy}, F=({Fx:.3f}, {Fy:.3f}), speed={speed:.3f}")

        if not obj_info:
            logging.warning("No valid bodies to apply external forces to; falling back to velocity reset.")
            try:
                self._apply_initial_velocities(objs, velocity_mode=dir_mode)
            except Exception:
                pass
            return 0

        current = int(start_frame)

        if force_mode == 'impulse':
            # Single impulse application then settling frames
            for o, body_id, Fx, Fy, obj_dir, mu_eff, mass, speed in obj_info:
                try:
                    try:
                        px, py, pz = o.position
                    except Exception:
                        px, py, pz = 0.0, 0.0, 0.0
                    # mass_est = max(1e-6, client.getDynamicsInfo(body_id, -1)[0]) if client else 1.0
                    # # Desired momentum change ~ m * target_speed (scaled) in direction
                    # speed = self.rng.uniform(0.6*target_speed, 1.5*target_speed)
                    # a = float(speed) / max(1e-6, duration_s)
                    
                    # Base impulse magnitude: m * target_speed * impulse_multiplier
                    I_base = mass * float(speed) * max(0.0, impulse_multiplier)
                    # Friction break check: convert to equivalent average force F_equiv = I_base / (1/frame_rate)
                    if ensure_break_static:
                        F_fric_threshold = mu_eff * mass * g * max(1.0, break_factor)
                        F_equiv = I_base * frame_rate  # impulse over one frame
                        if F_equiv < F_fric_threshold:
                            I_base = F_fric_threshold / frame_rate
                    Ix, Iy = I_base * obj_dir[0], I_base * obj_dir[1]
                    applied = False
                    # Try direct impulse API variants
                    for api_name in ('applyImpulse', 'applyExternalImpulse'):
                        if hasattr(client, api_name):
                            try:
                                getattr(client, api_name)(body_id, -1, (float(Ix), float(Iy), 0.0), (float(px), float(py), float(pz)))
                                applied = True
                                break
                            except Exception:
                                pass
                    if not applied:
                        # Fallback: approximate impulse by a very large one-frame external force
                        try:
                            client.applyExternalForce(body_id, -1, (float(Ix*frame_rate), float(Iy*frame_rate), 0.0), (float(px), float(py), float(pz)), p.WORLD_FRAME)
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug(f"Impulse application failed for body {body_id}: {e}")

            # Settling frames
            for _ in range(impulse_settle_frames):
                try:
                    self.simulator.run(frame_start=current, frame_end=current + 1)
                except Exception:
                    try:
                        client.stepSimulation()
                    except Exception:
                        break
                current += 1
            return current - int(start_frame)
        else:
            # Force mode (original behavior with enhancements)
            for _ in range(steps):
                for o, body_id, Fx, Fy, obj_dir, mu_eff, mass, speed in obj_info:
                    try:
                        try:
                            px, py, pz = o.position
                        except Exception:
                            px, py, pz = 0.0, 0.0, 0.0
                        client.applyExternalForce(body_id, -1, (float(Fx), float(Fy), 0.0), (float(px), float(py), float(pz)), p.WORLD_FRAME)
                    except Exception as e:
                        logging.debug(f"applyExternalForce failed for body {body_id}: {e}")
                try:
                    self.simulator.run(frame_start=current, frame_end=current + 1)
                except Exception as e:
                    try:
                        client.stepSimulation()
                    except Exception:
                        logging.warning(f"Failed to step simulation during force pulse: {e}")
                        break
                current += 1
            return current - int(start_frame)

    def configure_split(self):
        train_object_split, test_object_split = self.gso.get_test_split(fraction=0.1)
        train_background_split, test_background_split = self.hdri_source.get_test_split(fraction=0.1)
        if args.objects_split == "train":
            active_object_split = train_object_split
            active_background_split = train_background_split
        else:
            active_object_split = test_object_split
            active_background_split = test_background_split
        self.active_object_split = active_object_split
        self.active_background_split = active_background_split

        
    def setup_scene_background(self):
        hdri_id = self.rng.choice(self.active_background_split)
        background_hdri = self.hdri_source.create(asset_id=hdri_id)
        self.metadata["hdri_id"] = hdri_id
        assert isinstance(background_hdri, kb.Texture)
        logging.info("Using background %s", hdri_id)
        self.scene.metadata["background"] = hdri_id
        self.renderer._set_ambient_light_hdri(background_hdri.filename)
        # In all cases, also set the World background HDRI so it is visible if we don't use a dome
        # try:
        #     self.renderer._set_background_hdri(background_hdri.filename)
        # except Exception as e:
        #     logging.warning(f"Failed to set world background hdri: {e}")

        dome_blender = None
        if getattr(self.args, "background_mode", "dome") == "dome":
            # Dome geometry: texture the inside of a large sphere
            dome = self.kubasic.create(asset_id="dome", name="dome",
                    friction=0.5,
                    restitution=0.0,
                    static=True, background=True)
            assert isinstance(dome, kb.FileBasedObject)
            self.scene += dome
            dome_blender = dome.linked_objects[self.renderer]
            try:
                texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
                texture_node.image = bpy.data.images.load(background_hdri.filename)
            except Exception as e:
                logging.warning(f"Failed to apply HDRI to dome material: {e}")
        else:
            # World HDRI mode. Prefer Cycles for shadow-catcher/displacement, but respect CLI choice.
            try:
                self.renderer._set_background_hdri(background_hdri.filename)
            except Exception as e:
                logging.warning(f"Failed to set world background hdri: {e}")

        # --- Ground plane optional enhancement to improve near-field detail ---
        self.metadata["ground_mode"] = self.args.ground_mode
        if self.args.ground_mode != "none":
            bpy.ops.mesh.primitive_plane_add(size=self.args.ground_size, location=(0.0, 0.0, 0.0))
            ground_bl = bpy.context.active_object
            ground_bl.name = "ground_primitive"
            # Lift slightly to avoid z-fighting
            ground_bl.location.z = 0.001

            try:
                if self.args.ground_mode == "shadow":
                    # Shadow catcher requires Cycles; override if necessary.
                    eng = getattr(self.args, "engine", None)
                    use_cycles = eng.lower().startswith("c") if isinstance(eng, str) and eng else True
                    if not use_cycles:
                        logging.info("ground_mode=shadow requires Cycles; overriding engine to CYCLES.")
                    try:
                        bpy.context.scene.render.engine = 'CYCLES'
                        logging.info("Render engine set to CYCLES for shadow catcher ground.")
                    except Exception as e:
                        logging.warning(f"Failed to switch render engine to Cycles for shadow catcher: {e}")
                    if hasattr(ground_bl, 'cycles'):
                        ground_bl.cycles.is_shadow_catcher = True
                    logging.info("Ground plane set as shadow catcher.")
                
                
                # texture mode
                elif self.args.ground_mode == "textured":
                    # Prefer manually specified maps; otherwise select from manifest
                    import os, json
                    diffuse_path = self.args.ground_texture.strip() if isinstance(self.args.ground_texture, str) else ""
                    normal_path = self.args.ground_normal.strip() if isinstance(self.args.ground_normal, str) else ""
                    rough_path = self.args.ground_roughness.strip() if isinstance(self.args.ground_roughness, str) else ""
                    disp_path = self.args.ground_displacement.strip() if isinstance(self.args.ground_displacement, str) else ""
                    # Normalize to absolute paths to avoid CWD issues
                    def _abs(p: str) -> str:
                        try:
                            return os.path.abspath(p) if p else ""
                        except Exception:
                            return p or ""
                    diffuse_path = _abs(diffuse_path)
                    normal_path = _abs(normal_path)
                    rough_path = _abs(rough_path)
                    disp_path = _abs(disp_path)
                    chosen_asset_id = None
                    # If no manual inputs provided at all, try manifest-based random selection
                    if not any([diffuse_path, normal_path, rough_path, disp_path]):
                        manifest_path = getattr(self.args, "ground_texture_manifest", "/mnt/solo/kiddoray/Texture_haven.json")
                        try:
                            with open(manifest_path, 'r', encoding='utf-8') as f:
                                manifest = json.load(f)
                            download_root = manifest.get("download_dir")
                            assets = manifest.get("assets", {})
                            # Relaxed requirement: must have Diffuse; other maps optional
                            candidates = [
                                (aid, ainfo) for aid, ainfo in assets.items()
                                if isinstance(ainfo, dict) and "downloaded" in ainfo and "Diffuse" in ainfo["downloaded"]
                            ]
                            if not candidates:
                                raise RuntimeError("No assets with Diffuse map found in manifest.")
                            # Robust selection that works for numpy/random RNGs and avoids np.choice 2D error
                            def _pick_index(rng_obj, n):
                                for attr in ('integers', 'randint', 'randrange'):
                                    if hasattr(rng_obj, attr):
                                        try:
                                            if attr == 'integers':
                                                return int(getattr(rng_obj, attr)(0, n))
                                            elif attr == 'randint':
                                                return int(getattr(rng_obj, attr)(0, n))
                                            else:  # randrange
                                                return int(getattr(rng_obj, attr)(n))
                                        except Exception:
                                            continue
                                import random as _py_random
                                return _py_random.randrange(n)
                            idx = _pick_index(self.rng, len(candidates))
                            chosen_asset_id, ainfo = candidates[idx]
                            maps = ainfo["downloaded"]
                            download_root = download_root or ''
                            diffuse_path = _abs(os.path.join(download_root, maps["Diffuse"]["file"])) if "Diffuse" in maps else ""
                            print(f"Selected diffuse path: {diffuse_path}")
                            normal_path = _abs(os.path.join(download_root, maps["NormalGL"]["file"])) if "NormalGL" in maps else ""
                            rough_path = _abs(os.path.join(download_root, maps["Roughness"]["file"])) if "Roughness" in maps else ""
                            disp_path = _abs(os.path.join(download_root, maps["Displacement"]["file"])) if "Displacement" in maps else ""
                            logging.info(f"Random ground texture asset selected: {chosen_asset_id}")
                        except Exception as e:
                            logging.warning(f"Failed to select random texture from manifest '{manifest_path}': {e}. Falling back to simple color/material-only ground.")

                    # Ensure material exists
                    if len(ground_bl.data.materials) == 0:
                        mat = bpy.data.materials.new(name="GroundMaterial")
                        ground_bl.data.materials.append(mat)
                    else:
                        mat = ground_bl.data.materials[0]

                    mat.use_nodes = True
                    nt = mat.node_tree
                    for n in list(nt.nodes):
                        if n.type != 'OUTPUT_MATERIAL':
                            nt.nodes.remove(n)
                    principled = nt.nodes.new('ShaderNodeBsdfPrincipled')
                    principled.location = (-200, 0)
                    principled.inputs['Roughness'].default_value = 0.6
                    principled.inputs['Specular'].default_value = 0.25

                    # Diffuse / Base Color
                    if diffuse_path and os.path.exists(diffuse_path):
                        try:
                            img = bpy.data.images.load(diffuse_path)
                            img.colorspace_settings.name = 'sRGB'
                            tex_diff = nt.nodes.new('ShaderNodeTexImage')
                            tex_diff.image = img
                            tex_diff.location = (-600, 200)
                            nt.links.new(tex_diff.outputs['Color'], principled.inputs['Base Color'])
                            self.metadata['ground_diffuse'] = diffuse_path
                        except Exception as e:
                            logging.warning(f"Failed to load diffuse map '{diffuse_path}': {e}")
                    else:
                        principled.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)

                    # Normal map
                    normal_linked = False
                    if normal_path and os.path.exists(normal_path):
                        try:
                            nimg = bpy.data.images.load(normal_path)
                            nimg.colorspace_settings.name = 'Non-Color'
                            tex_normal = nt.nodes.new('ShaderNodeTexImage')
                            tex_normal.image = nimg
                            tex_normal.location = (-600, -200)
                            normal_map = nt.nodes.new('ShaderNodeNormalMap')
                            normal_map.location = (-400, -200)
                            nt.links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
                            nt.links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                            normal_linked = True
                            self.metadata['ground_normal'] = normal_path
                        except Exception as e:
                            logging.warning(f"Failed to load normal map '{normal_path}': {e}")

                    # Roughness map
                    if rough_path and os.path.exists(rough_path):
                        try:
                            rimg = bpy.data.images.load(rough_path)
                            rimg.colorspace_settings.name = 'Non-Color'
                            tex_rgh = nt.nodes.new('ShaderNodeTexImage')
                            tex_rgh.image = rimg
                            tex_rgh.location = (-600, 0)
                            nt.links.new(tex_rgh.outputs['Color'], principled.inputs['Roughness'])
                            self.metadata['ground_roughness'] = rough_path
                        except Exception as e:
                            logging.warning(f"Failed to load roughness map '{rough_path}': {e}")

                    # Displacement map
                    if disp_path and os.path.exists(disp_path):
                        try:
                            dimg = bpy.data.images.load(disp_path)
                            dimg.colorspace_settings.name = 'Non-Color'
                            tex_dsp = nt.nodes.new('ShaderNodeTexImage')
                            tex_dsp.image = dimg
                            tex_dsp.location = (-800, -450)
                            to_bw = nt.nodes.new('ShaderNodeRGBToBW')
                            to_bw.location = (-600, -450)
                            disp_node = nt.nodes.new('ShaderNodeDisplacement')
                            disp_node.location = (-400, -450)
                            disp_node.inputs['Scale'].default_value = 0.02
                            nt.links.new(tex_dsp.outputs['Color'], to_bw.inputs['Color'])
                            nt.links.new(to_bw.outputs['Val'], disp_node.inputs['Height'])
                            try:
                                output_node = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
                            except Exception:
                                output_node = None
                            if output_node is not None:
                                nt.links.new(disp_node.outputs['Displacement'], output_node.inputs['Displacement'])
                                try:
                                    mat.cycles.displacement_method = 'BOTH'
                                except Exception:
                                    pass
                            # Eevee fallback bump if no normal
                            engine = getattr(bpy.context.scene.render, 'engine', 'CYCLES')
                            if engine != 'CYCLES' and not normal_linked:
                                bump2 = nt.nodes.new('ShaderNodeBump')
                                bump2.location = (-200, -420)
                                bump2.inputs['Strength'].default_value = 1.0
                                nt.links.new(to_bw.outputs['Val'], bump2.inputs['Height'])
                                nt.links.new(bump2.outputs['Normal'], principled.inputs['Normal'])
                            self.metadata['ground_displacement'] = disp_path
                        except Exception as e:
                            logging.warning(f"Failed to load displacement map '{disp_path}': {e}")

                    # Final link
                    try:
                        output = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
                        nt.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                    except Exception:
                        pass
                    if chosen_asset_id:
                        self.metadata['ground_asset_id'] = chosen_asset_id
                    logging.info("Ground plane textured with randomly selected Poly Haven asset.")
                    # Load physics lookup if available and apply to ground
                    try:
                        manifest_path = getattr(self.args, "ground_texture_manifest", "/mnt/solo/kiddoray/Texture_haven.json")
                        root, ext = os.path.splitext(manifest_path)
                        lookup_path = root + "_physics.json"
                        with open(lookup_path, 'r', encoding='utf-8') as f:
                            lut = json.load(f)
                        rec = (lut.get("lookup", {}) or {}).get(chosen_asset_id, None)
                        if rec is not None:
                            mu = float(rec.get("friction", 0.8))
                            e = float(rec.get("restitution", 0.0))
                            self._ground_friction_override = mu
                            self._ground_restitution_override = e
                            self.metadata['ground_physics'] = {
                                'friction': mu,
                                'restitution': e,
                                'basis': rec.get('basis'),
                                'tags': rec.get('tags'),
                                'categories': rec.get('categories'),
                            }
                            logging.info(f"Ground physics loaded from lookup -> mu={mu:.2f}, e={e:.2f}")
                        else:
                            logging.info("No physics entry for selected asset in lookup; using defaults.")
                    except Exception as e:
                        logging.warning(f"Failed to load ground physics lookup: {e}")
            except Exception as e:
                logging.warning(f"Failed to configure ground plane material/mode: {e}")

            # Ensure there is a physics ground collider at z=0 so objects don't fall forever
            try:
                if getattr(self.args, "enable_physics_ground", True):
                    self.ensure_physics_ground()
            except Exception as e:
                logging.warning(f"Failed to ensure physics ground: {e}")
        
    def setup_camera(self):
        # Base static camera (original behavior)
        height = self.rng.uniform(0.7, 0.9)
        base_position = (-2.0, 0.0, float(height))
        self.scene.camera = kb.PerspectiveCamera(
            focal_length=self.args.focal_length,
            sensor_width=self.args.sensor_width,
            position=base_position,
            euler=(radians(90), 0, radians(270))
        )

        motion_mode = str(getattr(self.args, 'camera_motion', 'static')).lower()
        max_move = float(getattr(self.args, 'max_camera_movement', 4.0))
        frame_start = int(getattr(self.scene, 'frame_start', 0))
        frame_end = int(getattr(self.scene, 'frame_end', getattr(self.args, 'frame_end', 32)))

        if motion_mode == 'fixed_random':
            # Pick a random position in half-sphere shell and look at origin
            self.scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7., outer_radius=9., offset=0.1)
            try:
                self.scene.camera.look_at((0, 0, 0))
            except Exception:
                pass
            self.scene.metadata['camera_motion'] = {'mode': motion_mode, 'position': tuple(self.scene.camera.position)}
        elif motion_mode in ('linear', 'linear_pan'):
            is_panning = motion_mode == 'linear_pan'
            camera_start, camera_end = get_linear_camera_motion_start_end(self.rng, movement_speed=self.rng.uniform(low=0., high=max_move))
            if is_panning:
                lookat_start, lookat_end = get_linear_lookat_motion_start_end(self.rng)
            # Keyframe from one frame before start to one frame after end for consistent flow extraction
            for frame in range(frame_start - 1, frame_end + 2):
                # Normalized interpolation factor
                interp = (frame - frame_start + 1) / (frame_end - frame_start + 3)
                pos = interp * camera_start + (1 - interp) * camera_end
                self.scene.camera.position = tuple(pos.tolist())
                if is_panning:
                    la = interp * lookat_start + (1 - interp) * lookat_end
                    try:
                        self.scene.camera.look_at(tuple(la.tolist()))
                    except Exception:
                        pass
                else:
                    try:
                        self.scene.camera.look_at((0, 0, 0))
                    except Exception:
                        pass
                # Insert keyframes (silently ignore if backend doesn't support)
                try:
                    self.scene.camera.keyframe_insert('position', frame)
                    self.scene.camera.keyframe_insert('quaternion', frame)
                except Exception:
                    pass
            meta = {
                'mode': motion_mode,
                'camera_start': tuple(camera_start.tolist()),
                'camera_end': tuple(camera_end.tolist()),
            }
            if is_panning:
                meta['lookat_start'] = tuple(lookat_start.tolist())
                meta['lookat_end'] = tuple(lookat_end.tolist())
            self.scene.metadata['camera_motion'] = meta
        elif motion_mode == 'orbit':
            cycles = float(getattr(self.args, 'camera_orbit_cycles', 0.5))
            rmin = float(getattr(self.args, 'camera_orbit_radius_min', 3.0))
            rmax = float(getattr(self.args, 'camera_orbit_radius_max', 5.0))
            hmin = float(getattr(self.args, 'camera_orbit_height_min', 0.6))
            hmax = float(getattr(self.args, 'camera_orbit_height_max', 0.8))
            radius = float(self.rng.uniform(rmin, rmax))
            height = float(self.rng.uniform(hmin, hmax))
            start_angle = float(self.rng.uniform(0.0, 2*np.pi))
            for frame in range(frame_start - 1, frame_end + 2):
                interp = (frame - frame_start + 1) / (frame_end - frame_start + 3)
                angle = start_angle + interp * cycles * 2 * np.pi
                px = radius * np.cos(angle)
                py = radius * np.sin(angle)
                pz = height
                self.scene.camera.position = (float(px), float(py), float(pz))
                try:
                    self.scene.camera.look_at((0, 0, 0))
                except Exception:
                    pass
                try:
                    self.scene.camera.keyframe_insert('position', frame)
                    self.scene.camera.keyframe_insert('quaternion', frame)
                except Exception:
                    pass
            self.scene.metadata['camera_motion'] = {
                'mode': 'orbit',
                'radius': radius,
                'height': height,
                'cycles': cycles,
                'start_angle': start_angle,
            }
        else:
            # static base camera
            self.scene.metadata['camera_motion'] = {'mode': 'static', 'position': tuple(base_position)}
    
    def _choose_camera_views(self, num_views: int, mode: str = "random"):
        """Return a list of camera poses (position, look_at) around (0,0,0)."""
        # Use current camera height and a nominal radius based on current position
        cx, cy, cz = self.scene.camera.position
        radius = float(np.sqrt(cx*cx + cy*cy))
        if radius <= 1e-3:
            radius = 2.5
        height = float(cz)

        views = []
        if num_views <= 0:
            return views

        if mode == "circle":
            offset = float(self.rng.uniform(0.0, 2*np.pi))
            # evenly spaced angles
            angles = np.linspace(0.0, 2*np.pi, num_views, endpoint=False) + offset
        elif mode == "vertical":
            # Keep horizontal azimuth fixed; vary height from high (top-down) to low (bottom-up)
            phi0 = float(np.arctan2(cy, cx)) if radius > 1e-3 else 0.0
            r = radius if radius > 1e-3 else 2.5

            # Safe height range above ground
            z_top = float(max(height, 0.8))
            z_bottom = 0.2
            z_list = list(np.linspace(z_top, z_bottom, max(1, num_views)))

            # Adjust look_at z so the last view tends to be looking upward
            look_z_start = 0.0
            look_z_end = z_bottom + 0.6
            la_list = list(np.linspace(look_z_start, look_z_end, max(1, num_views)))

            for z, la_z in zip(z_list, la_list):
                px = r * float(np.cos(phi0))
                py = r * float(np.sin(phi0))
                views.append({
                    "position": (px, py, float(z)),
                    "look_at": (0.0, 0.0, float(la_z)),
                })
            return views
        else:  
            # random
            angles = [float(self.rng.uniform(0.0, 2*np.pi)) for _ in range(num_views)]

        for th in angles:
            px = radius * float(np.cos(th))
            py = radius * float(np.sin(th))
            views.append({
                "position": (px, py, height),
                "look_at": (0.0, 0.0, 0.0),
            })
        return views
     
    def place_dynamic_objects(self):
        objs = []
        for i in range(self.num_dynamic_objects):
            attempts = 0
            while attempts < 10:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                # Physics overrides
                self._apply_object_physics(obj, obj_id)
                samplers = [
                    kb.randomness.position_sampler(self.cfg.dynamic_spawn_region), 
                    kb.randomness.rotation_sampler(axis=None), 
                ]
                logging.info("Sampling dynamic object pose...")
                kb.resample_while(obj,samplers,self.simulator.check_overlap,rng=self.rng)

                min_corner, max_corner = get_world_object_bounds(obj) 
                drop_bounds = np.array([ # top left and bottom right of plane near camera
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) * (projected_points[0][1] - projected_points[1][1]))
                self.scene.remove(obj)
                if area / (self.scene.resolution[0] * self.scene.resolution[1]) > 0.025:
                    break
                attempts += 1
                if attempts < 10:
                    del obj

            objs.append(obj)
            

        return objs
    
    def place_throw_objects(self, v_scale=None):
        objs = []

        for i in range(self.num_dynamic_objects):
            attempts = 0
            while attempts < 10:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                # Physics overrides
                self._apply_object_physics(obj, obj_id)
                # Split sampling into two regions to avoid rejection sampling for x in [-1.5, 1.5]
                min_bound, max_bound = THROW_SPAWN_REGION
                
                if self.rng.uniform() < 0.5:
                    # Left region: x from min_x to -1.0
                    region = [
                        (min_bound[0], min_bound[1], min_bound[2]),
                        (-1.0, max_bound[1], max_bound[2])
                    ]
                else:
                    # Right region: x from 1.0 to max_x
                    region = [
                        (1.0, min_bound[1], min_bound[2]),
                        (max_bound[0], max_bound[1], max_bound[2])
                    ]

                samplers = [
                    kb.randomness.position_sampler(region), 
                    kb.randomness.rotation_sampler(axis=None), 
                ]
                logging.info("Sampling dynamic object pose...")

                # def validation_fn(obj):
                #     # Reject if overlapping with other objects
                #     if self.simulator.check_overlap(obj):
                #         return True
                #     return False

                kb.resample_while(obj, samplers, self.simulator.check_overlap, rng=self.rng)

                # Sample a target point near the center to throw towards
                # target_region = [(-0.2, -0.2, 0), (0.2, 0.2, 0)]
                # target_pos = self.rng.uniform(*target_region)
                if v_scale == None:
                    v_scale = self.rng.uniform(1.0, 2.0)
                obj.velocity = v_scale * (np.array([0, 0, 0]) - np.array([obj.position[0], obj.position[1], 0]))
                obj.metadata["is_dynamic"] = True

                # Enable CCD for high-speed thrown objects to prevent tunneling through static obstacles
                try:
                    body_id = obj.linked_objects[self.simulator]
                    self.simulator._physics_client.changeDynamics(int(body_id), -1, ccdSweptSphereRadius=0.05)
                except Exception:
                    pass

                min_corner, max_corner = get_world_object_bounds(obj) 
                drop_bounds = np.array([ # top left and bottom right of plane near camera
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) * (projected_points[0][1] - projected_points[1][1]))
                self.scene.remove(obj)
                if area / (self.scene.resolution[0] * self.scene.resolution[1]) > 0.025:
                    break
                
                attempts += 1
                if attempts < 10:
                    del obj

            objs.append(obj)

        return objs
    
    def place_radius_throw_objects(self, min_radius: float = 1.0, max_radius: float = 2.0, 
                             height_range: tuple = (0.3, 0.6), speed_range: tuple = (2.0, 4.0)):
        """
        Place objects in mid-air within an annular region (ring) around the origin.
        All objects have velocities directed toward the center (origin).
        
        Strategy:
        - Sample positions in an annular region: min_radius <= r <= max_radius
        - Z position is in mid-air (height_range)
        - Assign velocities pointing toward (0, 0, 0) with random speed
        
        Parameters:
            min_radius: inner radius of the spawn ring (meters)
            max_radius: outer radius of the spawn ring (meters)
            height_range: (min_z, max_z) for object spawn height
            speed_range: (min_speed, max_speed) for velocity magnitude
        """
        objs = []

        for i in range(self.num_dynamic_objects):
            attempts = 0
            chosen = None
            while attempts < 10:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                # Physics overrides
                self._apply_object_physics(obj, obj_id)
                
                # Sample position in annular region (ring around origin)
                # Use uniform sampling in the annulus: r = sqrt(uniform(r_min^2, r_max^2))
                theta = float(self.rng.uniform(0.0, 2 * np.pi))
                r_squared = self.rng.uniform(min_radius**2, max_radius**2)
                r = float(np.sqrt(r_squared))
                
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                pz = float(self.rng.uniform(height_range[0], height_range[1]))
                
                # Set position in mid-air
                obj.position = (float(px), float(py), pz)
                obj.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                # Check overlap with existing objects
                if self.simulator.check_overlap(obj):
                    self.scene.remove(obj)
                    del obj
                    attempts += 1
                    continue
                
                # Visibility check: project object's footprint to image
                min_corner, max_corner = get_world_object_bounds(obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) * 
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])
                
                # Accept if visible enough
                if coverage > 0.025:
                    chosen = obj
                    
                    # Calculate velocity toward origin (center)
                    speed = float(self.rng.uniform(speed_range[0], speed_range[1]))
                    # Direction: from current position toward (0, 0, 0) in XY plane
                    dist = np.sqrt(px**2 + py**2)
                    if dist > 1e-6:
                        vx = -speed * px / dist
                        vy = -speed * py / dist
                    else:
                        vx, vy = 0.0, 0.0
                    obj.velocity = (float(vx), float(vy), 0.0)
                    
                    print(f"Throw object {obj_id}: pos=({px:.2f}, {py:.2f}, {pz:.2f}), vel=({vx:.2f}, {vy:.2f}), speed={speed:.2f}")
                    obj.metadata["is_dynamic"] = True
                    
                    # Enable CCD for high-speed thrown objects to prevent tunneling through static obstacles
                    try:
                        body_id = obj.linked_objects[self.simulator]
                        self.simulator._physics_client.changeDynamics(int(body_id), -1, ccdSweptSphereRadius=0.05)
                    except Exception:
                        pass
                    
                    break
                
                # Reject and try again
                self.scene.remove(obj)
                del obj
                attempts += 1
            
            # Fallback: accept object even if visibility is poor
            if chosen is None:
                obj_id = self.rng.choice(self.active_object_split)
                chosen = self.gso.create(asset_id=obj_id)
                self.scene.add(chosen)
                self._apply_object_physics(chosen, obj_id)
                
                # Sample in annular region
                theta = float(self.rng.uniform(0.0, 2 * np.pi))
                r_squared = self.rng.uniform(min_radius**2, max_radius**2)
                r = float(np.sqrt(r_squared))
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                pz = float(self.rng.uniform(height_range[0], height_range[1]))
                
                chosen.position = (float(px), float(py), pz)
                chosen.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                # Calculate velocity toward origin
                speed = float(self.rng.uniform(speed_range[0], speed_range[1]))
                dist = np.sqrt(px**2 + py**2)
                if dist > 1e-6:
                    vx = -speed * px / dist
                    vy = -speed * py / dist
                else:
                    vx, vy = 0.0, 0.0
                chosen.velocity = (float(vx), float(vy), 0.0)
                
                print(f"Throw object (fallback) {obj_id}: pos=({px:.2f}, {py:.2f}, {pz:.2f})")
                chosen.metadata["is_dynamic"] = True
                
                # Enable CCD
                try:
                    body_id = chosen.linked_objects[self.simulator]
                    self.simulator._physics_client.changeDynamics(int(body_id), -1, ccdSweptSphereRadius=0.05)
                except Exception:
                    pass

            objs.append(chosen)

        return objs
    
    def place_collision_objects(self, min_radius: float = 1.0, max_radius: float = 2.0, speed_range: tuple = (3.0, 5.0)):
        """
        Place objects on the ground within an annular region (ring) around the origin.
        All objects have velocities directed toward the center (origin).
        
        Strategy:
        - Sample positions in an annular region: min_radius <= r <= max_radius
        - Adjust Z position so objects rest on ground plane
        - Assign velocities pointing toward (0, 0, 0) with random speed
        
        Parameters:
            min_radius: inner radius of the spawn ring (meters)
            max_radius: outer radius of the spawn ring (meters)
            speed_range: (min_speed, max_speed) for velocity magnitude
        """
        objs = []
        
        for _ in range(self.num_dynamic_objects):
            attempts = 0
            chosen = None
            while attempts < 10:
                obj_id = self.rng.choice(self.active_object_split)
                ground_obj = self.gso.create(asset_id=obj_id)
                self.scene.add(ground_obj)
                self._apply_object_physics(ground_obj, obj_id)
                
                # Sample position in annular region (ring around origin)
                # Use uniform sampling in the annulus: r = sqrt(uniform(r_min^2, r_max^2))
                theta = float(self.rng.uniform(0.0, 2 * np.pi))
                r_squared = self.rng.uniform(min_radius**2, max_radius**2)
                r = float(np.sqrt(r_squared))
                
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                
                # Set initial position (z will be adjusted after)
                ground_obj.position = (float(px), float(py), 0.5)
                ground_obj.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                # Check overlap with existing objects
                if self.simulator.check_overlap(ground_obj):
                    self.scene.remove(ground_obj)
                    del ground_obj
                    attempts += 1
                    continue
                
                # Adjust Z so that the object sits on ground plane (bottom touches z=0)
                z_offset = get_object_height_from_center(ground_obj) + 1e-3
                ground_obj.position = (float(px), float(py), z_offset)
                
                # Visibility check: project object's footprint to image
                min_corner, max_corner = get_world_object_bounds(ground_obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) *
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])
                
                # Accept if visible enough
                if coverage > 0.025:
                    chosen = ground_obj
                    
                    # Calculate velocity toward origin (center)
                    speed = float(self.rng.uniform(speed_range[0], speed_range[1]))
                    # Direction: from current position toward (0, 0)
                    dist = np.sqrt(px**2 + py**2)
                    if dist > 1e-6:
                        vx = -speed * px / dist
                        vy = -speed * py / dist
                    else:
                        vx, vy = 0.0, 0.0
                    ground_obj.velocity = (float(vx), float(vy), 0.0)
                    
                    print(f"Ground object {obj_id}: pos=({px:.2f}, {py:.2f}), vel=({vx:.2f}, {vy:.2f}), speed={speed:.2f}")
                    ground_obj.metadata["is_dynamic"] = True
                    break
                
                # Reject and try again
                self.scene.remove(ground_obj)
                del ground_obj
                attempts += 1
            
            # Fallback: accept object even if visibility is poor
            if chosen is None:
                obj_id = self.rng.choice(self.active_object_split)
                chosen = self.gso.create(asset_id=obj_id)
                self.scene.add(chosen)
                self._apply_object_physics(chosen, obj_id)
                
                # Sample in annular region
                theta = float(self.rng.uniform(0.0, 2 * np.pi))
                r_squared = self.rng.uniform(min_radius**2, max_radius**2)
                r = float(np.sqrt(r_squared))
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                
                chosen.position = (float(px), float(py), 0.5)
                chosen.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                z_offset = get_object_height_from_center(chosen) + 1e-3
                chosen.position = (float(px), float(py), z_offset)
                
                # Velocity toward origin
                speed = float(self.rng.uniform(speed_range[0], speed_range[1]))
                dist = np.sqrt(px**2 + py**2)
                if dist > 1e-6:
                    vx = -speed * px / dist
                    vy = -speed * py / dist
                else:
                    vx, vy = 0.0, 0.0
                chosen.velocity = (float(vx), float(vy), 0.0)
                
                print(f"Ground object (fallback) {obj_id}: pos=({px:.2f}, {py:.2f}), vel=({vx:.2f}, {vy:.2f})")
                chosen.metadata["is_dynamic"] = True

            objs.append(chosen)
        
        logging.info(f"Placed {len(objs)} ground objects in annular region (r={min_radius}-{max_radius}m) with velocities toward center")
        return objs

    def place_offscreen_influx(self, count: int, speed: float, margin: float = 0.25):
        """
        Spawn `count` dynamic objects just outside the visible spawn region so that
        they enter the scene from offscreen with approximately constant horizontal speed.

        - Objects are placed beyond the +/-X boundary of dynamic_spawn_region by `margin`.
        - Y is sampled within the region; Z is set so the object sits on ground.
        - Velocity is directed roughly toward the origin with magnitude `speed`.
        - Returns the list of created objects. Velocities are pushed to the physics backend.
        """
        objs = []
        (xmin, ymin, _), (xmax, ymax, _) = self.cfg.dynamic_spawn_region

        for idx in range(int(max(0, count))):
            attempts = 0
            chosen = None
            # side = int(self.rng.choice([-1, 1]))  # -1 from left (neg x), +1 from right
            side = self.rng.uniform(-1, 1)  # scale speed <0: left, >0: right
            while attempts < 20:
                obj = None
                try:
                    obj_id = self.rng.choice(self.active_object_split)
                    obj = self.gso.create(asset_id=obj_id)
                    self.scene.add(obj)
                    self._apply_object_physics(obj, obj_id)

                    # # Sample y within region; position x just beyond boundary with a small band
                    # y = float(self.rng.uniform(ymin, ymax))
                    # band = 0.02
                    # x = (xmin - (margin + band)) if side < 0 else (xmax + (margin + band))
                    x = float(self.rng.uniform(xmin, xmax))
                    y = float(self.rng.uniform(ymin, ymax))


                    # Place on ground (bottom touches z=0)
                    z_offset = get_object_height_from_center(obj) + 1e-3
                    obj.position = (x, y, z_offset)

                    if self.args.stage2_force == "none":
                        dir = 'center'
                        if self.args.stage2 == "mixed":
                            dir = str(self.rng.choice(['center', 'random'])).lower()
                        else:
                            dir = self.args.stage2

                        if dir == 'center':
                            # Aim toward the scene origin in XY with fixed speed
                            to_center = np.array([-x, -y, 0.0], dtype=float)
                            n = float(np.linalg.norm(to_center[:2]))
                            if n < 1e-6:
                                vx, vy = (-side, 0.0)
                            else:
                                vx = float(to_center[0] / n)
                                vy = float(to_center[1] / n)
                            
                            cur_speed = self.rng.uniform((0.8 * speed), (1.2 * speed))
                            obj.velocity = (float(cur_speed) * vx, float(cur_speed) * vy, 0.0)
                        else:
                            obj.velocity = self.rng.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1])

                    # Mark as dynamic and accept
                    try:
                        obj.metadata["is_dynamic"] = True
                    except Exception:
                        pass
                    chosen = obj
                    break
                except Exception as e:
                    logging.warning(f"[combined] offscreen spawn attempt {attempts+1} for idx {idx} failed: {e}")
                    # Cleanup and retry
                    try:
                        if obj is not None:
                            self.scene.remove(obj)
                    except Exception:
                        pass
                    attempts += 1
                    continue

            if chosen is not None:
                objs.append(chosen)

        # Defer pushing velocities to after registration tick in combined stage
        logging.info(f"Spawned {len(objs)} offscreen entrants at speed~{speed} m/s")
        return objs
       
    def add_slope(self):
        """Create a static inclined plane (slope) using Blender native mesh.
        
        Since Kubric only supports uniform scaling, we create the slope directly
        in Blender with proper non-uniform dimensions, then wrap it for physics.

        Parameters (read from self.args with defaults):
        - slope_angle_deg: tilt angle in degrees (default 25)
        - slope_length: length along the slope direction (default 3.0 m)
        - slope_width: width perpendicular to slope (default 2.0 m)
        - slope_thickness: slab thickness (default 0.1 m)
        - slope_friction: lateral friction coefficient (default 0.3)
        - slope_restitution: bounciness (default 0.1)
        
        Returns the created slope object or None on failure.
        """
        try:
            angle_deg = float(getattr(self.args, 'slope_angle_deg', 25.0))
        except Exception:
            angle_deg = 25.0
        angle_rad = float(np.deg2rad(angle_deg))
        
        L = float(getattr(self.args, 'slope_length', 3.0))
        W = float(getattr(self.args, 'slope_width', 2.0))
        T = float(getattr(self.args, 'slope_thickness', 0.1))
        mu = float(getattr(self.args, 'slope_friction', 0.3))
        e = float(getattr(self.args, 'slope_restitution', 0.1))
        
        # =====================================================================
        # Create slope using Blender native mesh (supports non-uniform scale)
        # =====================================================================
        try:
            # Create a cube in Blender with size 1, then scale non-uniformly
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
            slope_bl = bpy.context.active_object
            slope_bl.name = "slope_panel"
            
            # Apply non-uniform scale: X=length, Y=width, Z=thickness
            slope_bl.scale = (L, W, T)
            # Apply the scale to mesh data
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            # Position: low end at x~0, z=0
            # Center of tilted cube: 
            #   center_x = (L/2) * cos(angle) (so low end is near origin)
            #   center_z = (L/2) * sin(angle) + (T/2) * cos(angle)
            center_x = (L / 2.0) * np.cos(angle_rad)
            center_y = 0.0
            center_z = (L / 2.0) * np.sin(angle_rad) + (T / 2.0) * np.cos(angle_rad)
            
            slope_bl.location = (float(center_x), float(center_y), float(center_z))
            
            # Rotate around Y-axis to create the incline
            # Positive angle tilts the +X end up
            slope_bl.rotation_euler = Euler((0.0, -angle_rad, 0.0), 'XYZ')
            
            # =====================================================================
            # Apply random material for variety
            # =====================================================================
            material_types = ['metal', 'wood', 'rubber', 'stone', 'plastic', 'concrete']
            material_type = self.rng.choice(material_types)
            
            mat = bpy.data.materials.new(name=f"SlopeMaterial_{material_type}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            bsdf = nodes.get("Principled BSDF")
            
            # Default friction (will be overwritten based on material type)
            mu = float(self.rng.uniform(0.3, 0.5))
            
            if bsdf:
                if material_type == 'metal':
                    # Metallic surface - steel/aluminum look
                    metal_colors = [
                        (0.7, 0.7, 0.75, 1.0),   # Steel
                        (0.85, 0.65, 0.45, 1.0), # Copper
                        (0.9, 0.85, 0.5, 1.0),   # Gold/Brass
                        (0.6, 0.6, 0.65, 1.0),   # Dark steel
                    ]
                    color = metal_colors[self.rng.randint(0, len(metal_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 1.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.1, 0.4))
                    # Adjust friction for metal
                    mu = float(self.rng.uniform(0.15, 0.3))
                    
                elif material_type == 'wood':
                    # Wooden surface - brown tones
                    wood_colors = [
                        (0.4, 0.25, 0.12, 1.0),  # Dark wood
                        (0.6, 0.4, 0.2, 1.0),    # Medium wood
                        (0.75, 0.55, 0.35, 1.0), # Light wood
                        (0.35, 0.2, 0.1, 1.0),   # Walnut
                    ]
                    color = wood_colors[self.rng.randint(0, len(wood_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 0.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.4, 0.7))
                    mu = float(self.rng.uniform(0.3, 0.5))
                    
                elif material_type == 'rubber':
                    # Rubber surface - dark, matte
                    rubber_colors = [
                        (0.05, 0.05, 0.05, 1.0), # Black rubber
                        (0.15, 0.15, 0.15, 1.0), # Dark gray
                        (0.1, 0.05, 0.02, 1.0),  # Dark brown
                    ]
                    color = rubber_colors[self.rng.randint(0, len(rubber_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 0.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.7, 0.95))
                    mu = float(self.rng.uniform(0.6, 0.9))  # High friction
                    
                elif material_type == 'stone':
                    # Stone/marble surface
                    stone_colors = [
                        (0.5, 0.5, 0.5, 1.0),    # Gray stone
                        (0.8, 0.75, 0.7, 1.0),   # Light marble
                        (0.35, 0.35, 0.35, 1.0), # Dark stone
                        (0.6, 0.55, 0.5, 1.0),   # Sandstone
                    ]
                    color = stone_colors[self.rng.randint(0, len(stone_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 0.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.3, 0.6))
                    mu = float(self.rng.uniform(0.4, 0.6))
                    
                elif material_type == 'plastic':
                    # Colorful plastic
                    plastic_colors = [
                        (0.8, 0.2, 0.2, 1.0),    # Red
                        (0.2, 0.6, 0.8, 1.0),    # Blue
                        (0.2, 0.7, 0.3, 1.0),    # Green
                        (0.9, 0.7, 0.1, 1.0),    # Yellow
                        (0.9, 0.9, 0.9, 1.0),    # White
                    ]
                    color = plastic_colors[self.rng.randint(0, len(plastic_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 0.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.2, 0.5))
                    # Try different specular input names for Blender version compatibility
                    # Blender 3.x uses "Specular", Blender 4.x uses "Specular IOR Level"
                    try:
                        bsdf.inputs["Specular"].default_value = 0.5
                    except KeyError:
                        try:
                            bsdf.inputs["Specular IOR Level"].default_value = 0.5
                        except KeyError:
                            pass  # Skip if neither exists
                    mu = float(self.rng.uniform(0.25, 0.45))
                    
                else:  # concrete
                    # Concrete surface - gray tones
                    concrete_colors = [
                        (0.45, 0.45, 0.45, 1.0), # Medium gray
                        (0.55, 0.55, 0.52, 1.0), # Light gray
                        (0.35, 0.35, 0.35, 1.0), # Dark gray
                    ]
                    color = concrete_colors[self.rng.randint(0, len(concrete_colors))]
                    bsdf.inputs["Base Color"].default_value = color
                    bsdf.inputs["Metallic"].default_value = 0.0
                    bsdf.inputs["Roughness"].default_value = float(self.rng.uniform(0.6, 0.9))
                    mu = float(self.rng.uniform(0.5, 0.7))
            
            slope_bl.data.materials.append(mat)
            
            # Store material type for logging
            self._slope_material_type = material_type
            
            logging.info(f"Created Blender slope mesh: size=({L}x{W}x{T}), angle={angle_deg}°, material={material_type}")
            
        except Exception as ex:
            logging.error(f"Failed to create slope in Blender: {ex}")
            return None
        
        # =====================================================================
        # Create a Kubric wrapper for physics simulation
        # =====================================================================
        try:
            # Create a simple static cube from kubasic with uniform scale=1
            # We'll override its Blender mesh with our custom slope
            slope_kb = self.kubasic.create(
                asset_id="cube",
                name="slope_physics",
                scale=1.0,  # Uniform scale only!
                static=True,
            )
            slope_kb.friction = mu
            slope_kb.restitution = e
            
            # Add to scene
            self.scene.add(slope_kb)
            
            # Get the kubric object's Blender mesh and hide it
            kb_bl = slope_kb.linked_objects[self.renderer]
            kb_bl.hide_render = True
            kb_bl.hide_viewport = True
            
            # The visual slope is separate; we need to link physics properly
            # For physics, we use the kubric object but position it to match
            slope_kb.position = (float(center_x), float(center_y), float(center_z))
            
            # Set rotation
            q = pyquat.Quaternion(axis=(0.0, 1.0, 0.0), angle=-angle_rad)
            slope_kb.quaternion = (q.w, q.x, q.y, q.z)
            
        except Exception as ex:
            logging.warning(f"Failed to create kubric physics wrapper: {ex}")
            slope_kb = None
        
        # =====================================================================
        # Create physics collider using PyBullet directly
        # =====================================================================
        try:
            import pybullet as p
            client = self.simulator._physics_client
            
            # Create a box collision shape with proper dimensions
            half_extents = [L/2.0, W/2.0, T/2.0]
            collision_shape = client.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            
            # Create the rigid body at the correct position and orientation
            # Convert quaternion for PyBullet (x, y, z, w format)
            q = pyquat.Quaternion(axis=(0.0, 1.0, 0.0), angle=-angle_rad)
            quat_pybullet = [q.x, q.y, q.z, q.w]
            
            body_id = client.createMultiBody(
                baseMass=0,  # static
                baseCollisionShapeIndex=collision_shape,
                basePosition=[center_x, center_y, center_z],
                baseOrientation=quat_pybullet
            )
            
            # Set friction and restitution
            client.changeDynamics(
                body_id, -1,
                lateralFriction=mu,
                restitution=e
            )
            
            self._slope_body_id = body_id
            logging.info(f"Created PyBullet slope collider: body_id={body_id}, friction={mu}")
            
        except Exception as ex:
            logging.error(f"Failed to create PyBullet slope collider: {ex}")
            self._slope_body_id = None
        
        # =====================================================================
        # Store parameters for object placement
        # =====================================================================
        # High point of the slope surface (top face)
        high_x = center_x + (L / 2.0) * np.cos(angle_rad)
        high_z = L * np.sin(angle_rad) + T * np.cos(angle_rad)
        low_x = center_x - (L / 2.0) * np.cos(angle_rad)
        low_z = T * np.cos(angle_rad)
        
        self._slope_obj = slope_bl  # Store Blender object for reference
        self._slope_kb = slope_kb   # Kubric wrapper (may be None)
        self._slope_params = {
            'angle_rad': angle_rad,
            'angle_deg': angle_deg,
            'L': L,
            'W': W,
            'T': T,
            'center': (center_x, center_y, center_z),
            'high_point': (high_x, 0.0, high_z),
            'low_point': (low_x, 0.0, low_z),
            'mu': mu,
        }
        
        logging.info(f"Added slope: angle={angle_deg}°, size=({L}x{W}x{T}), "
                     f"center=({center_x:.2f}, {center_y:.2f}, {center_z:.2f}), "
                     f"high_z={high_z:.2f}m, friction={mu}")
        
        return slope_bl

    def place_objects_on_slope(self, slope, count: int | None = None, margin: float = 0.15):
        """Place dynamic objects at the top of the slope so they slide down naturally.

        Objects are positioned near the high end of the slope, slightly above the surface.
        Returns the list of placed objects.
        """
        if slope is None:
            logging.warning("place_objects_on_slope: slope is None")
            return []
        
        params = getattr(self, '_slope_params', None)
        if not params:
            logging.warning("place_objects_on_slope: no slope params found")
            return []
        
        angle_rad = float(params['angle_rad'])
        L = float(params['L'])
        W = float(params['W'])
        T = float(params['T'])
        cx, cy, cz = params['center']
        
        n = int(count if count is not None else max(1, getattr(self, 'num_dynamic_objects', 3)))
        
        objs = []
        
        # Spawn region: near the high end of the slope (upper portion)
        # In local slope coordinates, high end is at +L/2
        # We sample in the upper third of the slope
        spawn_fraction_min = 0.3  # start from 30% up
        spawn_fraction_max = 0.9  # up to 90% of the slope length
        
        y_min = -W / 2.0 + margin
        y_max = W / 2.0 - margin
        
        for i in range(n):
            attempts = 0
            placed = None
            
            while attempts < 15:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                self._apply_object_physics(obj, obj_id)
                
                # Sample position along slope (fraction from low to high end)
                frac = float(self.rng.uniform(spawn_fraction_min, spawn_fraction_max))
                
                # Position along slope direction (X in world coords after rotation)
                # Local slope coordinate: distance from center along slope = (frac - 0.5) * L
                local_along = (frac - 0.5) * L
                
                # World position (slope tilted around Y, so X and Z are affected)
                # Object sits on top of the slope surface
                world_x = cx + local_along * np.cos(angle_rad)
                world_z = cz + local_along * np.sin(angle_rad) + (T / 2.0) * np.cos(angle_rad)
                
                # Y position (width direction, unaffected by tilt)
                world_y = cy + float(self.rng.uniform(y_min, y_max))
                
                # Add object height offset so it sits on the surface
                obj_height = get_object_height_from_center(obj)
                world_z += obj_height + 0.01  # small gap to avoid penetration
                
                obj.position = (float(world_x), float(world_y), float(world_z))
                
                # Random rotation
                obj.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                # Check overlap
                if self.simulator.check_overlap(obj):
                    self.scene.remove(obj)
                    del obj
                    attempts += 1
                    continue
                
                # Visibility check
                try:
                    min_corner, max_corner = get_world_object_bounds(obj)
                    drop_bounds = np.array([
                        [min_corner[0], max_corner[1], max_corner[2]],
                        [min_corner[0], min_corner[1], 0],
                    ])
                    projected_points = project_points_to_image(drop_bounds, self.scene)
                    area = abs((projected_points[0][0] - projected_points[1][0]) *
                               (projected_points[0][1] - projected_points[1][1]))
                    coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])
                except Exception:
                    coverage = 0.05  # assume visible if check fails
                
                if coverage > 0.01:
                    try:
                        obj.metadata["is_dynamic"] = True
                    except Exception:
                        pass
                    placed = obj
                    break
                
                self.scene.remove(obj)
                del obj
                attempts += 1
            
            # Fallback: place even if visibility check failed
            if placed is None:
                obj_id = self.rng.choice(self.active_object_split)
                placed = self.gso.create(asset_id=obj_id)
                self.scene.add(placed)
                self._apply_object_physics(placed, obj_id)
                
                # Position at mid-high point
                frac = 0.7
                local_along = (frac - 0.5) * L
                world_x = cx + local_along * np.cos(angle_rad)
                world_z = cz + local_along * np.sin(angle_rad) + (T / 2.0) * np.cos(angle_rad)
                world_y = cy
                obj_height = get_object_height_from_center(placed)
                world_z += obj_height + 0.01
                
                placed.position = (float(world_x), float(world_y), float(world_z))
                placed.quaternion = kb.randomness.random_rotation(rng=self.rng)
                
                try:
                    placed.metadata["is_dynamic"] = True
                except Exception:
                    pass
            
            # Enable CCD for smooth collision detection on slope
            try:
                body_id = placed.linked_objects[self.simulator]
                self.simulator._physics_client.changeDynamics(
                    int(body_id), -1, 
                    ccdSweptSphereRadius=0.02
                )
            except Exception:
                pass
            
            objs.append(placed)
            logging.info(f"Placed object {i+1}/{n} on slope at ({placed.position[0]:.2f}, {placed.position[1]:.2f}, {placed.position[2]:.2f})")
        
        logging.info(f"Placed {len(objs)} dynamic objects on slope")
        return objs
      
    def place_static_objects_under_dynamic_obj(self, dynamic_obj):
        tower = []
        tower_size = self.rng.randint(self.cfg.min_tower_size, self.cfg.max_tower_size)
        if tower_size == 0:
            return tower
        obj_ids = self.rng.choice(self.active_object_split, tower_size, replace=False)
        z,offset = 0.0, 0.001
        logging.info(f"Making a tower of {tower_size} objects...")
        for obj_id in obj_ids:
            obj = self.gso.create(asset_id=obj_id)
            # Randomize object scale uniformly in [1, 1.5] before computing heights/placement
            try:
                s = float(self.rng.uniform(1, 1.5))
                try:
                    obj.scale = s  # many kubric FileBasedObject accept scalar uniform scale
                except Exception:
                    obj.scale = (s, s, s)  # fallback to vector scale if needed
            except Exception:
                pass
            self._apply_object_physics(obj, obj_id)
            obj.position = (dynamic_obj.position[0], dynamic_obj.position[1], z + offset +get_object_height_from_center(obj))
            _, max_corner = get_world_object_bounds(obj)
            z = max_corner[2]
            tower.append(obj)
            self.scene.add(obj)
            self.static_objects.append(obj)
        return tower

    # ------------------------------------------------------------------
    def place_static_objects_offset_stack(
        self,
        dynamic_obj,
        num_stacks: int | None = None,
        max_stack_height: int = 2,
        offset_radius: float = 0.6,
        stack_size_mode: str = "random",
        force_min_stack: int = 1,
        force_max_stack: int | None = None,
        radial_mode: str = "uniform",
        avoid_center: bool = True,
        min_separation: float = 0.15,
        scale_range: tuple[float, float] = (1.2, 2.0),
    ) -> list:
        """Place multiple stacks of static objects near (but not necessarily under) a dynamic object.

        Goals:
        1. Allow horizontal offset of static objects (radial sampling around dynamic object).
        2. Each base position can spawn a vertical stack of 1-3 objects (configurable).

        Parameters:
            dynamic_obj: reference object to position stacks around.
            num_stacks: number of distinct stack bases. If None -> random 1-3.
            max_stack_height: hard cap for stack height (default 3).
            offset_radius: maximum radial distance from dynamic_obj (meters).
            stack_size_mode: 'random' or 'fixed'. If 'fixed' uses force_max_stack/force_min_stack.
            force_min_stack: minimum stack height when mode='fixed' or lower bound for random.
            force_max_stack: maximum stack height when mode='fixed'; defaults to max_stack_height if None.
            radial_mode: 'uniform' (sample radius in [0,R]) or 'edge' (radius ~ R) or 'annulus' (avoid inner region).
            avoid_center: if True, prefer radius > 0.25*offset_radius to not block the fall directly.
            min_separation: minimum distance between stack base centers.
            scale_range: tuple (min,max) for uniform random scaling applied per object before height calc.

        Returns:
            flat list of all placed static objects. Each object metadata augmented with:
              - is_dynamic = False
              - stack_index (int base id)
              - stack_level (0 = bottom)

        Notes:
            - This function does NOT remove or alter the legacy tower function.
            - Uses simple rejection sampling for separated base positions.
            - Overlap among stack objects handled by vertical placement on top of previous.
        """
        placed: list = []
        if dynamic_obj is None:
            logging.warning("place_static_objects_offset_stack: dynamic_obj is None")
            return placed

        force_max_stack = force_max_stack if force_max_stack is not None else max_stack_height
        if num_stacks is None:
            num_stacks = int(self.rng.randint(1, 4))  # 1-3 inclusive typical
        num_stacks = max(0, int(num_stacks))
        if num_stacks == 0:
            return placed

        cx, cy, cz = dynamic_obj.position
        base_centers: list[tuple[float, float]] = []
        attempts = 0
        max_attempts = num_stacks * 20
        while len(base_centers) < num_stacks and attempts < max_attempts:
            attempts += 1
            # Sample radius
            if radial_mode == "edge":
                r = float(self.rng.uniform(0.85 * offset_radius, offset_radius))
            elif radial_mode == "annulus":
                inner = 0.3 * offset_radius
                r = float(self.rng.uniform(inner, offset_radius))
            else:  # uniform disk
                r = float(self.rng.uniform(0.0, offset_radius))
            if avoid_center and r < 0.25 * offset_radius:
                r = float(self.rng.uniform(0.25 * offset_radius, offset_radius))
            theta = float(self.rng.uniform(0.0, 2 * np.pi))
            bx = cx + r * np.cos(theta)
            by = cy + r * np.sin(theta)

            # Separation constraint
            ok = True
            for (ox, oy) in base_centers:
                if (bx - ox) ** 2 + (by - oy) ** 2 < (min_separation ** 2):
                    ok = False
                    break
            if not ok:
                continue
            base_centers.append((bx, by))

        if len(base_centers) < num_stacks:
            logging.info(f"Requested {num_stacks} stack bases, created {len(base_centers)} after attempts={attempts}")

        for stack_index, (bx, by) in enumerate(base_centers):
            # Determine stack height
            if stack_size_mode == "fixed":
                stack_h = max(force_min_stack, min(force_max_stack, max_stack_height))
            else:  # random
                low = max(1, force_min_stack)
                high = min(max_stack_height, force_max_stack)
                try:
                    stack_h = int(self.rng.randint(low, high + 1))
                except Exception:
                    stack_h = low

            current_z_top = 0.0  # relative tower top elevation (world z updated each placement)
            for level in range(stack_h):
                # Sample object
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)

                # Random scale
                try:
                    s = float(self.rng.uniform(scale_range[0], scale_range[1]))
                    try:
                        obj.scale = s
                    except Exception:
                        obj.scale = (s, s, s)
                    obj.metadata["scale"] = s
                except Exception:
                    pass
                # scale = self.rng.uniform(1.0, 2.0)
                # obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])

                self._apply_object_physics(obj, obj_id)
                self.scene.add(obj)

                # After adding, adjust mass by volume scale and enable CCD to reduce tunneling
                client = getattr(self.simulator, "_physics_client", None)
                try:
                    body_id = obj.linked_objects[self.simulator]
                except Exception:
                    body_id = None
                if client is not None and body_id is not None:
                    # Scale mass ~ s^3 if dynamics info is available
                    try:
                        dyn = client.getDynamicsInfo(int(body_id), -1)
                        m0 = float(dyn[0]) if dyn and len(dyn) > 0 else 1.0
                        vol_scale = float(s) * float(s) * float(s)
                        m_new = max(1e-6, m0 * vol_scale)
                        client.changeDynamics(int(body_id), -1, mass=m_new, contactProcessingThreshold=0)
                    except Exception:
                        pass

                h = get_object_height_from_center(obj)
                # Gap proportional to height to avoid initial interpenetration
                gap = max(1e-3, 0.002 * float(h))
                # Place bottom at ground or on previous object's top
                base_z = max(0.0, cz * 0.0)  # ground plane approx at z=0
                z_pos = base_z + current_z_top + h + gap
                obj.position = (bx, by, z_pos)

                # Enable CCD radius (estimate from AABB diag or height)
                if client is not None and body_id is not None:
                    try:
                        aabb = client.getAABB(int(body_id), -1)
                        if aabb:
                            (xmin, ymin, zmin), (xmax, ymax, zmax) = aabb
                            diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
                            ccd_r = max(0.01, 0.1 * float(diag))
                        else:
                            ccd_r = max(0.02, 0.5 * float(h))
                        client.changeDynamics(int(body_id), -1, ccdSweptSphereRadius=float(ccd_r))
                    except Exception:
                        pass

                # Give the solver a small tick to settle contacts before stacking next
                try:
                    client.stepSimulation()
                except Exception:
                    pass

                # Update tower elevation
                try:
                    _, max_corner = get_world_object_bounds(obj)
                    current_z_top = max_corner[2]
                except Exception:
                    current_z_top += 2 * h

                # Metadata
                try:
                    obj.metadata["is_dynamic"] = False
                    obj.metadata["stack_index"] = stack_index
                    obj.metadata["stack_level"] = level
                except Exception:
                    pass
                placed.append(obj)
                self.static_objects.append(obj)

        logging.info(f"Placed {len(placed)} static objects in {len(base_centers)} offset stacks around dynamic object '{getattr(dynamic_obj, 'asset_id', 'unknown')}'")
        return placed

    def place_scene_default_static_objects(self, object_list=None, n_objects=None, center_region=None, mode: str = "kubasic"):
        
        if n_objects is None:
            n_objects = int(self.rng.randint(1, 5))

        if center_region is None:
            center_region = self.cfg.static_spawn_region
        
        if object_list is None:
            if mode == 'kubasic':
                object_list = KUBASIC_OBJECTS
            else:
                object_list = self.active_object_split
        
        placed = []

        #
        for _ in range(n_objects):
            attempts = 0
            chosen = None
            while attempts < 10:

                if mode == 'gso':
                    obj_id = self.rng.choice(object_list)
                    obj = self.gso.create(asset_id=obj_id)

                    scale = self.rng.uniform(3, 5)
                    obj.scale = scale
                    # obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
                    obj.metadata["scale"] = scale
                    # obj.static = True
                    self._apply_object_physics(obj, obj_id)

                    # Adjust mass by volume scale (s^3) to avoid instability
                    client = getattr(self.simulator, "_physics_client", None)
                    try:
                        body_id = obj.linked_objects[self.simulator]
                        if client is not None:
                            dyn = client.getDynamicsInfo(int(body_id), -1)
                            m0 = float(dyn[0]) if dyn and len(dyn) > 0 else 1.0
                            # Scale mass by volume (scale^3)
                            s = float(obj.scale[0]) if hasattr(obj.scale, '__len__') else float(obj.scale)
                            m_new = m0 * (s ** 3)
                            client.changeDynamics(int(body_id), -1, mass=m_new)
                    except Exception:
                        pass

                else:  # kubasic
                    obj_id = self.rng.choice(object_list)
                    # Use smaller scale for kubasic objects to fit in spawn region
                    # KuBasic objects are ~1m in size, so scale 0.4-0.8 gives 0.4-0.8m objects
                    size = float(self.rng.uniform(0.4, 0.8))
                    size_label = None
                    color_label, random_color = kb.randomness.sample_color("uniform_hue", self.rng)

                    material_name = self.rng.choice(["metal", "rubber"])
                    obj = self.kubasic.create(
                        asset_id=obj_id, scale=size,
                        name=f"{size_label} {color_label} {material_name} {obj_id}")
                    assert isinstance(obj, kb.FileBasedObject)

                    if material_name == "metal":
                        obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                                                roughness=0.2, ior=2.5)
                        obj.friction = 0.4
                        obj.restitution = 0.3
                        obj.mass *= 2.7 * size**3
                    else:  # material_name == "rubber"
                        obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                                                ior=1.25, roughness=0.7,
                                                                specular=0.33)
                        obj.friction = 0.8
                        obj.restitution = 0.7
                        obj.mass *= 1.1 * size**3

                    obj.metadata = {
                        "shape": obj.asset_id.lower(),
                        "size": size,
                        "size_label": size_label,
                        "material": material_name.lower(),
                        "color": random_color.rgb,
                        "color_label": color_label,
                    }

                self.scene.add(obj)

                # Sample position/orientation in the center region until no overlap
                # Use axis="Z" to only rotate around vertical axis, keeping furniture upright
                kb.resample_while(
                    obj,
                    [kb.randomness.position_sampler(center_region),
                     kb.randomness.rotation_sampler(axis="Z")],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )

                # Rest on ground plane approximately
                z_offset = get_object_height_from_center(obj) + 1e-3
                px, py, _ = obj.position
                obj.position = (px, py, z_offset)

                # # Sync physics body position after manual Z adjustment
                # try:
                #     body_id = obj.linked_objects[self.simulator]
                #     quat = self.simulator._physics_client.getBasePositionAndOrientation(int(body_id))[1]
                #     self.simulator._physics_client.resetBasePositionAndOrientation(
                #         int(body_id), [px, py, z_offset], quat
                #     )
                # except Exception:
                #     pass

                # Quick visibility check
                min_corner, max_corner = get_world_object_bounds(obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) *
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])

                if coverage > 0.01:  # small but visible
                    chosen = obj
                    # Mark as static
                    try:
                        chosen.metadata["is_dynamic"] = False
                    except Exception:
                        pass
                    break

                self.scene.remove(obj)
                del obj
                attempts += 1

            if chosen is None:
                # As a fallback, accept one object without strict visibility
                obj_id = self.rng.choice(object_list)
                chosen = self.gso.create(asset_id=obj_id)
                
                scale = self.rng.uniform(3, 5)
                chosen.scale = scale
                chosen.metadata["scale"] = scale
                # chosen.static = True

                self.scene.add(chosen)
                self._apply_object_physics(chosen, obj_id)

                # Adjust mass by volume scale (s^3)
                client = getattr(self.simulator, "_physics_client", None)
                try:
                    body_id = chosen.linked_objects[self.simulator]
                    if client is not None:
                        dyn = client.getDynamicsInfo(int(body_id), -1)
                        m0 = float(dyn[0]) if dyn and len(dyn) > 0 else 1.0
                        s = float(chosen.scale[0]) if hasattr(chosen.scale, '__len__') else float(chosen.scale)
                        m_new = m0 * (s ** 3)
                        client.changeDynamics(int(body_id), -1, mass=m_new)
                except Exception:
                    pass
                # Use axis="Z" to only rotate around vertical axis, keeping furniture upright
                kb.resample_while(
                    chosen,
                    [kb.randomness.position_sampler(center_region),
                     kb.randomness.rotation_sampler(axis="Z")],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )
                z_offset = get_object_height_from_center(chosen) + 1e-3
                px, py, _ = chosen.position
                chosen.position = (px, py, z_offset)
                try:
                    chosen.metadata["is_dynamic"] = False
                except Exception:
                    pass

            # Keep track and add to static list
            placed.append(chosen)
            self.static_objects.append(chosen)

        logging.info(f"Placed {len(placed)} center static objects in region {center_region}")
        return placed


    def place_center_static_objects(self, n_objects: int | None = None, center_region: list | None = None):
        """
        Place a few static objects near the scene center within a small spawn region.
        Objects are resampled until no overlap and minimal visibility are satisfied.

        n_objects: number of objects to place (defaults to 2-5)
        center_region: [[xmin, ymin, zmin], [xmax, ymax, zmax]] in meters. If None,
                        a small box around the center is derived from dynamic_spawn_region.
        """
        if n_objects is None:
            n_objects = int(self.rng.randint(4, 6))

        # Derive a compact central region if not provided
        if center_region is None:
            dsr = self.cfg.static_spawn_region
            (xmin, ymin, zmin), (xmax, ymax, zmax) = dsr
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax) # 0 0
            # shrink to a small box (30% of full width/height), near ground
            fx = 0.50 * (xmax - xmin)
            fy = 0.50 * (ymax - ymin)
            center_region = [
                [cx - fx, cy - fy, max(0.02, zmin * 0.0 + 0.02)],
                [cx + fx, cy + fy, 0.50],
            ]

        placed = []

        #
        for _ in range(n_objects):
            attempts = 0
            chosen = None
            while attempts < 12:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                self._apply_object_physics(obj, obj_id)

                # Sample position/orientation in the center region until no overlap
                kb.resample_while(
                    obj,
                    [kb.randomness.position_sampler(center_region),
                     kb.randomness.rotation_sampler(axis=None)],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )

                # Rest on ground plane approximately
                z_offset = get_object_height_from_center(obj) + 1e-3
                px, py, _ = obj.position
                obj.position = (px, py, z_offset)

                # Quick visibility check
                min_corner, max_corner = get_world_object_bounds(obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) *
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])

                if coverage > 0.01:  # small but visible
                    chosen = obj
                    # Mark as static
                    try:
                        chosen.metadata["is_dynamic"] = False
                    except Exception:
                        pass
                    break

                self.scene.remove(obj)
                del obj
                attempts += 1

            if chosen is None:
                # As a fallback, accept one object without strict visibility
                obj_id = self.rng.choice(self.active_object_split)
                chosen = self.gso.create(asset_id=obj_id)
                self.scene.add(chosen)
                self._apply_object_physics(chosen, obj_id)
                kb.resample_while(
                    chosen,
                    [kb.randomness.position_sampler(center_region),
                     kb.randomness.rotation_sampler(axis=None)],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )
                z_offset = get_object_height_from_center(chosen) + 1e-3
                px, py, _ = chosen.position
                chosen.position = (px, py, z_offset)
                try:
                    chosen.metadata["is_dynamic"] = False
                except Exception:
                    pass

            # Keep track and add to static list
            placed.append(chosen)
            self.static_objects.append(chosen)

        logging.info(f"Placed {len(placed)} center static objects in region {center_region}")
        return placed

    def place_center_static_stack_objects(
        self,
        n_stacks: int | None = None,
        center_region: list | None = None,
        max_stack_height: int = 3,
        min_stack_height: int = 1,
        stack_size_mode: str = "random",
        scale_range: tuple[float, float] = (1.0, 1.5),
        min_separation: float = 0.3,
    ):
        """
        Place stacks of static objects in the center region. Each stack position
        can have multiple objects stacked vertically.

        Parameters:
            n_stacks: number of stack positions to create. If None, random 2-4.
            center_region: [[xmin, ymin, zmin], [xmax, ymax, zmax]] in meters.
                           If None, derived from static_spawn_region.
            max_stack_height: maximum number of objects per stack (default 3).
            min_stack_height: minimum number of objects per stack (default 1).
            stack_size_mode: 'random' for random height in [min, max], 'fixed' for max_stack_height.
            scale_range: (min_scale, max_scale) for random scaling of each object.
            min_separation: minimum horizontal distance between stack base centers.

        Returns:
            List of all placed static objects.
        """
        if n_stacks is None:
            n_stacks = int(self.rng.randint(2, 5))  # 2-4 stacks

        # Derive center region if not provided
        if center_region is None:
            dsr = self.cfg.static_spawn_region
            (xmin, ymin, zmin), (xmax, ymax, zmax) = dsr
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            fx = 0.50 * (xmax - xmin)
            fy = 0.50 * (ymax - ymin)
            center_region = [
                [cx - fx, cy - fy, 0.0],
                [cx + fx, cy + fy, 0.5],
            ]

        placed = []
        base_centers: list[tuple[float, float]] = []

        # Collect existing object positions from scene (including static_objects)
        existing_positions = []
        for obj in self.static_objects:
            try:
                ox, oy, oz = obj.position
                # Estimate object radius from bounds
                try:
                    min_c, max_c = get_world_object_bounds(obj)
                    radius = max(abs(max_c[0] - min_c[0]), abs(max_c[1] - min_c[1])) / 2.0
                except Exception:
                    radius = 0.3  # fallback
                existing_positions.append((ox, oy, radius))
            except Exception:
                pass

        # Sample n_stacks base positions with minimum separation
        (rx_min, ry_min, _), (rx_max, ry_max, _) = center_region
        attempts = 0
        max_attempts = n_stacks * 50

        while len(base_centers) < n_stacks and attempts < max_attempts:
            attempts += 1
            bx = float(self.rng.uniform(rx_min, rx_max))
            by = float(self.rng.uniform(ry_min, ry_max))

            # Check separation from existing bases (within this function)
            ok = True
            for (ox, oy) in base_centers:
                if (bx - ox) ** 2 + (by - oy) ** 2 < (min_separation ** 2):
                    ok = False
                    break
            
            # Check separation from already placed static objects in scene
            if ok:
                for (ox, oy, obj_radius) in existing_positions:
                    required_dist = min_separation + obj_radius
                    if (bx - ox) ** 2 + (by - oy) ** 2 < (required_dist ** 2):
                        ok = False
                        break
            
            if ok:
                base_centers.append((bx, by))

        if len(base_centers) < n_stacks:
            logging.info(f"Requested {n_stacks} stack bases, created {len(base_centers)} after {attempts} attempts")

        # For each base position, create a stack
        for stack_index, (bx, by) in enumerate(base_centers):
            # Determine stack height
            if stack_size_mode == "fixed":
                stack_h = max_stack_height
            else:  # random
                low = max(1, min_stack_height)
                high = max(low, max_stack_height)
                stack_h = int(self.rng.randint(low, high + 1))

            current_z_top = 0.0  # Track top of current stack

            for level in range(stack_h):
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)

                # Random scale
                try:
                    s = float(self.rng.uniform(scale_range[0], scale_range[1]))
                    try:
                        obj.scale = s
                    except Exception:
                        obj.scale = (s, s, s)
                    obj.metadata["scale"] = s
                except Exception:
                    s = 1.0

                self._apply_object_physics(obj, obj_id)
                self.scene.add(obj)

                # Adjust mass by volume scale
                client = getattr(self.simulator, "_physics_client", None)
                try:
                    body_id = obj.linked_objects[self.simulator]
                except Exception:
                    body_id = None

                if client is not None and body_id is not None:
                    try:
                        dyn = client.getDynamicsInfo(int(body_id), -1)
                        m0 = float(dyn[0]) if dyn and len(dyn) > 0 else 1.0
                        vol_scale = s * s * s
                        m_new = max(1e-6, m0 * vol_scale)
                        client.changeDynamics(int(body_id), -1, mass=m_new)
                    except Exception:
                        pass

                # Calculate height and position
                h = get_object_height_from_center(obj)
                gap = max(1e-3, 0.002 * float(h))
                z_pos = current_z_top + h + gap
                obj.position = (bx, by, z_pos)

                # Random rotation around Z axis only (keep upright)
                try:
                    angle = float(self.rng.uniform(0, 2 * np.pi))
                    # Convert to quaternion (rotation around Z)
                    qw = np.cos(angle / 2)
                    qx, qy = 0.0, 0.0
                    qz = np.sin(angle / 2)
                    obj.quaternion = (qw, qx, qy, qz)
                except Exception:
                    pass

                # Check overlap with existing objects and resample if needed (only for base level)
                if level == 0:
                    resample_attempts = 0
                    while self.simulator.check_overlap(obj) and resample_attempts < 5:
                        # Try a slightly different position
                        offset_x = float(self.rng.uniform(-0.1, 0.1))
                        offset_y = float(self.rng.uniform(-0.1, 0.1))
                        obj.position = (bx + offset_x, by + offset_y, z_pos)
                        resample_attempts += 1

                # Enable CCD
                if client is not None and body_id is not None:
                    try:
                        ccd_r = max(0.02, 0.3 * float(h))
                        client.changeDynamics(int(body_id), -1, ccdSweptSphereRadius=float(ccd_r))
                    except Exception:
                        pass

                # Update stack top
                try:
                    _, max_corner = get_world_object_bounds(obj)
                    current_z_top = max_corner[2]
                except Exception:
                    current_z_top += 2 * h

                # Metadata
                try:
                    obj.metadata["is_dynamic"] = False
                    obj.metadata["stack_index"] = stack_index
                    obj.metadata["stack_level"] = level
                except Exception:
                    pass

                placed.append(obj)
                self.static_objects.append(obj)
                
                # Add to existing positions for subsequent stacks
                try:
                    min_c, max_c = get_world_object_bounds(obj)
                    radius = max(abs(max_c[0] - min_c[0]), abs(max_c[1] - min_c[1])) / 2.0
                    existing_positions.append((bx, by, radius))
                except Exception:
                    existing_positions.append((bx, by, 0.2))

        logging.info(f"Placed {len(placed)} static objects in {len(base_centers)} stacks (center region)")
        return placed
    
    def move_camera_back(self):
        while not all(obj_drop_visible_from_camera(obj,self.scene) for obj in self.dynamic_objects):
            logging.info("Moving camera back...")
            position = (
                self.scene.camera.position[0] - abs(self.rng.normal(0, 0.05)), 
                self.scene.camera.position[1], 
                self.scene.camera.position[2]
            )
            self.scene.camera.position = position
    
    def move_camera_back_static(self):
        while not all(obj_drop_visible_from_camera(obj,self.scene) for obj in self.static_objects):
            logging.info("Moving camera back...")
            position = (
                self.scene.camera.position[0] - abs(self.rng.normal(0, 0.05)), 
                self.scene.camera.position[1], 
                self.scene.camera.position[2]
            )
            self.scene.camera.position = position

    def setup_camera_for_slope(self):
        """Position camera to observe the slope and sliding objects from a side angle."""
        params = getattr(self, '_slope_params', None)
        if not params:
            return
        
        # Get slope dimensions
        L = params.get('L', 3.0)
        high_z = params.get('high_point', (0, 0, 1.0))[2]
        
        # Position camera to the side of the slope, looking at the middle
        # Camera should be able to see both the top and bottom of the slope
        slope_center_x = 0.0
        slope_center_z = high_z / 2.0
        
        # Camera at a 45-degree angle to the side, elevated
        cam_distance = L * 1.5
        cam_x = -cam_distance * 0.7
        cam_y = cam_distance * 0.7  # offset to the side
        cam_z = max(1.0, high_z * 0.8)
        
        self.scene.camera.position = (float(cam_x), float(cam_y), float(cam_z))
        
        # Look at the middle of the slope
        try:
            self.scene.camera.look_at((slope_center_x, 0.0, slope_center_z))
        except Exception:
            pass
        
        logging.info(f"Camera positioned for slope view: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")
        
        # Ensure all objects are visible
        self.move_camera_back()

    def setup_lights(
        self, 
    ):
        # this code is borrowed from TaskMeAnything (https://github.com/JieyuZ2/TaskMeAnything/blob/906f7f6ed5ddfd5ec5e94f3c4a5eb6e9670e2e72/tma/imageqa/tabletop_3d/run_blender.py#L223)
        orientation = self.rng.choice([-1, 1])
        key_light_horizontal_angle = orientation * radians(self.rng.uniform(15, 45))
        fill_light_horizontal_angle = - orientation * radians(self.rng.uniform(15, 60))
        key_light_vertical_angle = -radians(self.rng.uniform(15, 45))
        fill_light_vertical_angle = -radians(self.rng.uniform(0, 30))

        sun_x, sun_y = radians(self.rng.uniform(0, 45)), radians(self.rng.uniform(0, 45))
        sun_energy = self.rng.uniform(1.0, 6.0)

        
        # for setting up the three point lighting, we mostly follow https://courses.cs.washington.edu/courses/cse458/05au/reading/3point_lighting.pdf
        # in order to keep lights and camera on the hemisphere pointing to origin, we use a hierarchy of empties

        # create the sun

        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.rotation_euler = Euler((sun_x, sun_y, 0), "XYZ")
        sun.data.energy = sun_energy

        # create global empty

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
        empty = bpy.context.scene.objects.get("Empty")


        # radius = random.uniform(1.8,2.2)
        radius = 2.5

        # bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-radius, 0, 0), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
        cam = bpy.context.scene.objects.get("PerspectiveCamera")

        # create camera empty

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=cam.location, scale=(1, 1, 1))
        x_rot, y_rot, z_rot = radians(90), radians(0), radians(270)
        cam_empty = bpy.context.scene.objects.get("Empty.001")
        cam_empty.name = "camera_empty"

        # make camera empty parent of camera

        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)
        cam_empty.select_set(True)
        bpy.context.view_layer.objects.active = cam_empty
        bpy.ops.object.parent_set()
        bpy.ops.object.select_all(action='DESELECT')

        # make camera empty parent of global empty

        bpy.ops.object.select_all(action='DESELECT')
        cam_empty.select_set(True)
        empty.select_set(True)
        bpy.context.view_layer.objects.active = empty
        bpy.ops.object.parent_set()
        bpy.ops.object.select_all(action='DESELECT')

        light_names = ["key_light", "fill_light", "back_light"]
        light_energies = [100., 75., 75.]

        for light_name, light_energy in zip(light_names, light_energies):
            # create light empty

            empty_name = light_name + "_empty"
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
            light_empty = bpy.context.scene.objects.get("Empty.001")
            light_empty.name = empty_name

            # parent light empty to main (camera) empty

            bpy.ops.object.select_all(action='DESELECT')
            light_empty.select_set(True)
            empty.select_set(True)
            bpy.context.view_layer.objects.active = empty
            bpy.ops.object.parent_set()
            bpy.ops.object.select_all(action='DESELECT')

            # create light

            x_loc, y_loc, z_loc = -radius, 0, 0
            bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(x_loc, y_loc, z_loc), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
            bpy.data.objects["Point"].name = light_name
            light = bpy.data.objects[light_name]
            light.data.energy = light_energy
            # light.data.size = 0.5

            # parent light empty to light

            bpy.ops.object.select_all(action='DESELECT')
            light.select_set(True)
            light_empty.select_set(True)
            bpy.context.view_layer.objects.active = light_empty
            bpy.ops.object.parent_set()
            bpy.ops.object.select_all(action='DESELECT')

       

        bpy.context.view_layer.update()

        back_light_horizontal_angle = radians(180)
        light_horizontal_angles = [key_light_horizontal_angle, fill_light_horizontal_angle, back_light_horizontal_angle]
        for light_angle, light_name in zip(light_horizontal_angles, light_names):
            light_empty = bpy.data.objects[light_name + "_empty"]
            global_z = (light_empty.matrix_world.inverted() @ Vector((0.0, 0.0, 1.0, 0.0)))[:3]
            quat = Quaternion(global_z, light_angle)
            light_empty.rotation_euler = quat.to_euler()

        back_light_vertical_angle = 0
        light_vertical_angles = [key_light_vertical_angle, fill_light_vertical_angle, back_light_vertical_angle]

        for light_angle, light_name in zip(light_vertical_angles, light_names):
            light_empty = bpy.data.objects[light_name + "_empty"]
            global_x = (light_empty.matrix_world.inverted() @ Vector((1.0, 0.0, 0.0, 0.0)))[:3]
            quat = Quaternion(global_x, light_angle)
            euler_add = quat.to_euler()
            euler_current = light_empty.rotation_euler
            new_euler = Euler((euler_add[0] + euler_current[0], euler_add[1] + euler_current[1], euler_add[2] + euler_current[2]))
            light_empty.rotation_euler = new_euler
            light = bpy.data.objects[light_name]
            light.location = [light.location[0], light.location[1], 1.5]


        return cam, empty

    def ensure_physics_ground(self):
        """
        Add an invisible static collider aligned so that its top surface is at z=0.
        This guarantees PyBullet has a ground even when the visual ground is a shadow catcher
        or disabled. The collider is a large, thin kb.Cube hidden from rendering.
        """
        if getattr(self, "_physics_ground_added", False):
            # If already added, still apply any updated overrides to dynamics
            try:
                if hasattr(self, "_physics_ground_obj") and self._physics_ground_obj is not None:
                    mu = float(getattr(self, "_ground_friction_override", 0.8))
                    e = float(getattr(self, "_ground_restitution_override", 0.0))
                    try:
                        self._physics_ground_obj.friction = mu
                        self._physics_ground_obj.restitution = e
                    except Exception:
                        pass
                    try:
                        body_id = self._physics_ground_obj.linked_objects[self.simulator]
                        self.simulator._physics_client.changeDynamics(int(body_id), -1, lateralFriction=mu, restitution=e)
                    except Exception:
                        pass
            except Exception:
                pass
            return

        size = float(self.args.physics_ground_size) if self.args.physics_ground_size else float(self.args.ground_size) if hasattr(self.args, 'ground_size') else 40.0
        thickness = max(1e-3, float(self.args.physics_ground_thickness))
        mu = float(getattr(self.args, "_ground_friction_override", 0.8))
        e = float(getattr(self.args, "_ground_restitution_override", 0.0))

        try:
            ground_phys = kb.Cube(name="physics_ground",
                                  scale=(size, size, thickness),
                                  friction=mu,
                                  restitution=e,
                                  static=True)
        except Exception:
            # Fallback if kb.Cube cannot be constructed from current kubric import style
            ground_phys = kb.core.Cube(name="physics_ground",
                                       scale=(size, size, thickness),
                                       friction=mu,
                                       restitution=e,
                                       static=True)

        # Position so the top face sits at z=0
        ground_phys.position = (0.0, 0.0, -thickness / 2.0)
        try:
            self.scene.add(ground_phys)
        except Exception:
            self.scene += ground_phys

        # Hide from rendering (keep only as collider)
        try:
            bl = ground_phys.linked_objects[self.renderer]
            bl.hide_render = True
            bl.hide_viewport = True
        except Exception:
            pass

        # Store and push dynamics overrides if possible
        try:
            self._physics_ground_obj = ground_phys
        except Exception:
            pass
        try:
            body_id = ground_phys.linked_objects[self.simulator]
            self.simulator._physics_client.changeDynamics(int(body_id), -1, lateralFriction=mu, restitution=e)
        except Exception:
            pass

        self._physics_ground_added = True


 
    @time_limit(1200) # 20 minute timeout
    def render_a_scene(self):
        self.configure_split()
        self.setup_scene_background()
        self.setup_camera()
        self.dynamic_objects = []
        self.static_objects = []
        self.metadata["static_objects"] = []
        self.metadata["dynamic_objects"] = []

        # combined scenario
        if self.args.scenario == "combined":
            if self.args.stage1 == "throw":
                towers = []

                # towers = self.place_center_static_objects()
                gso_list = ["DINNING_ROOM_FURNITURE_SET_1","BEDROOM_CLASSIC", "BEDROOM_CLASSIC_Gi22DjScTVS", "GARDEN_SWING"]
                mode = self.rng.choice(['gso','kubasic'])
                towers = self.place_scene_default_static_objects(mode=mode, center_region=([-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]))
                towers.extend(self.place_center_static_stack_objects())
                
                self.simulator.run(frame_start=-200, frame_end=0)

                # self.throw_objects = self.place_throw_objects()
                self.dynamic_objects = self.place_radius_throw_objects(min_radius=1.5, max_radius=2.0, height_range=(0.5,0.8), speed_range=(2.0,4.0))
                # self.dynamic_objects = self.throw_objects

                base, ext = os.path.splitext(args.scene_save_path)
                if args.scene_save_path:
                    save_state(self.renderer, f"{base}_combined_setup{ext}")
                if args.scene_save_path:
                    save_state(self.renderer, f"{base}_combined_settled{ext}")
                
                # for i, obj in enumerate(self.dynamic_objects):
                #     obj = copy.deepcopy(obj)
                #     self.dynamic_objects[i] = obj
                #     self.scene.add(obj) # Workaround
                
                logging.info(f"Placed {len(self.dynamic_objects)} thrown objects for combined scenario.")

                # for dyn_obj, tower in zip(self.dynamic_objects, towers):
                #     if tower:
                #         max_height = max(
                #             get_world_object_bounds(obj)[1][2] for obj in tower
                #         )
                #         min_corner, _ = get_world_object_bounds(dyn_obj)
                #         if min_corner[2] < max_height:
                #             logging.info("Raising throwed object...")
                #             dyn_obj.position = (dyn_obj.position[0], dyn_obj.position[1], max_height + abs(self.rng.normal(0.005, 0.005)))
            
            # objects fall
            else:

                # # add center static obstacles
                # self.place_center_static_objects()

                # add dynamic objects on towers
                towers = []
                self.dynamic_objects = self.place_dynamic_objects()
                for dynamic_obj in self.dynamic_objects:
                    # towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
                    stacks = self.rng.randint(1,4)
                    offset = self.rng.uniform(0.0, 0.1)
                    scale = (1.0, 2.0)
                    towers.append(self.place_static_objects_offset_stack(dynamic_obj, num_stacks=stacks, offset_radius=offset, scale_range=scale))
                
                mode = self.rng.choice(['gso','kubasic'])
                self.place_scene_default_static_objects(mode=mode, center_region=([-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]))

                base, ext = os.path.splitext(args.scene_save_path)
                if args.scene_save_path:
                    save_state(self.renderer, f"{base}_combined_setup{ext}")
                self.simulator.run(frame_start=-200, frame_end=0)
                if args.scene_save_path:
                    save_state(self.renderer, f"{base}_combined_settled{ext}")
                
                for i,obj in enumerate(self.dynamic_objects):
                    obj = copy.deepcopy(obj)
                    self.dynamic_objects[i] = obj
                    self.scene.add(obj) # Workaround

                for dyn_obj, tower in zip(self.dynamic_objects, towers):
                    if tower:
                        max_height = max(
                            get_world_object_bounds(obj)[1][2] for obj in tower
                        )
                        min_corner, _ = get_world_object_bounds(dyn_obj)
                        if min_corner[2] < max_height:
                            logging.info("Raising dynamic object...")
                            dyn_obj.position = (dyn_obj.position[0], dyn_obj.position[1], max_height + abs(self.rng.normal(0.005, 0.005)))
            

        elif self.args.scenario == "throw":
            # add center static obstacles
            # self.place_center_static_objects()
            mode = self.rng.choice(['gso','kubasic'])
            self.place_scene_default_static_objects(mode=mode, center_region=([-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]))
            self.place_center_static_stack_objects()
            self.simulator.run(frame_start=-200, frame_end=0)

            # add dynamic objects
            self.dynamic_objects = self.place_radius_throw_objects(min_radius=1.0, max_radius=2.0, height_range=(0.3,0.6), speed_range=(2.0,4.0))

            base, ext = os.path.splitext(args.scene_save_path)
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_throw_setup{ext}")
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_throw_settled{ext}")
        
        # collision scenario
        elif self.args.scenario == "collision":

            # add center static obstacles
            # self.place_center_static_objects()
            mode = self.rng.choice(['gso','kubasic'])
            self.place_scene_default_static_objects(mode=mode, center_region=([-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]))
            self.place_center_static_stack_objects()
            self.simulator.run(frame_start=-200, frame_end=0)

            # add dynamic objects
            self.dynamic_objects = self.place_collision_objects(min_radius=1.0, max_radius=2.0)

            base, ext = os.path.splitext(args.scene_save_path)
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_collision_setup{ext}")
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_collision_settled{ext}")

            # Re-apply intended initial velocities AFTER preroll so motion starts at frame 0
            # self._apply_initial_velocities(self.dynamic_objects)
        
        
        # Slope scenario: add slope and let objects slide down
        elif self.args.scenario == 'slope':
            logging.info("Setting up SLOPE scenario...")
            
            # Add the slope structure
            slope = self.add_slope()
            if slope is None:
                logging.error("Failed to create slope, falling back to default scenario")
                self.static_objects = []
                self.dynamic_objects = []
            else:
                # self.static_objects = [slope]
                
                # Place dynamic objects at the top of the slope
                self.dynamic_objects = self.place_objects_on_slope(
                    slope, 
                    count=self.num_dynamic_objects
                )
                
                # # Preroll to let objects settle onto the slope surface
                # # Use more frames for proper settling
                # try:
                #     logging.info("Running preroll simulation to settle objects on slope...")
                #     self.simulator.run(frame_start=-10, frame_end=0)
                # except Exception as ex:
                #     logging.warning(f"Preroll simulation failed: {ex}")
                
                # Save scene state if requested
                base, ext = os.path.splitext(args.scene_save_path)
                if args.scene_save_path:
                    save_state(self.renderer, f"{base}_slope_setup{ext}")
                
                # Adjust camera for slope view
                self.setup_camera_for_slope()
            
            logging.info(f"Slope scenario setup complete: {len(self.dynamic_objects)} objects on slope")

        # fall scenario
        else:
            towers = []
            self.dynamic_objects = self.place_dynamic_objects()
            for dynamic_obj in self.dynamic_objects:
                # towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
                stacks = self.rng.randint(1,4)
                offset = self.rng.uniform(0.0, 0.1)
                scale = (1.0, 2.0)
                towers.append(self.place_static_objects_offset_stack(dynamic_obj, num_stacks=stacks, offset_radius=offset, scale_range=scale))
            base, ext = os.path.splitext(args.scene_save_path)
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_tower_setup{ext}")
            self.simulator.run(frame_start=-200, frame_end=0)
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_tower_settled{ext}")
            
            for i,obj in enumerate(self.dynamic_objects):
                obj = copy.deepcopy(obj)
                self.dynamic_objects[i] = obj
                self.scene.add(obj) # Workaround

            for dyn_obj, tower in zip(self.dynamic_objects, towers):
                if tower:
                    max_height = max(
                        get_world_object_bounds(obj)[1][2] for obj in tower
                    )
                    min_corner, _ = get_world_object_bounds(dyn_obj)
                    if min_corner[2] < max_height:
                        logging.info("Raising dynamic object...")
                        dyn_obj.position = (dyn_obj.position[0], dyn_obj.position[1], max_height + abs(self.rng.normal(0.005, 0.005)))
                    
        
        if self.args.camera_motion in ["static", "fixed_random"]:
            self.move_camera_back()
            # self.move_camera_back_static()
        
        # # thrown objects don't have to be shown in the first frame of the scene
        # if self.args.stage1 == "throw" and self.args.scenario == "combined":
        #     self.dynamic_objects.extend(self.throw_objects)

        # For 'combined', run the simulation in two stages and spawn offscreen entrants at t=split
        if self.args.scenario == "combined":
            split = self.args.frame_end // 2
            total = self.args.frame_end

            logging.info(f"[combined] Stage A simulation 0..{split}")
            self.simulator.run(frame_start=0, frame_end=split)

            # Spawn offscreen entrants now so they appear only in the second half
            # self.late_dynamic_objects = self.place_offscreen_influx(
            #     count=self.args.combined_influx_count,
            #     speed=float(self.args.stage2_speed),
            #     margin=float(self.args.combined_margin),
            # )
            self.late_dynamic_objects = self.place_collision_objects(min_radius=1.0, max_radius=2.0)
            self.dynamic_objects.extend(self.late_dynamic_objects)

            # Important: ensure newly added objects are registered in PyBullet before
            # applying velocities; then explicitly (re-)apply their velocities so they move.
            try:
                # Zero-advance run to force registration at the current frame index
                self.simulator.run(frame_start=split, frame_end=split)
            except Exception as e:
                logging.warning(f"[combined] registration tick failed: {e}")

            # Option 1 (preferred): accelerate using an external force pulse for realism
            
            if self.args.stage2_force == "none":
                stepped = 0
                # self._apply_initial_velocities(self.late_dynamic_objects, velocity_mode=self.args.stage2)

            elif self.args.stage2_force == "impulse" or self.args.stage2_force == "force":
                try:
                    stepped = self._accelerate_objects_with_force(
                        self.late_dynamic_objects,
                        target_speed=float(self.args.stage2_speed),
                        dir_mode=self.args.stage2,
                        duration_s=self.args.stage2_force_duration,
                        start_frame=split,
                        force_mode=self.args.stage2_force,
                    )
                except Exception as e:
                    logging.warning(f"[combined] external-force acceleration failed: {e}")
                    stepped = 0

                # # Fallback (if force pulse not applied): directly set base velocities
                # if stepped <= 0:
                #     try:
                #         self._apply_initial_velocities(self.late_dynamic_objects, velocity_mode=self.args.stage2)
                #     except Exception as e:
                #         logging.warning(f"[combined] failed to apply velocities to late entrants: {e}")

            # Continue Stage B after any force-pulse frames
            stage_b_start = split + max(0, stepped)
            logging.info(f"[combined] Stage B simulation {stage_b_start}..{total}")
            self.simulator.run(frame_start=stage_b_start, frame_end=total + 1)

        self.metadata["camera"] = get_camera_metadata(self.scene.camera)
        
        for obj in self.dynamic_objects:
            self.metadata["dynamic_objects"].append(get_object_metadata(obj))
        for obj in self.static_objects:
            self.metadata["static_objects"].append(get_object_metadata(obj))
        
        self.setup_lights()
        if args.scene_save_path:
            save_state(self.renderer,f"{base}_init{ext}")

        if self.args.scenario not in ("combined",):
            logging.info("Running simulation...")
            self.simulator.run(frame_start=0, frame_end=self.scene.frame_end+1)

        logging.info("Rendering frames...")
        if self.num_threads is not None:
            bpy.context.scene.render.threads_mode = 'FIXED'
            bpy.context.scene.render.threads = self.num_threads
        
        # # original code
        # data_stack = self.renderer.render(return_layers=["rgba","segmentation","depth"]) 
    
        # os.makedirs(self.output_dir, exist_ok=True)
        # logging.info("Saving metadata...")
        # metadata_path = os.path.join(self.output_dir, "metadata.json")
        # with open(metadata_path, "w") as f:
        #     json.dump(self.metadata, f, indent=4)
        
        # for i in range(data_stack['rgba'].shape[0]):
        #     img_path = os.path.join(self.output_dir, f"rgba_{i:05d}.jpg")
        #     io.imsave(img_path, 
        #             data_stack["rgba"][i][..., :3])
            
        #     np.save(os.path.join(self.output_dir, f"segmentation_{i:05d}.npy"), data_stack["segmentation"][i])
        #     np.save(os.path.join(self.output_dir, f"depth_{i:05d}.npy"), data_stack["depth"][i])

        # New render code for multi-view support
        views = self._choose_camera_views(self.args.num_views, self.args.view_sampling)
        if self.args.num_views <= 1 or not views:
            views = [{
                "position": tuple(self.scene.camera.position),
                "look_at": (0.0, 0.0, 0.0)
            }]

        for vidx, vp in enumerate(views, start=1):
            # Set camera pose
            self.scene.camera.position = vp["position"]
            try:
                self.scene.camera.look_at(vp.get("look_at", (0.0, 0.0, 0.0)))
            except Exception:
                pass
            
            # # Render
            # # Also request optical flow from Kubric's Blender renderer
            # # Supported layers include "forward_flow" and "backward_flow"
            # data_stack = self.renderer.render(return_layers=[
            #     "rgba", "segmentation", "depth", "forward_flow", "backward_flow"
            # ]) 

            # Decide flow rendering strategy
            want_flow = (not self.args.disable_flow)
            flow_mode = str(getattr(self.args, 'flow_render_mode', 'auto')).lower()
            eng_choice = str(getattr(self.args, 'engine', 'cycles') or 'cycles').lower()
            if flow_mode == 'auto':
                flow_mode = 'split' if eng_choice == 'eevee' else 'same'

            data_stack = {}
            if want_flow and flow_mode == 'same':
                # Request everything in one pass (requires Cycles to get flow reliably)
                layers = ["rgba", "segmentation", "depth", "forward_flow", "backward_flow"]
                # If user chose eevee but still forced 'same', warn and fallback to cycles temporarily
                prev_engine = bpy.context.scene.render.engine
                if eng_choice == 'eevee' and prev_engine != 'CYCLES':
                    try:
                        bpy.context.scene.render.engine = 'CYCLES'
                        logging.info("Temporarily switched to CYCLES for joint flow rendering (flow_render_mode=same).")
                    except Exception:
                        logging.warning("Failed to switch to CYCLES; optical flow may be unavailable.")
                data_stack = self.renderer.render(return_layers=layers)
                # Restore engine if changed
                if eng_choice == 'eevee' and prev_engine != bpy.context.scene.render.engine:
                    try:
                        bpy.context.scene.render.engine = prev_engine
                    except Exception:
                        pass
            elif want_flow and flow_mode == 'split' and eng_choice == 'eevee':
                eevee_time = time.time()
                logging.info("Using split Eevee+Cycles rendering for optical flow...")
                # Pass 1: Eevee fast main passes (no flow)
                main_layers = ["rgba", "segmentation", "depth"]
                data_main = self.renderer.render(return_layers=main_layers)
                eevee_time = time.time() - eevee_time
                logging.info(f"Eevee main pass took {eevee_time:.2f} seconds.")

                # Pass 2: switch to Cycles for flow only
                cycles_time = time.time()
                prev_engine = bpy.context.scene.render.engine
                
                # Save previous settings to restore later
                prev_samples = self.renderer.samples_per_pixel
                logging.info(f"Previous samples per pixel: {prev_samples}")
                prev_use_denoising = getattr(self.renderer, "use_denoising", True)
                prev_cycles_settings = {}
                try:
                    c = bpy.context.scene.cycles
                    prev_cycles_settings = {
                        "max_bounces": c.max_bounces,
                        "diffuse_bounces": c.diffuse_bounces,
                        "glossy_bounces": c.glossy_bounces,
                        "transmission_bounces": c.transmission_bounces,
                        "volume_bounces": c.volume_bounces,
                        "transparent_max_bounces": c.transparent_max_bounces,
                    }
                except Exception:
                    pass

                try:
                    bpy.context.scene.render.engine = 'CYCLES'

                    # Verify OptiX usage
                    try:
                        prefs = bpy.context.preferences.addons['cycles'].preferences
                        logging.info(f"Cycles Compute Device Type: {prefs.compute_device_type}")
                        for dev in prefs.devices:
                            if dev.use:
                                logging.info(f"Using Device: {dev.name} ({dev.type})")
                    except Exception as e:
                        logging.warning(f"Failed to check Cycles devices: {e}")
                    
                    # 1. Samples = 1 (Geometric pass doesn't need convergence)
                    self.renderer.samples_per_pixel = 1 
                    
                    # 2. Disable Denoising (Not needed for vector pass)
                    if hasattr(self.renderer, "use_denoising"):
                        self.renderer.use_denoising = False
                    try:
                        bpy.context.scene.cycles.use_denoising = False
                    except Exception:
                        pass

                    # 3. Minimize Light Bounces (Geometry is independent of lighting)
                    # BlenderProc uses: (1, 0, 0, 1, 0, 8, 0) -> max, diffuse, glossy, trans, vol, transp
                    try:
                        c = bpy.context.scene.cycles
                        c.max_bounces = 1
                        c.diffuse_bounces = 0
                        c.glossy_bounces = 0
                        c.transmission_bounces = 0
                        c.volume_bounces = 0
                        c.transparent_max_bounces = 8
                    except Exception:
                        pass

                    # # 4. Enable Persistent Data to avoid re-syncing geometry per frame
                    # bpy.context.scene.render.use_persistent_data = True

                    flow_layers = ["forward_flow", "backward_flow"]
                    data_flow = self.renderer.render(return_layers=flow_layers)
                finally:
                    # Restore samples, engine, and settings
                    try:
                        if prev_engine:
                            bpy.context.scene.render.engine = prev_engine
                    except Exception:
                        pass
                    
                    try:
                        self.renderer.samples_per_pixel = prev_samples
                        if hasattr(self.renderer, "use_denoising"):
                            self.renderer.use_denoising = prev_use_denoising
                        
                        # Restore Cycles settings
                        c = bpy.context.scene.cycles
                        for k, v in prev_cycles_settings.items():
                            setattr(c, k, v)
                    except Exception:
                        pass
                cycles_time = time.time() - cycles_time
                logging.info(f"Cycles flow pass took {cycles_time:.2f} seconds.")
                # Merge
                data_stack = {
                    **data_main,
                    **data_flow,
                }
            

            
            else:
                # No flow requested or engine already cycles but flow disabled
                layers = ["rgba", "segmentation", "depth"]
                data_stack = self.renderer.render(return_layers=layers)

            # Output directory per view
            view_dir = self.output_dir if self.args.num_views == 1 else os.path.join(self.output_dir, f"view_{vidx:02d}")
            os.makedirs(view_dir, exist_ok=True)

            # Save metadata (with per-view camera) and frames
            logging.info(f"Saving metadata & frames for view {vidx} -> {view_dir}")
            meta = dict(self.metadata)
            meta["camera_view_index"] = vidx
            meta["camera_pose"] = {
                "position": list(self.scene.camera.position),
            }
            with open(os.path.join(view_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4, cls=NumpyEncoder)

            for i in range(data_stack['rgba'].shape[0]):
                img_path = os.path.join(view_dir, f"rgba_{i:05d}.jpg")
                io.imsave(img_path, data_stack["rgba"][i][..., :3])
                np.save(os.path.join(view_dir, f"segmentation_{i:05d}.npy"), data_stack["segmentation"][i])
                np.save(os.path.join(view_dir, f"depth_{i:05d}.npy"), data_stack["depth"][i])
                # Save optical flow as numpy arrays (HxWx2 float32): forward and backward
                if want_flow and "forward_flow" in data_stack:
                    try:
                        if i < data_stack["forward_flow"].shape[0]:
                            np.save(os.path.join(view_dir, f"forward_flow_{i:05d}.npy"), data_stack["forward_flow"][i])
                    except Exception as e:
                        logging.warning(f"Failed to save forward_flow for frame {i}: {e}")
                if want_flow and "backward_flow" in data_stack:
                    try:
                        if i < data_stack["backward_flow"].shape[0]:
                            np.save(os.path.join(view_dir, f"backward_flow_{i:05d}.npy"), data_stack["backward_flow"][i])
                    except Exception as e:
                        logging.warning(f"Failed to save backward_flow for frame {i}: {e}")

            # Optional videos per view
            if args.save_mp4 or args.save_gif:
                save_video(view_dir, "rgba.mp4", "rgba_%05d.jpg", fps=args.frame_rate)
            if args.save_gif:
                video_path = os.path.join(view_dir, "rgba.mp4")
                gif_path = os.path.join(view_dir, "rgba.gif")
                subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'fps={}'.format(args.frame_rate), gif_path])

        logging.info("Deleting scratch directory...")
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)
        # # original code
        # if args.save_mp4 or args.save_gif:
        #     save_video(self.output_dir, "rgba.mp4", "rgba_%05d.jpg", fps=args.frame_rate)
        # if args.save_gif:
        #     video_path = os.path.join(self.output_dir, "rgba.mp4")
        #     gif_path = os.path.join(self.output_dir, "rgba.gif")
        #     subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'fps={}'.format(args.frame_rate), gif_path])
        if args.tar:
            # Tar the root output (which may contain multiple views)
            tar_path = os.path.join(self.base_dir, f"{self.video_id}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(self.output_dir, arcname=os.path.basename(self.output_dir))            
            # #original code
            # if os.path.exists(self.output_dir):
            #     shutil.rmtree(self.output_dir)



if __name__ == "__main__":
    intuit = Intuitive(args)
    try:
        start_time = time.time()
        intuit.render_a_scene()
        end_time = time.time()
        logging.info(f"Total time taken: {end_time - start_time} seconds")
    except TimeoutException:
        logging.error("Timed out")
        sys.exit(1)
