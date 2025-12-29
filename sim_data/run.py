
import os 
import sys
import pyquaternion as pyquat
import bpy
print("Rendering Engine:", bpy.context.scene.render.engine)
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

# os.environ["KUBRIC_USE_GPU"] = "0"
if "KUBRIC_USE_GPU" not in os.environ:
    os.environ["KUBRIC_USE_GPU"] = "1"

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
    "--combined_stage1",
    type=str,
    default="fall",
    help="Scenario to run during the first half of 'combined' scenario, fall or throw"
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
    help="Camera movement mode: static (original), fixed_random (random pose looking at origin), linear (linear position interpolation look-at origin), linear_pan (linear position + linear look-at)"
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
    default=7.0,
    help="Minimum radius for orbit mode sampling"
)
parser.add_argument(
    "--camera_orbit_radius_max",
    type=float,
    default=9.0,
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
    choices=["cycles", "eevee"],
    default=None,
    help="Render engine to use: 'cycles' or 'eevee'. If omitted, defaults to 'cycles'.",
)

# parser.set_defaults(resolution="1024x1024", frame_rate=16, frame_end=32)
parser.set_defaults(resolution="1024x1024")
args = parser.parse_args()

VELOCITY_RANGE = [(-2., -2., 0.), (2., 2., 0.)]
THROW_SPAWN_REGION = [(-1., -1., 1.), (1., 1., 1.5)]

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
        use_cycles = eng.lower().startswith("c") if isinstance(eng, str) and eng else True
        try:
            bpy.context.scene.render.engine = "CYCLES" if use_cycles else "BLENDER_EEVEE"
            logging.info(f"Render engine set to {'CYCLES' if use_cycles else 'BLENDER_EEVEE'} (engine={eng})")
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

    def _apply_initial_velocities(self, objs):
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

    def _accelerate_objects_with_force(self, objs, target_speed: float, dir_mode: str, duration_s: float, start_frame: int, use_impulse: bool = False) -> int:
        """Apply a short pulse of external force to accelerate objects realistically.

        - Uses PyBullet applyExternalForce in WORLD frame at the object's COM.
        - Force magnitude per object: F = m * (target_speed / duration_s)
        - Direction: from current position toward scene origin unless obj.velocity
          is already set (then use its XY direction).
        - Steps the simulator one frame at a time for the given duration.

                Extended modes:
                - Impulse mode (args.force_mode == 'impulse' or args.use_impulse=True): apply a single
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
                self._apply_initial_velocities(objs)
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

        # Impulse mode flags & parameters
        # force_mode = str(getattr(self.args, 'force_mode', 'force')).lower()
        # use_impulse = bool(getattr(self.args, 'use_impulse', False)) or force_mode.startswith('imp')

        try:
            impulse_multiplier = float(getattr(self.args, 'impulse_multiplier', 1.0))
        except Exception:
            impulse_multiplier = 1.0
        try:
            impulse_settle_frames = int(getattr(self.args, 'impulse_settle_frames', max(1, steps)))
        except Exception:
            impulse_settle_frames = max(1, steps)
        
        # If user explicitly sets impulse_mode, we override steps to 1 for application then settle
        if use_impulse:
            steps = 1  # only one application phase; settling handled separately

        client = getattr(self.simulator, '_physics_client', None)
        if client is None:
            logging.warning("Physics client unavailable; cannot apply external forces.")
            try:
                self._apply_initial_velocities(objs)
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
                speed = float(self.rng.uniform(0.6*target_speed, 1.5*target_speed))
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
                self._apply_initial_velocities(objs)
            except Exception:
                pass
            return 0

        current = int(start_frame)

        if use_impulse:
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

            # eng = getattr(self.args, "engine", None)
            # use_cycles = eng.lower().startswith("c") if isinstance(eng, str) and eng else True
            # if use_cycles:
            #     try:
            #         bpy.context.scene.render.engine = 'CYCLES'
            #         logging.info("Render engine set to CYCLES for World HDRI.")
            #     except Exception as e:
            #         logging.warning(f"Failed to switch render engine to Cycles: {e}")

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
        height = self.rng.uniform(0.6, 0.8)
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
            rmin = float(getattr(self.args, 'camera_orbit_radius_min', 7.0))
            rmax = float(getattr(self.args, 'camera_orbit_radius_max', 9.0))
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
    
    def place_throw_objects(self):
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
                    kb.randomness.position_sampler(THROW_SPAWN_REGION), 
                    kb.randomness.rotation_sampler(axis=None), 
                ]
                logging.info("Sampling dynamic object pose...")
                kb.resample_while(obj,samplers,self.simulator.check_overlap,rng=self.rng)

                obj.velocity = (self.rng.uniform(*VELOCITY_RANGE) -
                            [obj.position[0], obj.position[1], 0])
                obj.metadata["is_dynamic"] = True
                
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
    
    def place_ground_objects(self):
        """
        Place 2-6 objects on the ground with random horizontal velocities.
        
        Strategy:
        - Sample N objects near ground level (z ~ 0)
        - Adjust their Z position so they rest on the ground plane
        - Check visibility using projected footprint area
        - Assign random horizontal velocities (vx, vy) with vz=0
        - Set velocities directly via PyBullet after registration
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
                
                # Sample position near ground with random rotation
                kb.resample_while(
                    ground_obj,
                    [kb.randomness.position_sampler(self.cfg.dynamic_spawn_region), 
                     kb.randomness.rotation_sampler(axis=None)],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )
                
                # Adjust Z so that the object sits on ground plane (bottom touches z=0)
                z_offset = get_object_height_from_center(ground_obj) + 1e-3
                px, py, _ = ground_obj.position
                ground_obj.position = (px, py, z_offset)
                
                # Visibility check: project object's footprint to image
                min_corner, max_corner = get_world_object_bounds(ground_obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],  # top-left corner
                    [min_corner[0], min_corner[1], 0],              # bottom-left at ground
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) *
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])
                
                # Accept if visible enough (lower threshold for ground objects)
                if coverage > 0.025:
                    chosen = ground_obj

                    ground_obj.velocity = (self.rng.uniform(*VELOCITY_RANGE) - [ground_obj.position[0], ground_obj.position[1], 0])
                    # ground_obj.velocity = (self.rng.uniform(*VELOCITY_RANGE))
                    
                    # Sample an isotropic XY velocity: random angle + random speed
                    # theta = float(self.rng.uniform(0.0, 2*np.pi))
                    # speed = float(self.rng.uniform(0.8, 3.0))
                    # vx, vy = speed*np.cos(theta), speed*np.sin(theta)
                    # ground_obj.velocity = (vx, vy, 0.0)

                    print(f"The velocity of ground object {obj_id} is {ground_obj.velocity}")
                    ground_obj.metadata["is_dynamic"] = True
                    break
                
                # Reject and try again
                self.scene.remove(ground_obj)
                del ground_obj
                attempts += 1
            
            # Fallback: accept last candidate even if visibility is poor
            if chosen is None:
                obj_id = self.rng.choice(self.active_object_split)
                chosen = self.gso.create(asset_id=obj_id)
                self.scene.add(chosen)
                self._apply_object_physics(chosen, obj_id)
                kb.resample_while(
                    chosen,
                    [kb.randomness.position_sampler(self.cfg.dynamic_spawn_region),
                     kb.randomness.rotation_sampler(axis=None)],
                    self.simulator.check_overlap,
                    rng=self.rng,
                )
                z_offset = get_object_height_from_center(chosen) + 1e-3
                px, py, _ = chosen.position
                chosen.position = (px, py, z_offset)

                # chosen.velocity = (self.rng.uniform(*VELOCITY_RANGE))
                chosen.velocity = (self.rng.uniform(*VELOCITY_RANGE) - [chosen.position[0], chosen.position[1], 0])
                # theta = float(self.rng.uniform(0.0, 2*np.pi))
                # speed = float(self.rng.uniform(0.8, 3.0))
                # vx, vy = speed*np.cos(theta), speed*np.sin(theta)
                # chosen.velocity = (vx, vy, 0.0)

                print(f"The velocity of ground object {obj_id} is {chosen.velocity}")
                chosen.metadata["is_dynamic"] = True

            objs.append(chosen)
        
        # Ensure all objects are registered in PyBullet before setting velocities
        # for _ in range(2):
        #     self.simulator._physics_client.stepSimulation()
        
        # # Assign random horizontal velocities to all ground objects
        # velocities = {}
        # for obj in objs:
        #     velocity = self.rng.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0])
        #     velocities[obj] = velocity
        
        # # Apply velocities both via traitlet (for metadata) and directly in PyBullet
        # for obj, v in velocities.items():
        #     # Set property for bookkeeping/metadata
        #     try:
        #         obj.velocity = v
        #     except Exception:
        #         pass
            
        #     # Enforce in physics backend
        #     try:
        #         obj_idx = obj.linked_objects[self.simulator]
        #     except KeyError:
        #         self.simulator._physics_client.stepSimulation()
        #         obj_idx = obj.linked_objects[self.simulator]
        #     self.simulator._physics_client.resetBaseVelocity(obj_idx, v, (0.0, 0.0, 0.0))
        
        # # Small tick to commit state
        # self.simulator._physics_client.stepSimulation()
        
        logging.info(f"Placed {len(objs)} ground objects with random horizontal velocities")
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

                    # # Aim toward the scene origin in XY with fixed speed
                    # to_center = np.array([-x, -y, 0.0], dtype=float)
                    # n = float(np.linalg.norm(to_center[:2]))
                    # if n < 1e-6:
                    #     vx, vy = (-side, 0.0)
                    # else:
                    #     vx = float(to_center[0] / n)
                    #     vy = float(to_center[1] / n)
                    # obj.velocity = (float(speed) * vx, float(speed) * vy, 0.0)

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
        """Create a static inclined plane (slope) that objects can slide down.

        Parameters (read from self.args with defaults):
        - slope_angle_deg: tilt angle in degrees (default 20)
        - slope_length: along the rising direction (default 4.0 m)
        - slope_width: perpendicular width (default 2.0 m)
        - slope_thickness: slab thickness (default 0.05 m)
        - slope_axis: 'x' or 'y' direction of ascent (default 'x')
        - slope_position: center position tuple (default (0, 0, 0.05))
        - slope_friction, slope_restitution: dynamics (defaults 0.8, 0.0)
        - slope_visible: if False, hide from rendering (default True)
        Returns the created Kubric object or None on failure.
        """
        try:
            angle_deg = float(getattr(self.args, 'slope_angle_deg', 20.0))
        except Exception:
            angle_deg = 20.0
        angle_rad = float(np.deg2rad(angle_deg))
        L = float(getattr(self.args, 'slope_length', 4.0))
        W = float(getattr(self.args, 'slope_width', 2.0))
        T = float(getattr(self.args, 'slope_thickness', 0.05))
        axis = str(getattr(self.args, 'slope_axis', 'x')).lower()
        pos = getattr(self.args, 'slope_position', (0.0, 0.0, max(0.02, T*0.5)))
        mu = float(getattr(self.args, 'slope_friction', 0.8))
        e = float(getattr(self.args, 'slope_restitution', 0.0))
        visible = bool(getattr(self.args, 'slope_visible', True))

        try:
            slope = kb.Cube(name="slope",
                            scale=(L, W, T),
                            friction=mu,
                            restitution=e,
                            static=True)
        except Exception:
            # fallback for different kubric import styles
            slope = kb.core.Cube(name="slope",
                                 scale=(L, W, T),
                                 friction=mu,
                                 restitution=e,
                                 static=True)

        # Place and rotate the slope
        try:
            slope.position = tuple(pos)
        except Exception:
            pass
        try:
            axis_vec = (1.0, 0.0, 0.0) if axis.startswith('x') else (0.0, 1.0, 0.0)
            # Rotate so that the plane ascends along +axis (tilt around perpendicular axis)
            rot_axis = (0.0, 1.0, 0.0) if axis.startswith('x') else (1.0, 0.0, 0.0)
            q = pyquat.Quaternion(axis=rot_axis, angle=angle_rad)
            # Kubric objects typically accept (w, x, y, z)
            slope.quaternion = (q.w, q.x, q.y, q.z)
        except Exception:
            # Try Blender-level rotation as a fallback
            try:
                bl = slope.linked_objects[self.renderer]
                if axis.startswith('x'):
                    bl.rotation_euler = Euler((0.0, angle_rad, 0.0), "XYZ")
                else:
                    bl.rotation_euler = Euler((angle_rad, 0.0, 0.0), "XYZ")
            except Exception:
                pass

        try:
            self.scene.add(slope)
        except Exception:
            self.scene += slope

        # Hide from render if requested
        if not visible:
            try:
                bl = slope.linked_objects[self.renderer]
                bl.hide_render = True
                bl.hide_viewport = True
            except Exception:
                pass

        # Push dynamics overrides to physics backend if possible
        try:
            body_id = slope.linked_objects[self.simulator]
            self.simulator._physics_client.changeDynamics(int(body_id), -1, lateralFriction=mu, restitution=e)
        except Exception:
            pass

        # Store for potential later use
        try:
            self._slope_obj = slope
            self._slope_params = {
                'angle_rad': angle_rad, 'L': L, 'W': W, 'T': T, 'axis': axis,
                'position': tuple(getattr(slope, 'position', pos))
            }
        except Exception:
            pass

        logging.info(f"Added slope: angle={angle_deg}deg, size=({L},{W},{T}), axis={axis}, mu={mu}")
        return slope

    def place_objects_on_slope(self, slope, count: int | None = None, margin: float = 0.1):
        """Place a number of dynamic objects on the upper part of the slope so they slide down.

        - Computes an approximate surface height using the slope params.
        - Places objects slightly above the slope top face to avoid initial penetration.
        - Applies object physics overrides.
        Returns the list of placed objects.
        """
        if slope is None:
            return []
        params = getattr(self, '_slope_params', None)
        if not params:
            return []
        angle = float(params['angle_rad'])
        L = float(params['L'])
        W = float(params['W'])
        T = float(params['T'])
        axis = str(params['axis'])
        cx, cy, cz = params['position']
        n = int(count if count is not None else max(1, getattr(self, 'num_dynamic_objects', 3)))

        objs = []
        # x spans [-L/2, L/2] around center; choose upper half so they have room to slide down
        x_min, x_max = -0.5 * L + margin, 0.5 * L - margin
        y_min, y_max = -0.5 * W + margin, 0.5 * W - margin

        for _ in range(n):
            attempts = 0
            placed = None
            while attempts < 12:
                obj_id = self.rng.choice(self.active_object_split)
                obj = self.gso.create(asset_id=obj_id)
                self.scene.add(obj)
                self._apply_object_physics(obj, obj_id)

                # Sample near the high end of the slope (positive along ascent axis)
                if axis.startswith('x'):
                    x_rel = self.rng.uniform(0.0, x_max)
                    y_rel = self.rng.uniform(y_min, y_max)
                    # Height of the center plane at x_rel (approx): z = cz + x_rel * np.sin(angle)
                    z_surface = cz + x_rel * np.sin(angle)
                    px, py = cx + x_rel, cy + y_rel
                else:
                    x_rel = self.rng.uniform(x_min, x_max)
                    y_rel = self.rng.uniform(0.0, y_max)
                    z_surface = cz + y_rel * np.sin(angle)
                    px, py = cx + x_rel, cy + y_rel

                # Place object slightly above surface plus half-height offset
                z_offset = get_object_height_from_center(obj) + 2e-3
                pz = z_surface + z_offset + 0.5 * T
                obj.position = (px, py, pz)

                # Quick visibility/overlap check similar to other placements
                try:
                    kb.resample_while(
                        obj,
                        [kb.randomness.rotation_sampler(axis=None)],
                        self.simulator.check_overlap,
                        rng=self.rng,
                    )
                except Exception:
                    pass

                # Minimal visibility requirement
                min_corner, max_corner = get_world_object_bounds(obj)
                drop_bounds = np.array([
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], min_corner[1], 0],
                ])
                projected_points = project_points_to_image(drop_bounds, self.scene)
                area = abs((projected_points[0][0] - projected_points[1][0]) *
                           (projected_points[0][1] - projected_points[1][1]))
                coverage = area / (self.scene.resolution[0] * self.scene.resolution[1])
                if coverage > 0.01:
                    try:
                        obj.metadata["is_dynamic"] = True
                    except Exception:
                        pass
                    placed = obj
                    break

                # Reject and retry
                try:
                    self.scene.remove(obj)
                except Exception:
                    pass
                del obj
                attempts += 1

            if placed is None:
                # Keep last attempt even if not visible enough, position above high end
                obj_id = self.rng.choice(self.active_object_split)
                placed = self.gso.create(asset_id=obj_id)
                self.scene.add(placed)
                self._apply_object_physics(placed, obj_id)
                if axis.startswith('x'):
                    x_rel = 0.5 * x_max
                    y_rel = 0.0
                    z_surface = cz + x_rel * np.sin(angle)
                    px, py = cx + x_rel, cy + y_rel
                else:
                    x_rel = 0.0
                    y_rel = 0.5 * y_max
                    z_surface = cz + y_rel * np.sin(angle)
                    px, py = cx + x_rel, cy + y_rel
                z_offset = get_object_height_from_center(placed) + 2e-3
                pz = z_surface + z_offset + 0.5 * T
                placed.position = (px, py, pz)
                try:
                    placed.metadata["is_dynamic"] = True
                except Exception:
                    pass

            objs.append(placed)

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
        max_stack_height: int = 3,
        offset_radius: float = 0.6,
        stack_size_mode: str = "random",
        force_min_stack: int = 1,
        force_max_stack: int | None = None,
        radial_mode: str = "uniform",
        avoid_center: bool = True,
        min_separation: float = 0.15,
        scale_range: tuple[float, float] = (2.0, 2.5),
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
                except Exception:
                    pass
                # scale = self.rng.uniform(1.0, 2.0)
                # obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
                # obj.metadata["scale"] = scale

                self.scene.add(obj)
                self._apply_object_physics(obj, obj_id)

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

    def place_center_static_objects(self, n_objects: int | None = None, center_region: list | None = None):
        """
        Place a few static objects near the scene center within a small spawn region.
        Objects are resampled until no overlap and minimal visibility are satisfied.

        n_objects: number of objects to place (defaults to 2-5)
        center_region: [[xmin, ymin, zmin], [xmax, ymax, zmax]] in meters. If None,
                        a small box around the center is derived from dynamic_spawn_region.
        """
        if n_objects is None:
            n_objects = int(self.rng.randint(2, 6))

        # Derive a compact central region if not provided
        if center_region is None:
            dsr = self.cfg.dynamic_spawn_region
            (xmin, ymin, zmin), (xmax, ymax, zmax) = dsr
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            # shrink to a small box (30% of full width/height), near ground
            fx = 0.30 * (xmax - xmin)
            fy = 0.30 * (ymax - ymin)
            center_region = [
                [cx - fx, cy - fy, max(0.02, zmin * 0.0 + 0.02)],
                [cx + fx, cy + fy, 0.20],
            ]

        placed = []
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
    
    def move_camera_back(self):
        while not all(obj_drop_visible_from_camera(obj,self.scene) for obj in self.dynamic_objects):
            logging.info("Moving camera back...")
            position = (
                self.scene.camera.position[0] - abs(self.rng.normal(0, 0.05)), 
                self.scene.camera.position[1], 
                self.scene.camera.position[2]
            )
            self.scene.camera.position = position

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

        size = float(self.args.physics_ground_size) if self.args.physics_ground_size else float(self.args.ground_size) if hasattr(self.args, 'ground_size') else 20.0
        thickness = max(1e-3, float(self.args.physics_ground_thickness))
        mu = float(getattr(self, "_ground_friction_override", 0.8))
        e = float(getattr(self, "_ground_restitution_override", 0.0))

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
        # self.dynamic_objects = self.place_dynamic_objects()
        self.static_objects = []
        self.metadata["static_objects"] = []
        self.metadata["dynamic_objects"] = []

        # combined scenario
        if self.args.scenario == "combined":

            

            # # add center static obstacles
            # self.place_center_static_objects()

            # add dynamic objects on towers
            towers = []
            self.dynamic_objects = self.place_dynamic_objects()
            for dynamic_obj in self.dynamic_objects:
                # towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
                towers.append(self.place_static_objects_offset_stack(dynamic_obj, num_stacks=1, offset_radius=0.2))
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
            
            self.thrown_objects = self.place_throw_objects()
            for i, obj in enumerate(self.thrown_objects):
                obj = copy.deepcopy(obj)
                self.thrown_objects[i] = obj
                self.scene.add(obj) # Workaround
            self.dynamic_objects.extend(self.thrown_objects)
            logging.info(f"Placed {len(self.thrown_objects)} thrown objects for combined scenario.")

        # collision scenario
        elif self.args.scenario == "collision":

            # add center static obstacles
            self.place_center_static_objects()
            self.simulator.run(frame_start=-200, frame_end=0)

            # add dynamic objects
            self.dynamic_objects = self.place_ground_objects()

            base, ext = os.path.splitext(args.scene_save_path)
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_collision_setup{ext}")
            if args.scene_save_path:
                save_state(self.renderer, f"{base}_collision_settled{ext}")

            # Re-apply intended initial velocities AFTER preroll so motion starts at frame 0
            # self._apply_initial_velocities(self.dynamic_objects)
        
        
        # Slope scenario: add slope and let objects slide down
        elif self.args.scenario == 'slope':
            slope = self.add_slope()
            self.static_objects = []
            self.dynamic_objects = self.place_objects_on_slope(slope, count=self.num_dynamic_objects)
            # Small preroll to settle onto slope before rendering window
            try:
                self.simulator.run(frame_start=-60, frame_end=0)
            except Exception:
                pass

        else:
            towers = []
            self.dynamic_objects = self.place_dynamic_objects()
            for dynamic_obj in self.dynamic_objects:
                towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
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
                    
        
        self.move_camera_back()

        # For 'combined', run the simulation in two stages and spawn offscreen entrants at t=split
        if self.args.scenario == "combined":
            split = self.args.frame_end // 2
            total = self.args.frame_end

            logging.info(f"[combined] Stage A simulation 0..{split}")
            self.simulator.run(frame_start=0, frame_end=split)

            # Spawn offscreen entrants now so they appear only in the second half
            self.late_dynamic_objects = self.place_offscreen_influx(
                count=self.args.combined_influx_count,
                speed=float(self.args.combined_influx_speed),
                margin=float(self.args.combined_margin),
            )
            self.dynamic_objects.extend(self.late_dynamic_objects)

            # Important: ensure newly added objects are registered in PyBullet before
            # applying velocities; then explicitly (re-)apply their velocities so they move.
            try:
                # Zero-advance run to force registration at the current frame index
                self.simulator.run(frame_start=split, frame_end=split)
            except Exception as e:
                logging.warning(f"[combined] registration tick failed: {e}")

            # Option 1 (preferred): accelerate using an external force pulse for realism
            force_duration = float(getattr(self.args, 'combined_force_duration', 0.2))
            try:
                stepped = self._accelerate_objects_with_force(
                    self.late_dynamic_objects,
                    target_speed=float(self.args.combined_influx_speed),
                    dir_mode="mixed",
                    duration_s=force_duration,
                    start_frame=split,
                    use_impulse=False,
                )
            except Exception as e:
                logging.warning(f"[combined] external-force acceleration failed: {e}")
                stepped = 0

            # Fallback (if force pulse not applied): directly set base velocities
            if stepped <= 0:
                try:
                    self._apply_initial_velocities(self.late_dynamic_objects)
                except Exception as e:
                    logging.warning(f"[combined] failed to apply velocities to late entrants: {e}")

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

            # Render
            # Also request optical flow from Kubric's Blender renderer
            # Supported layers include "forward_flow" and "backward_flow"
            data_stack = self.renderer.render(return_layers=[
                "rgba", "segmentation", "depth", "forward_flow", "backward_flow"
            ]) 

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
                if "forward_flow" in data_stack:
                    try:
                        if i < data_stack["forward_flow"].shape[0]:
                            np.save(os.path.join(view_dir, f"forward_flow_{i:05d}.npy"), data_stack["forward_flow"][i])
                    except Exception as e:
                        logging.warning(f"Failed to save forward_flow for frame {i}: {e}")
                if "backward_flow" in data_stack:
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
        intuit.render_a_scene()
    except TimeoutException:
        logging.error("Timed out")
        sys.exit(1)
