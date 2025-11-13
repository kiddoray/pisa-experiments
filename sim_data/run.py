
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
    "--combined_influx_count",
    type=int,
    default=3,
    help="Number of offscreen objects to enter the scene during the second half of 'combined' scenario"
)
parser.add_argument(
    "--combined_influx_speed",
    type=float,
    default=6.0,
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

# parser.set_defaults(resolution="1024x1024", frame_rate=16, frame_end=32)
parser.set_defaults(resolution="1024x1024")
args = parser.parse_args()

VELOCITY_RANGE = [(-2., -2., 0.), (2., 2., 0.)]

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
        return scene, rng, output_dir, scratch_dir, renderer, simulator

    def load_scene(self, args):
        hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
        kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
        gso = kb.AssetSource.from_manifest(args.gso_assets)
        return kubasic, gso, hdri_source

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
                self.simulator._physics_client.resetBaseVelocity(obj_idx, tuple(v), (0.0, 0.0, 0.0))
            except Exception as e:
                logging.warning(f"Failed to apply velocity for object {getattr(o, 'name', 'unknown')}: {e}")

        # Small tick to commit
        try:
            self.simulator._physics_client.stepSimulation()
        except Exception:
            pass

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
            # World HDRI mode (recommended for Cycles + Shadow Catcher)
            # try:
            #     self.renderer._set_background_hdri(background_hdri.filename)
            # except Exception as e:
            #     logging.warning(f"Failed to set world background hdri: {e}")
            try:
                bpy.context.scene.render.engine = 'CYCLES'
                logging.info("Render engine set to CYCLES for World HDRI.")
            except Exception as e:
                logging.warning(f"Failed to switch render engine to Cycles: {e}")

        # --- Ground plane optional enhancement to improve near-field detail ---
        self.metadata["ground_mode"] = self.args.ground_mode
        if self.args.ground_mode != "none":
            try:
                # Try to use an existing KuBasic 'plane' asset first
                ground = self.kubasic.create(asset_id="plane", name="ground",
                                             friction=0.5, restitution=0.0,
                                             static=True, background=False)
                self.scene += ground
                # Scale & position (slightly above 0 to avoid z-fighting)
                s = float(self.args.ground_size)
                ground.scale = (s, s, 1.0)
                ground.position = (0.0, 0.0, 0.001)
                ground_bl = ground.linked_objects[self.renderer]
            except Exception as e:
                logging.warning(f"Failed to load KuBasic plane asset, falling back to Blender primitive: {e}")
                # Fallback: create Blender primitive plane directly
                bpy.ops.mesh.primitive_plane_add(size=self.args.ground_size, location=(0.0, 0.0, 0.0))
                ground_bl = bpy.context.active_object
                ground_bl.name = "ground_primitive"
                # Lift slightly
                ground_bl.location.z = 0.001

            try:
                if self.args.ground_mode == "shadow":
                    # Shadow catcher mode (visible only for shadows; background from HDRI)
                    try:
                        bpy.context.scene.render.engine = 'CYCLES'
                        logging.info("Render engine set to CYCLES for shadow catcher ground.")
                    except Exception as e:
                        logging.warning(f"Failed to switch render engine to Cycles for shadow catcher: {e}")
                    if hasattr(ground_bl, 'cycles'):
                        ground_bl.cycles.is_shadow_catcher = True
                    logging.info("Ground plane set as shadow catcher.")
                elif self.args.ground_mode == "textured":
                    # Apply a simple Principled BSDF material with optional color & subtle roughness
                    import os
                    hex_color = self.args.ground_color
                    def _parse_hex(h: str):
                        if isinstance(h, str):
                            h = h.strip().lstrip('#')
                            if len(h) == 6:
                                try:
                                    return tuple(int(h[i:i+2], 16)/255.0 for i in range(0,6,2))
                                except Exception:
                                    pass
                        return (0.5, 0.5, 0.5)
                    base_col = _parse_hex(hex_color)
                    # Ensure material exists
                    if len(ground_bl.data.materials) == 0:
                        mat = bpy.data.materials.new(name="GroundMaterial")
                        ground_bl.data.materials.append(mat)
                    else:
                        mat = ground_bl.data.materials[0]
                    # If requested, copy the dome's material onto the ground (note: mapping may differ on a plane)
                    if getattr(self.args, "ground_use_dome_material", False):
                        try:
                            if dome_blender and dome_blender.data.materials:
                                dome_mat = dome_blender.data.materials[0]
                                if dome_mat is not None:
                                    ground_mat = dome_mat.copy()
                                    if len(ground_bl.data.materials) == 0:
                                        ground_bl.data.materials.append(ground_mat)
                                    else:
                                        ground_bl.data.materials[0] = ground_mat
                                    logging.info("Ground assigned a copy of the dome material (ground_use_dome_material).")
                                    # Skip building a new principled stack
                                    output = None
                                    try:
                                        output = next(n for n in ground_mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL')
                                    except Exception:
                                        pass
                                    # End textured branch early
                                    return
                            else:
                                logging.warning("ground_use_dome_material is True but no dome is present (background_mode='world'). Skipping dome material copy.")
                        except Exception as e:
                            logging.warning(f"Failed to assign dome material to ground, will build principled stack instead: {e}")

                    mat.use_nodes = True
                    nt = mat.node_tree
                    for n in list(nt.nodes):
                        if n.type != 'OUTPUT_MATERIAL':
                            nt.nodes.remove(n)
                    # Create Principled BSDF
                    principled = nt.nodes.new('ShaderNodeBsdfPrincipled')
                    principled.location = (-200, 0)
                    principled.inputs['Roughness'].default_value = 0.6
                    principled.inputs['Specular'].default_value = 0.2
                    # Base Color: texture overrides color if provided
                    tex_color_node = None
                    if isinstance(self.args.ground_texture, str) and self.args.ground_texture.strip():
                        try:
                            tex_path = os.path.abspath(self.args.ground_texture)
                            img = bpy.data.images.load(tex_path)
                            img.colorspace_settings.name = 'sRGB'
                            tex_color_node = nt.nodes.new('ShaderNodeTexImage')
                            tex_color_node.image = img
                            tex_color_node.location = (-600, 200)
                            nt.links.new(tex_color_node.outputs['Color'], principled.inputs['Base Color'])
                            logging.info(f"Ground albedo texture loaded: {tex_path}")
                            self.metadata["ground_texture"] = tex_path
                        except Exception as e:
                            logging.warning(f"Failed to load ground_texture '{self.args.ground_texture}', fallback to color: {e}")
                    if tex_color_node is None:
                        principled.inputs['Base Color'].default_value = (*base_col, 1.0)

                    # Normal map: if provided, use it; else add subtle noise bump
                    normal_linked = False
                    if isinstance(self.args.ground_normal, str) and self.args.ground_normal.strip():
                        try:
                            nrm_path = os.path.abspath(self.args.ground_normal)
                            nimg = bpy.data.images.load(nrm_path)
                            nimg.colorspace_settings.name = 'Non-Color'
                            tex_normal = nt.nodes.new('ShaderNodeTexImage')
                            tex_normal.image = nimg
                            tex_normal.location = (-600, -200)
                            normal_map = nt.nodes.new('ShaderNodeNormalMap')
                            normal_map.location = (-400, -200)
                            nt.links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
                            nt.links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                            normal_linked = True
                            logging.info(f"Ground normal map loaded: {nrm_path}")
                            self.metadata["ground_normal"] = nrm_path
                        except Exception as e:
                            logging.warning(f"Failed to load ground_normal '{self.args.ground_normal}', fallback to noise bump: {e}")
                    if not normal_linked:
                        noise = nt.nodes.new('ShaderNodeTexNoise')
                        noise.location = (-600, -100)
                        noise.inputs['Scale'].default_value = 8.0
                        noise.inputs['Detail'].default_value = 2.0
                        bump = nt.nodes.new('ShaderNodeBump')
                        bump.location = (-400, -120)
                        nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
                        nt.links.new(bump.outputs['Normal'], principled.inputs['Normal'])

                    # Roughness map: if provided, use it; else keep default roughness
                    if isinstance(self.args.ground_roughness, str) and self.args.ground_roughness.strip():
                        try:
                            rgh_path = os.path.abspath(self.args.ground_roughness)
                            rimg = bpy.data.images.load(rgh_path)
                            rimg.colorspace_settings.name = 'Non-Color'
                            tex_rgh = nt.nodes.new('ShaderNodeTexImage')
                            tex_rgh.image = rimg
                            tex_rgh.location = (-600, 0)
                            nt.links.new(tex_rgh.outputs['Color'], principled.inputs['Roughness'])
                            logging.info(f"Ground roughness map loaded: {rgh_path}")
                            self.metadata["ground_roughness"] = rgh_path
                        except Exception as e:
                            logging.warning(f"Failed to load ground_roughness '{self.args.ground_roughness}', keep default: {e}")

                    # Displacement/height map: feed to Material Output Displacement (Cycles),
                    # and if not using Cycles and no normal map, also feed as bump to Principled.Normal
                    if isinstance(self.args.ground_displacement, str) and self.args.ground_displacement.strip():
                        try:
                            dsp_path = os.path.abspath(self.args.ground_displacement)
                            dimg = bpy.data.images.load(dsp_path)
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
                            # Connect to material output displacement (will only render in Cycles)
                            try:
                                output = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
                            except Exception:
                                output = None
                            if output is not None:
                                nt.links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
                                # Prefer BOTH so bump+true displacement can co-exist in Cycles
                                try:
                                    mat.cycles.displacement_method = 'BOTH'
                                except Exception:
                                    pass

                            # Eevee fallback: if no normal map set, reuse displacement as bump normal
                            try:
                                engine = bpy.context.scene.render.engine
                            except Exception:
                                engine = 'CYCLES'
                            if engine != 'CYCLES' and not normal_linked:
                                bump2 = nt.nodes.new('ShaderNodeBump')
                                bump2.location = (-200, -420)
                                bump2.inputs['Strength'].default_value = 1.0
                                nt.links.new(to_bw.outputs['Val'], bump2.inputs['Height'])
                                nt.links.new(bump2.outputs['Normal'], principled.inputs['Normal'])

                            logging.info(f"Ground displacement map loaded: {dsp_path}")
                            self.metadata["ground_displacement"] = dsp_path
                        except Exception as e:
                            logging.warning(f"Failed to load ground_displacement '{self.args.ground_displacement}', skipping: {e}")
                    # Output
                    output = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
                    nt.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                    logging.info(f"Ground plane textured with color {base_col}.")
            except Exception as e:
                logging.warning(f"Failed to configure ground plane material/mode: {e}")

        # # Ensure there is a physics ground collider at z=0 so objects don't fall forever
        # try:
        #     if getattr(self.args, "enable_physics_ground", True):
        #         self.ensure_physics_ground()
        # except Exception as e:
        #     logging.warning(f"Failed to ensure physics ground: {e}")
        
    def setup_camera(self):
        height = self.rng.uniform(0.4, 0.6)
        # tilt = random.uniform(-15, 15)
        position = (-2,0,height)
        self.scene.camera = kb.PerspectiveCamera(
            focal_length=self.args.focal_length, 
            sensor_width=self.args.sensor_width,
            position=position,
            euler=(radians(90), 0, radians(270))
        )
    
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
            side = int(self.rng.choice([-1, 1]))  # -1 from left (neg x), +1 from right
            while attempts < 20:
                obj = None
                try:
                    obj_id = self.rng.choice(self.active_object_split)
                    obj = self.gso.create(asset_id=obj_id)
                    self.scene.add(obj)

                    # Sample y within region; position x just beyond boundary with a small band
                    y = float(self.rng.uniform(ymin, ymax))
                    band = 0.02
                    x = (xmin - (margin + band)) if side < 0 else (xmax + (margin + band))

                    # Place on ground (bottom touches z=0)
                    z_offset = get_object_height_from_center(obj) + 1e-3
                    obj.position = (x, y, z_offset)

                    # Aim toward the scene origin in XY with fixed speed
                    to_center = np.array([-x, -y, 0.0], dtype=float)
                    n = float(np.linalg.norm(to_center[:2]))
                    if n < 1e-6:
                        vx, vy = (-side, 0.0)
                    else:
                        vx = float(to_center[0] / n)
                        vy = float(to_center[1] / n)
                    obj.velocity = (float(speed) * vx, float(speed) * vy, 0.0)

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
            obj.position = (dynamic_obj.position[0], dynamic_obj.position[1], z + offset +get_object_height_from_center(obj))
            _, max_corner = get_world_object_bounds(obj)
            z = max_corner[2]
            tower.append(obj)
            self.scene.add(obj)
            self.static_objects.append(obj)
        return tower

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
            return

        size = float(self.args.physics_ground_size) if self.args.physics_ground_size else float(self.args.ground_size) if hasattr(self.args, 'ground_size') else 20.0
        thickness = max(1e-3, float(self.args.physics_ground_thickness))

        try:
            ground_phys = kb.Cube(name="physics_ground",
                                  scale=(size, size, thickness),
                                  friction=0.8,
                                  restitution=0.0,
                                  static=True)
        except Exception:
            # Fallback if kb.Cube cannot be constructed from current kubric import style
            ground_phys = kb.core.Cube(name="physics_ground",
                                       scale=(size, size, thickness),
                                       friction=0.8,
                                       restitution=0.0,
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
                towers.append(self.place_static_objects_under_dynamic_obj(dynamic_obj))
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

        # For 'combined', run the simulation in two stages and spawn offscreen entrants at t=2s
        if self.args.scenario == "combined":
            # fps = getattr(self.args, "frame_rate", 16)
            # try:
            #     fps = int(fps)
            # except Exception:
            #     fps = 16
            # split = int(2 * fps)
            split = self.args.frame_end // 2
            total = self.args.frame_end
            # try:
            #     self.scene.frame_end = total
            # except Exception:
            #     pass

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

            # Re-apply velocities directly in the physics backend for late entrants
            try:
                self._apply_initial_velocities(self.late_dynamic_objects)
            except Exception as e:
                logging.warning(f"[combined] failed to apply velocities to late entrants: {e}")

            logging.info(f"[combined] Stage B simulation {split}..{total}")
            self.simulator.run(frame_start=split, frame_end=total + 1)

        self.metadata["camera"] = get_camera_metadata(self.scene.camera)
        
        for obj in self.dynamic_objects:
            self.metadata["dynamic_objects"].append(get_object_metadata(obj))
        for obj in self.static_objects:
            self.metadata["static_objects"].append(get_object_metadata(obj))
        
        self.setup_lights()
        if args.scene_save_path:
            save_state(self.renderer,f"{base}_init{ext}")

        if self.args.scenario != "combined":
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
            data_stack = self.renderer.render(return_layers=["rgba","segmentation","depth"]) 

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
