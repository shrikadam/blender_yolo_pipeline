import blenderproc as bproc
import numpy as np
import bpy
import os
import glob
import argparse

# =====================================================================
# 1. CONFIGURATION (Nominal Poses & Randomization Limits)
# =====================================================================

ASSEMBLY_CONFIG = {
    "nic": {
        "nominal_loc": np.array([-0.08, -0.02, -0.04]),
        "nominal_rot": np.array([0, 0, np.radians(90)]),
        "loc_limit": [-0.0215, 0.0234], # Meters
        "rot_limit": [-10, 10]          # Degrees
    },
    "sc": {
        "nominal_loc": np.array([0.05, -0.02, 0.05]),
        "nominal_rot": np.array([0, 0, 0]),
        "loc_limit": [-0.06, 0.055],    # Meters
        "rot_limit": [0, 0]             # No rotation jitter specified
    }
}

# =====================================================================
# 2. INITIALIZATION & SETUP FUNCTIONS
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic data with assembly domain randomization.")
    parser.add_argument("--meshes_dir", type=str, default="./meshes/", help="Path to directory containing all .glb files")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--update_freq", type=int, default=5, help="How often (in frames) to randomize the assembly pose")
    return parser.parse_args()

def setup_environment():
    """Sets up the camera intrinsics, ground plane, and shadowless lighting."""
    K = [
        [1235.83, 0.0, 576.0],
        [0.0, 1235.83, 512.0],
        [0.0, 0.0, 1.0]
    ]
    bproc.camera.set_intrinsics_from_K_matrix(K, 1152, 1024)

    ground = bproc.object.create_primitive('PLANE', scale=[5, 5, 1])
    mat_ground = bproc.material.create("ground_mat")
    mat_ground.set_principled_shader_value("Base Color", [0.95, 0.95, 0.95, 1.0])
    mat_ground.set_principled_shader_value("Roughness", 0.9)
    ground.replace_materials(mat_ground)

    # --- UPDATED: High-Ambient, Soft-Shadow Lighting ---
    # Format: ([Pitch, Roll, Yaw], Energy)
    light_params = [
        ([0, 0, 0], 1.2),        # Top looking down (Main fill)
        ([180, 0, 0], 0.8),      # Bottom looking up (Kills under-shadows)
        ([90, 0, 0], 0.6),       # Front looking back
        ([-90, 0, 0], 0.6),      # Back looking front
        ([0, 90, 0], 0.6),       # Right looking left
        ([0, -90, 0], 0.6)       # Left looking right
    ]
    
    for rotation, energy in light_params:
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_energy(4 * energy)
        light.set_rotation_euler(np.radians(rotation))
        
        # CRITICAL FIX: Soften the shadows to mimic Gazebo's ambient light
        # A higher angle (e.g., 20+ degrees) makes shadows extremely soft/blurry
        light.blender_obj.data.angle = np.radians(45.0)

# =====================================================================
# 3. MESH LOADING & ASSEMBLY FUNCTIONS
# =====================================================================

def load_and_weld_mesh(filepath, obj_name, category_id=None):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping.")
        return None

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=filepath)
    
    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_meshes: return None

    bpy.ops.object.select_all(action='DESELECT')
    for mesh in imported_meshes: mesh.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]
    
    if len(imported_meshes) > 1: bpy.ops.object.join()
        
    bproc_obj = bproc.types.MeshObject(bpy.context.active_object)
    bproc_obj.set_name(obj_name)
    if category_id is not None: bproc_obj.set_cp("category_id", category_id)
        
    return bproc_obj

def setup_base_components(meshes_dir):
    """Loads Base, NIC, and SC. Only the base is permanently anchored here."""
    base = load_and_weld_mesh(os.path.join(meshes_dir, "base.glb"), "Task_Base")
    if base:
        base.set_location([0, -0.02, 0.0])
        base.set_rotation_euler([0, 0, np.radians(90)])

    nic = load_and_weld_mesh(os.path.join(meshes_dir, "nic.glb"), "NIC_Target", category_id=1)
    sc = load_and_weld_mesh(os.path.join(meshes_dir, "sc.glb"), "SC_Target", category_id=2)
    
    return base, nic, sc

def randomize_assembly(nic, sc):
    """Applies mathematically bounded noise to the nominal poses."""
    if nic:
        cfg = ASSEMBLY_CONFIG["nic"]
        # Generate random offsets (applied to all 3 axes)
        loc_offset = np.random.uniform(cfg["loc_limit"][0], cfg["loc_limit"][1], size=3)
        loc_offset[1] = max(0, loc_offset[2]) # Ensure that Z is not negative, or the component would get buried under the base
        rot_offset = np.radians(np.random.uniform(cfg["rot_limit"][0], cfg["rot_limit"][1], size=3))
        
        nic.set_location(cfg["nominal_loc"] + loc_offset)
        nic.set_rotation_euler(cfg["nominal_rot"] + rot_offset)

    if sc:
        cfg = ASSEMBLY_CONFIG["sc"]
        loc_offset = np.random.uniform(cfg["loc_limit"][0], cfg["loc_limit"][1], size=3)
        loc_offset[1] = max(0, loc_offset[2]) # Ensure that Z is not negative, or the component would get buried under the base
        rot_offset = np.radians(np.random.uniform(cfg["rot_limit"][0], cfg["rot_limit"][1], size=3))
        
        sc.set_location(cfg["nominal_loc"] + loc_offset)
        sc.set_rotation_euler(cfg["nominal_rot"] + rot_offset)

def load_distractors(meshes_dir):
    distractor_objs = []
    fixed_files = ["base.glb", "nic.glb", "sc.glb"]
    glb_files = glob.glob(os.path.join(meshes_dir, "*.glb"))

    for filepath in glb_files:
        filename = os.path.basename(filepath).lower()
        if filename in fixed_files: continue
            
        obj = load_and_weld_mesh(filepath, filename.replace(".glb", ""))
        if obj: distractor_objs.append(obj)
            
    return distractor_objs

# =====================================================================
# 4. GENERATION LOOP & EXPORT
# =====================================================================

def generate_dataset(nic, sc, distractor_objs, args):
    print(f"Generating {args.num_images} images...")
    
    for i in range(args.num_images):
        # 1. Update the fixed assembly poses every N frames
        if i % args.update_freq == 0:
            randomize_assembly(nic, sc)

        # 2. Scatter the floating distractors every single frame
        for obj in distractor_objs:
            location = np.random.uniform([-0.15, -0.15, 0.1], [0.15, 0.15, 0.5])
            rotation = np.random.uniform([0, 0, 0], [np.pi*2, np.pi*2, np.pi*2])
            obj.set_location(location)
            obj.set_rotation_euler(rotation)

        # 3. Sample Camera view on a dome looking at the assembly every frame
        poi = [0, -0.02, 0.1]
        location = bproc.sampler.shell(center=poi, radius_min=0.3, radius_max=0.8,
                                       elevation_min=20, elevation_max=85)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world)

    print("Rendering RGB and Segmaps...")
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

    output_dir = "dataset"
    print(f"Writing COCO annotations to {output_dir}/coco_data...")
    bproc.writer.write_coco_annotations(
        os.path.join(output_dir, "coco_data"),
        instance_segmaps=seg_data["instance_segmaps"],
        instance_attribute_maps=seg_data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG"
    )

# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    args = parse_arguments()
    bproc.init()
    
    # 1. Tell Cycles to use the GPU
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    # 2. Access Blender's system preferences
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences

    # 3. Force it to use OptiX (Best performance for RTX A6000)
    cycles_prefs.compute_device_type = 'OPTIX'

    # 4. Refresh the list of available hardware
    cycles_prefs.get_devices()

    # 5. Iterate through the hardware and ONLY enable the GPU
    for device in cycles_prefs.devices:
        if device.type == 'CPU':
            device.use = False  # Turn off the CPU
            print(f"Disabled CPU: {device.name}")
        else:
            device.use = True   # Turn on the A6000
            print(f"Enabled GPU: {device.name}")
    
    setup_environment()
    base, nic, sc = setup_base_components(args.meshes_dir)
    distractor_objs = load_distractors(args.meshes_dir)
    
    generate_dataset(nic, sc, distractor_objs, args)
    print("Pipeline finished successfully!")