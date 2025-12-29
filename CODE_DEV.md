
### 12.10
run_mixed.py as the main run file

python sim_data/visualize_npy.py --root dir

### 12.14
revise place_throw_objects()

add place_scene_default_static_objects()

add move_camera_back_static()

### 12.15

revise get_world_object_bounds() to consider object rotation

add kubasic option in  place_scene_default_static_objects()

revise plac_ground_objects() to sample from angular areas


### 12.26

revise place_throw_objects() to align with plac_collision_objects()

### 12.27
fix bug in place_radius_throw_objects()
add batch run script

### 12.28
randomize all parameters

1. fall
dynamic object numbers
place_static_objects_offset_stack:
    offset (0-0.1)
    stack(1-3)
    scale(1-2)

place_scene_default_static_objects:
    n_objects (1,4)
    center_region [-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]

2. throw
place_radius_throw_objects：
    min_radius 1
    max_radius 2
    height_range (0.3, 0.6)
    speed_range (2, 4)

place_center_static_stack_objects:
    num_stacks (2, 4)
    stack_height (1, 3)
    scale_range (1, 1.5)

place_scene_default_static_objects:
    n_objects (1,4)
    center_region [-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]


3.collision
place_collision_objects:
    min_radius 1
    max_radius 2
    speed_range: (3,5)

place_center_static_stack_objects:
    num_stacks (2, 4)
    stack_height (1, 3)
    scale_range (1, 1.5)

place_scene_default_static_objects:
    n_objects (1,4)
    center_region [-2.0, -2.0, 0.0], [2.0, 2.0 , 2.0]


4.combined
    fall + collision
    throw + collision

5. camera
movement
views

