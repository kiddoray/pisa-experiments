# #!/usr/bin/env bash
# set -euo pipefail

# # Ensure texture physics lookup exists (build once if missing)
# MANIFEST=${MANIFEST:-/mnt/solo/kiddoray/Texture_haven.json}
# PHYS_LOOKUP="${MANIFEST%.*}_physics.json"
# if [[ ! -f "$PHYS_LOOKUP" ]]; then
#     echo "[prep] Building texture physics lookup -> $PHYS_LOOKUP"
#     python scripts/sim_data/build_texture_physics_lookup.py --manifest "$MANIFEST"
# fi

# # Ensure object (GSO) physics lookup exists
# GSO_MANIFEST=${GSO_MANIFEST:-gs://kubric-public/assets/GSO/GSO.json}
# GSO_PHYS_BASENAME="${GSO_MANIFEST##*/}"
# GSO_PHYS="${GSO_PHYS_BASENAME%.*}_physics.json"
# if [[ ! -f "$GSO_PHYS" ]]; then
#     echo "[prep] Building object physics lookup -> $GSO_PHYS from $GSO_MANIFEST"
#     python scripts/sim_data/build_object_physics_lookup.py --manifest "$GSO_MANIFEST" --out "$GSO_PHYS" --overwrite || true
# fi

# GPU Selection: Set which GPU to use (0, 1, 2, etc.)
# CUDA_VISIBLE_DEVICES controls which GPUs the process can see
# When CUDA_VISIBLE_DEVICES=1, the physical GPU 1 becomes logical GPU 0
# So we don't need KUBRIC_GPU_ID when using CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=1

KUBRIC_USE_GPU=1 python sim_data/run_mixed.py \
    --video_id="0105_throw_collide" \
    --output_dir=sim_output \
    --resolution=512x512 \
    --num_threads=8 \
    --save_mp4 \
    --save_gif \
    --objects_split train \
    --backgrounds_split train \
    --seed 0 \
    --frame_rate 16 \
    --frame_end 64 \
    --scenario="combined" \
    --stage1="throw" \
    --stage2="mixed" \
    --stage2_speed=6.0 \
    --stage2_force="none" \
    --stage2_force_duration=0.3 \
    --num_views 1 \
    --camera_motion static \
    --view_sampling="circle" \
    --ground_mode="textured" \
    --background_mode="dome" \
    --enable_physics_ground \
    --hdri_assets /mnt/solo/kiddoray/HDRI_haven.json \
    --gso_physics_lookup /mnt/solo/kiddoray/GSO_physics.json \
    --engine cycles \
    --disable_flow
    # --flow_render_mode split \
    # --disable_flow
    # --ground_texture_manifest "$MANIFEST"
    # --ground_texture /mnt/solo/kiddoray/Texture_haven/ceramic_roof_01/ceramic_roof_01_diff_4k.jpg \
    # --ground_normal /mnt/solo/kiddoray/Texture_haven/ceramic_roof_01/ceramic_roof_01_nor_gl_4k.exr \
    # --ground_roughness /mnt/solo/kiddoray/Texture_haven/ceramic_roof_01/ceramic_roof_01_rough_4k.jpg \
    # --ground_displacement /mnt/solo/kiddoray/Texture_haven/ceramic_roof_01/ceramic_roof_01_disp_4k.exr \