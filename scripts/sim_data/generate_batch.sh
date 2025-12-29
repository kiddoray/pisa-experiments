#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch Data Generation Script
# =============================================================================
# This script generates multiple samples in parallel on specified GPUs.
# Each GPU can run multiple processes simultaneously.
#
# Usage:
#   ./scripts/sim_data/generate_batch.sh [OPTIONS]
#
# Options:
#   --start_id <int>       Starting sample ID (default: 0)
#   --num_samples <int>    Number of samples to generate (default: 100)
#   --gpus <str>           Comma-separated GPU IDs to use (default: "0,1,2,3")
#   --per_gpu <int>        Number of parallel processes per GPU (default: 8)
#   --output_dir <str>     Output directory (default: sim_output)
#
# Example:
#   ./scripts/sim_data/generate_batch.sh --start_id 0 --num_samples 1000 --gpus "1,2" --per_gpu 10
# =============================================================================

# Default parameters
START_ID=0
NUM_SAMPLES=100
GPUS="0,1,2,3"
PER_GPU=8
OUTPUT_DIR="sim_output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start_id)
            START_ID="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --per_gpu)
            PER_GPU="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert GPUS string to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
TOTAL_WORKERS=$((NUM_GPUS * PER_GPU))

echo "=============================================="
echo "Batch Data Generation"
echo "=============================================="
echo "Start ID:        $START_ID"
echo "Num Samples:     $NUM_SAMPLES"
echo "GPUs:            $GPUS (${NUM_GPUS} GPUs)"
echo "Per GPU:         $PER_GPU processes"
echo "Total Workers:   $TOTAL_WORKERS"
echo "Output Dir:      $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Function to run a single sample
run_sample() {
    local sample_id=$1
    local gpu_id=$2
    local log_file="$OUTPUT_DIR/logs/sample_${sample_id}.log"
    
    # Generate unique video_id with timestamp
    local video_id="sample_$(printf '%06d' $sample_id)"
    
    # Use sample_id as seed for reproducibility
    local seed=$sample_id
    
    echo "[$(date '+%H:%M:%S')] Starting sample $sample_id on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id KUBRIC_USE_GPU=1 python sim_data/run_mixed.py \
        --video_id="$video_id" \
        --output_dir="$OUTPUT_DIR" \
        --resolution=512x512 \
        --num_threads=4 \
        --save_mp4 \
        --objects_split train \
        --backgrounds_split train \
        --seed "$seed" \
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
        --disable_flow \
        > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Sample $sample_id completed successfully"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Sample $sample_id FAILED (exit code: $status)"
    fi
    return $status
}

export -f run_sample
export OUTPUT_DIR

# Generate list of (sample_id, gpu_id) pairs
# Use GNU parallel or simple background jobs approach

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job scheduling..."
    
    # Create job list: sample_id,gpu_id
    job_file=$(mktemp)
    for ((i=0; i<NUM_SAMPLES; i++)); do
        sample_id=$((START_ID + i))
        # Round-robin GPU assignment
        gpu_idx=$((i % NUM_GPUS))
        gpu_id=${GPU_ARRAY[$gpu_idx]}
        echo "$sample_id,$gpu_id"
    done > "$job_file"
    
    # Run with GNU parallel
    # --jobs controls concurrency, --colsep splits input by comma
    cat "$job_file" | parallel --jobs "$TOTAL_WORKERS" --colsep ',' \
        "CUDA_VISIBLE_DEVICES={2} run_sample {1} {2}"
    
    rm "$job_file"
else
    echo "GNU parallel not found. Using background jobs with proper semaphore..."
    
    # Use a simpler and more reliable approach:
    # Maintain separate PID arrays for each GPU
    # This properly tracks which jobs are running on which GPU
    
    # Create associative arrays to track PIDs per GPU
    declare -A gpu_pids  # gpu_pids[$gpu] = space-separated list of PIDs
    for gpu in "${GPU_ARRAY[@]}"; do
        gpu_pids[$gpu]=""
    done
    
    # Function to count running jobs for a specific GPU
    count_running_jobs() {
        local gpu=$1
        local count=0
        local new_pids=""
        
        for pid in ${gpu_pids[$gpu]}; do
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
                new_pids="$new_pids $pid"
            fi
        done
        
        # Update the PID list (remove finished jobs)
        gpu_pids[$gpu]="$new_pids"
        echo $count
    }
    
    # Function to wait until a GPU has capacity
    wait_for_gpu_slot() {
        local gpu=$1
        while true; do
            local running=$(count_running_jobs "$gpu")
            if (( running < PER_GPU )); then
                break
            fi
            sleep 0.5
        done
    }
    
    echo "Starting batch generation with $NUM_GPUS GPUs, $PER_GPU jobs per GPU..."
    
    for ((i=0; i<NUM_SAMPLES; i++)); do
        sample_id=$((START_ID + i))
        
        # Round-robin GPU assignment
        gpu_idx=$((i % NUM_GPUS))
        selected_gpu=${GPU_ARRAY[$gpu_idx]}
        
        # Wait until this GPU has an available slot
        wait_for_gpu_slot "$selected_gpu"
        
        # Launch job in background
        run_sample "$sample_id" "$selected_gpu" &
        pid=$!
        
        # Track this PID for this GPU
        gpu_pids[$selected_gpu]="${gpu_pids[$selected_gpu]} $pid"
        
        echo "[$(date '+%H:%M:%S')] Queued sample $sample_id on GPU $selected_gpu (PID: $pid)"
        
        # Small delay to stagger launches
        sleep 0.2
    done
    
    # Wait for all remaining jobs to complete
    echo ""
    echo "All jobs queued. Waiting for completion..."
    
    for gpu in "${GPU_ARRAY[@]}"; do
        for pid in ${gpu_pids[$gpu]}; do
            wait "$pid" 2>/dev/null || true
        done
    done
fi

echo "=============================================="
echo "Batch generation complete!"
echo "Output directory: $OUTPUT_DIR"
echo "Logs: $OUTPUT_DIR/logs/"
echo "=============================================="

# Summary: count successes and failures
success_count=$(grep -l "completed successfully" "$OUTPUT_DIR/logs/"*.log 2>/dev/null | wc -l || echo 0)
total_logs=$(ls "$OUTPUT_DIR/logs/"*.log 2>/dev/null | wc -l || echo 0)
failed_count=$((total_logs - success_count))

echo "Results: $success_count succeeded, $failed_count failed out of $total_logs total"
