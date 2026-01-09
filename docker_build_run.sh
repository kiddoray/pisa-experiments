#!/bin/bash
# Build and run PISA Docker image
# Usage: ./docker_build_run.sh [build|run|shell|push]

set -e

# Configuration
IMAGE_NAME="kubric-physics"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Data directories on host (modify these to match your setup)
# This directory contains HDRI_haven.json, GSO_physics.json, textures, etc.
HOST_DATA_DIR="/mnt/solo/kiddoray"

# Registry (optional - modify for your registry)
REGISTRY=""  # e.g., "docker.io/yourusername" or "ghcr.io/yourusername"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build the Docker image
build() {
    print_info "Building Docker image: ${FULL_IMAGE}"
    
    cd "${SCRIPT_DIR}"
    
    docker build \
        --progress=plain \
        -t "${FULL_IMAGE}" \
        -f Dockerfile \
        .
    
    print_info "Build complete: ${FULL_IMAGE}"
}

# Run the container interactively
run() {
    print_info "Running container: ${FULL_IMAGE}"
    
    # Check if NVIDIA runtime is available
    if docker info 2>/dev/null | grep -q "nvidia"; then
        GPU_FLAGS="--runtime=nvidia"
        print_info "NVIDIA GPU support detected"
    else
        GPU_FLAGS=""
        print_warn "No NVIDIA GPU support detected, running without GPU"
    fi
    
    docker run -it --rm \
        ${GPU_FLAGS} \
        --shm-size=16g \
        -v "${SCRIPT_DIR}/sim_output:/workspace/pisa-experiments/sim_output" \
        -v "${SCRIPT_DIR}/data_new:/workspace/pisa-experiments/data_new" \
        -v "${HOST_DATA_DIR}:${HOST_DATA_DIR}:ro" \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
        -e XDG_RUNTIME_DIR=/tmp/runtime-root \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        "${FULL_IMAGE}" \
        "$@"
}

# Run with specific GPU
run_gpu() {
    GPU_ID="${1:-0}"
    shift || true
    
    print_info "Running container with GPU ${GPU_ID}: ${FULL_IMAGE}"
    
    docker run -it --rm \
        --runtime=nvidia \
        --shm-size=16g \
        -v "${SCRIPT_DIR}/sim_output:/workspace/pisa-experiments/sim_output" \
        -v "${SCRIPT_DIR}/data_new:/workspace/pisa-experiments/data_new" \
        -v "${HOST_DATA_DIR}:${HOST_DATA_DIR}:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e __EGL_DEVICE_ID=0 \
        -e NVIDIA_VISIBLE_DEVICES=${GPU_ID} \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
        -e XDG_RUNTIME_DIR=/tmp/runtime-root \
        "${FULL_IMAGE}" \
        "$@"
}

# Run in Development mode (Mounts current directory to container)
# Usage: ./docker_build_run.sh dev [command]
# Example: ./docker_build_run.sh dev python sim_data/run_mixed.py
dev() {
    print_info "Running container in DEV mode (Host code mounted): ${FULL_IMAGE}"
    
    # Check if NVIDIA runtime is available
    if docker info 2>/dev/null | grep -q "nvidia"; then
        GPU_FLAGS="--runtime=nvidia"
    else
        GPU_FLAGS=""
        print_warn "No NVIDIA GPU support detected"
    fi
    
    docker run -it --rm \
        ${GPU_FLAGS} \
        --shm-size=16g \
        -v "${SCRIPT_DIR}:/workspace/pisa-experiments" \
        -v "${HOST_DATA_DIR}:${HOST_DATA_DIR}:ro" \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
        -e XDG_RUNTIME_DIR=/tmp/runtime-root \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        "${FULL_IMAGE}" \
        "$@"
}

# Open a shell in the container
shell() {
    run /bin/bash
}

# Run simulation inside container
simulate() {
    SCENARIO="${1:-collision}"
    SEED="${2:-42}"
    
    print_info "Running simulation: scenario=${SCENARIO}, seed=${SEED}"
    
    run bash -c "cd /workspace/pisa-experiments && bash scripts/sim_data/generate_single_sample.sh"
}

# Push to registry
push() {
    if [ -z "${REGISTRY}" ]; then
        print_error "Registry not configured. Edit REGISTRY variable in this script."
        exit 1
    fi
    
    REMOTE_IMAGE="${REGISTRY}/${FULL_IMAGE}"
    
    print_info "Tagging image as ${REMOTE_IMAGE}"
    docker tag "${FULL_IMAGE}" "${REMOTE_IMAGE}"
    
    print_info "Pushing to registry..."
    docker push "${REMOTE_IMAGE}"
    
    print_info "Push complete: ${REMOTE_IMAGE}"
}

# Show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Run the container interactively"
    echo "  run_gpu N   Run with specific GPU (N = GPU ID)"
    echo "  shell       Open a bash shell in the container"
    echo "  simulate    Run a simulation"
    echo "  push        Push image to registry"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the image"
    echo "  $0 shell                    # Open interactive shell"
    echo "  $0 run_gpu 0                # Run with GPU 0"
    echo "  $0 run python sim_data/run_mixed.py --help"
}

# Main
case "${1:-help}" in
    build)
        build
        ;;
    run)
        shift
        run "$@"
        ;;
    run_gpu)
        shift
        run_gpu "$@"
        ;;
    shell)
        shell
        ;;
    simulate)
        shift
        simulate "$@"
        ;;
    push)
        push
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
