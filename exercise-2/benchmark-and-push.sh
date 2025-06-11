#!/bin/bash

# GPU Cluster Benchmark and Container Push Script
# Complete workflow: Launch SkyPilot job -> Wait for completion -> Build container -> Push to registry
# Usage: ./benchmark-and-push [OPTIONS]

set -e  # Exit on any error

# Configuration with defaults
CLUSTER_NAME=${CLUSTER_NAME:-"sky-llm-benchmark"}
REGISTRY_URL=${REGISTRY_URL:-""}
IMAGE_NAME="llm-benchmark-validated"
DOCKERFILE_PATH="./Dockerfile"
BUILD_CONTEXT="."
SKYPILOT_CONFIG="llm-distributed-benchmark.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "=============================================================="
    echo "  GPU Cluster Benchmark & Container Validation Pipeline"
    echo "=============================================================="
    echo -e "${NC}"
}

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Complete GPU cluster testing pipeline:"
    echo "1. Launch SkyPilot benchmark job"
    echo "2. Wait for job completion"
    echo "3. Build validated container image"
    echo "4. Push to container registry"
    echo ""
    echo "Options:"
    echo "  -c, --cluster NAME       SkyPilot cluster name (default: sky-llm-benchmark)"
    echo "  -r, --registry URL       Container registry URL (e.g., ghcr.io/org/repo)"
    echo "  -f, --config FILE        SkyPilot config file (default: llm-distributed-benchmark.yaml)"
    echo "  --skip-launch           Skip SkyPilot launch (use existing cluster)"
    echo "  --skip-build            Skip container build"
    echo "  --skip-push             Skip registry push"
    echo "  --logs                  Show recent job logs"
    echo "  --dry-run               Show what would be executed without running"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Environment variables:"
    echo "  CLUSTER_NAME            Same as --cluster"
    echo "  REGISTRY_URL            Same as --registry"
    echo "  GHCR_TOKEN              GitHub Container Registry token"
    echo ""
    echo "Examples:"
    echo "  $0 --registry ghcr.io/company/gpu-validator"
    echo "  $0 --cluster my-cluster --skip-launch"
    echo "  $0 --dry-run --registry ghcr.io/test/repo"
}

# Function to check if required tools are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v sky &> /dev/null; then
        missing_deps+=("SkyPilot (sky command)")
    fi
    
    if [ "$SKIP_BUILD" != true ] && ! command -v docker &> /dev/null; then
        missing_deps+=("Docker")
    fi
    
    if [ "$SKIP_BUILD" != true ] && ! docker info &> /dev/null 2>&1; then
        missing_deps+=("Docker daemon (not running)")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Function to launch SkyPilot job
launch_skypilot_job() {
    if [ "$SKIP_LAUNCH" = true ]; then
        print_info "Skipping SkyPilot launch as requested"
        return 0
    fi
    
    print_info "Launching SkyPilot benchmark job..."
    
    if [ ! -f "$SKYPILOT_CONFIG" ]; then
        print_error "SkyPilot config file not found: $SKYPILOT_CONFIG"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would execute: sky launch $SKYPILOT_CONFIG -c $CLUSTER_NAME"
        return 0
    fi
    
    sky launch "$SKYPILOT_CONFIG" -c "$CLUSTER_NAME"
    print_success "SkyPilot job launched successfully"
}

# Function to check SkyPilot job status
check_job_status() {
    print_info "Checking SkyPilot job status for cluster: $CLUSTER_NAME"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would check job status for cluster: $CLUSTER_NAME"
        return 0
    fi
    
    # Check if cluster exists
    if ! sky status "$CLUSTER_NAME" &> /dev/null; then
        print_error "Cluster '$CLUSTER_NAME' not found or inaccessible"
        print_info "Available clusters:"
        sky status
        exit 1
    fi
    
    # Get the latest job status
    JOB_STATUS=$(sky logs "$CLUSTER_NAME" --status 2>/dev/null | tail -1 || echo "UNKNOWN")
    
    print_info "Latest job status: $JOB_STATUS"
    
    if [[ "$JOB_STATUS" == *"SUCCEEDED"* ]]; then
        print_success "SkyPilot job completed successfully!"
        return 0
    elif [[ "$JOB_STATUS" == *"FAILED"* ]]; then
        print_error "SkyPilot job failed. Cannot proceed with container build."
        print_info "Check logs with: sky logs $CLUSTER_NAME"
        exit 1
    elif [[ "$JOB_STATUS" == *"RUNNING"* ]]; then
        print_warning "SkyPilot job is still running. Will wait for completion..."
        return 1
    else
        print_warning "Job status unclear: $JOB_STATUS"
        print_info "You can check manually with: sky logs $CLUSTER_NAME"
        if [ "$DRY_RUN" != true ]; then
            read -p "Do you want to proceed with the build anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        return 0
    fi
}

# Function to wait for job completion
wait_for_job_completion() {
    print_info "Waiting for SkyPilot job to complete..."
    local max_wait=3600  # 1 hour max wait
    local wait_time=0
    local check_interval=30
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would wait up to ${max_wait} seconds for job completion"
        return 0
    fi
    
    while [ $wait_time -lt $max_wait ]; do
        if check_job_status; then
            return 0
        fi
        
        print_info "Job still running. Waiting $check_interval seconds... (${wait_time}/${max_wait}s elapsed)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    print_error "Timeout waiting for job completion after ${max_wait} seconds"
    exit 1
}

# Function to show logs
show_recent_logs() {
    print_info "Recent job logs from cluster: $CLUSTER_NAME"
    echo "----------------------------------------"
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would show logs with: sky logs $CLUSTER_NAME --tail 20"
    else
        sky logs "$CLUSTER_NAME" --tail 20
    fi
    echo "----------------------------------------"
}

# Function to build Docker image
build_container_image() {
    if [ "$SKIP_BUILD" = true ]; then
        print_info "Skipping container build as requested"
        return 0
    fi
    
    print_info "Building validated container image..."
    
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        print_error "Dockerfile not found at: $DOCKERFILE_PATH"
        exit 1
    fi
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    VERSIONED_TAG="${IMAGE_NAME}:${timestamp}"
    LATEST_TAG="${IMAGE_NAME}:latest"
    
    print_info "Building image with tags: $VERSIONED_TAG, $LATEST_TAG"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would execute: docker build -t $VERSIONED_TAG -t $LATEST_TAG $BUILD_CONTEXT"
        return 0
    fi
    
    docker build -t "$VERSIONED_TAG" -t "$LATEST_TAG" "$BUILD_CONTEXT"
    
    print_success "Container image built successfully"
    print_info "Local tags created:"
    print_info "  - $VERSIONED_TAG"
    print_info "  - $LATEST_TAG"
}

# Function to push to registry
push_to_registry() {
    if [ "$SKIP_PUSH" = true ]; then
        print_info "Skipping registry push as requested"
        return 0
    fi
    
    if [ -z "$REGISTRY_URL" ]; then
        print_warning "REGISTRY_URL not set. Skipping registry push."
        print_info "To enable registry push, use --registry option or set REGISTRY_URL environment variable"
        print_info "Example: --registry ghcr.io/your-org/your-repo"
        return 0
    fi
    
    print_info "Pushing to registry: $REGISTRY_URL"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local registry_versioned="${REGISTRY_URL}:${timestamp}"
    local registry_latest="${REGISTRY_URL}:latest"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would push images:"
        print_info "  - $registry_versioned"
        print_info "  - $registry_latest"
        return 0
    fi
    
    # Tag for registry
    docker tag "$VERSIONED_TAG" "$registry_versioned"
    docker tag "$LATEST_TAG" "$registry_latest"
    
    # Push both tags
    print_info "Pushing versioned tag: $registry_versioned"
    docker push "$registry_versioned"
    
    print_info "Pushing latest tag: $registry_latest"
    docker push "$registry_latest"
    
    print_success "Successfully pushed to registry!"
    print_info "Images available at:"
    print_info "  - $registry_versioned"
    print_info "  - $registry_latest"
}

# Function to cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Pipeline failed. Check logs above for details."
        print_info "You can retry with: sky logs $CLUSTER_NAME"
    fi
}

# Main execution function
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cluster)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY_URL="$2"
                shift 2
                ;;
            -f|--config)
                SKYPILOT_CONFIG="$2"
                shift 2
                ;;
            --skip-launch)
                SKIP_LAUNCH=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --logs)
                SHOW_LOGS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Show configuration
    print_info "Configuration:"
    print_info "  Cluster: $CLUSTER_NAME"
    print_info "  Registry: ${REGISTRY_URL:-"Not set (will skip push)"}"
    print_info "  Config: $SKYPILOT_CONFIG"
    print_info "  Skip launch: ${SKIP_LAUNCH:-false}"
    print_info "  Skip build: ${SKIP_BUILD:-false}"
    print_info "  Skip push: ${SKIP_PUSH:-false}"
    print_info "  Dry run: ${DRY_RUN:-false}"
    echo
    
    # Execute pipeline
    check_dependencies
    
    if [ "$SHOW_LOGS" = true ]; then
        show_recent_logs
        echo
    fi
    
    # Step 1: Launch SkyPilot job
    launch_skypilot_job
    echo
    
    # Step 2: Wait for job completion
    if ! check_job_status; then
        wait_for_job_completion
    fi
    echo
    
    # Step 3: Build container image
    build_container_image
    echo
    
    # Step 4: Push to registry
    push_to_registry
    echo
    
    print_success "🎉 GPU Cluster Benchmark & Container Pipeline completed successfully!"
    
    if [ "$DRY_RUN" != true ]; then
        print_info "Summary:"
        print_info "  ✓ Cluster tested: $CLUSTER_NAME"
        if [ "$SKIP_BUILD" != true ]; then
            print_info "  ✓ Container built: $LATEST_TAG"
        fi
        if [ -n "$REGISTRY_URL" ] && [ "$SKIP_PUSH" != true ]; then
            print_info "  ✓ Pushed to registry: $REGISTRY_URL"
        fi
    else
        print_info "This was a dry run. Use without --dry-run to execute the pipeline."
    fi
}

# Run main function with all arguments
main "$@" 