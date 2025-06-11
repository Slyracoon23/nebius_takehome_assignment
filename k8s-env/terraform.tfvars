# SSH config
ssh_user_name = "ubuntu"
ssh_public_key = {
  key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGpoCqO0aBrJrL66y1ngC50fmu7iKSe1uVzn1Y67Gjr1 earl.potters@gmail.com"
}

# K8s nodes - CORRECTED for eu-north1 and H100
cpu_nodes_count            = 2
gpu_nodes_count_per_group  = 2  # 2 nodes with 8 GPUs each = 16 H100s
gpu_node_groups            = 1
gpu_nodes_platform         = "gpu-h100-sxm"  # ✅ H100 as required by assignment
gpu_nodes_preset           = "8gpu-128vcpu-1600gb"
infiniband_fabric          = "fabric-3"      # ✅ Correct for eu-north1
gpu_nodes_driverfull_image = true
enable_k8s_node_group_sa   = true

# CPU nodes optimized for eu-north1
cpu_nodes_platform = "cpu-d3"
cpu_nodes_preset   = "16vcpu-64gb"

# Individual Network Disk Storage - NOT SHARED between nodes
# Each CPU node gets its own individual network disk
cpu_disk_type = "NETWORK_SSD"          # Network-attached SSD storage
cpu_disk_size = "128"                  # 128GB per CPU node (2 nodes × 128GB = 256GB total individual storage)

# Each GPU node gets its own individual network disk  
gpu_disk_type = "NETWORK_SSD"          # Network-attached SSD storage
gpu_disk_size = "1023"                 # 1TB per GPU node (2 nodes × 1TB = 2TB total individual storage)

# MIG configuration - disabled for full training
# mig_strategy = "single"
# mig_parted_config = "all-disabled"

# Observability
enable_prometheus = true
enable_loki       = false

# Shared Network Storage - SHARED across ALL nodes
# This creates a 2TB shared filesystem accessible by all 4 nodes simultaneously
enable_filestore     = true
filestore_disk_type  = "NETWORK_SSD"    # Network-attached SSD storage for shared filesystem
filestore_disk_size  = 2 * (1024 * 1024 * 1024 * 1024) # 2TB shared filesystem accessible by all nodes
filestore_block_size = 4096

# TOTAL NETWORK STORAGE SUMMARY:
# - Individual storage: 256GB (CPU nodes) + 2TB (GPU nodes) = ~2.25TB (isolated per node)
# - Shared storage: 2TB (accessible by all nodes)
# - Grand total: ~4.25TB network storage capacity

# KubeRay - disabled for this training
enable_kuberay = false