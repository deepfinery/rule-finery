variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type        = string
  description = "GCP region"
}

variable "cluster_name" {
  type        = string
  description = "Cluster name"
}

variable "gpu_machine_type" {
  type        = string
  description = "GKE machine type (a2-highgpu-1g for A100, a3-highgpu-8g for H100)"
}

variable "gpu_type" {
  type        = string
  description = "Accelerator type (nvidia-tesla-a100, nvidia-h100-80gb, etc.)"
}

variable "gpu_count" {
  type        = number
  description = "GPU count per node"
  default     = 1
}

variable "node_count" {
  type        = number
  description = "Number of nodes in GPU pool"
  default     = 1
}
