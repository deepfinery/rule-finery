variable "cluster_name" {
  type        = string
  description = "EKS cluster name"
}

variable "region" {
  type        = string
  description = "AWS region"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID for the cluster"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for worker nodes"
}

variable "gpu_instance_type" {
  type        = string
  description = "GPU instance type (p4d.24xlarge for A100, p5.48xlarge for H100)"
}

variable "desired_size" {
  type        = number
  description = "Desired node count for GPU node group"
  default     = 1
}

variable "min_size" {
  type        = number
  description = "Minimum node count for GPU node group"
  default     = 0
}

variable "max_size" {
  type        = number
  description = "Maximum node count for GPU node group"
  default     = 3
}
