variable "resource_group_name" {
  type        = string
  description = "Azure resource group name"
}

variable "location" {
  type        = string
  description = "Azure region"
}

variable "cluster_name" {
  type        = string
  description = "AKS cluster name"
}

variable "gpu_vm_size" {
  type        = string
  description = "GPU VM size (Standard_ND96amsr_A100_v4 for A100, Standard_ND_H100-8 for H100)"
}

variable "node_count" {
  type        = number
  description = "Node count for GPU pool"
  default     = 1
}
