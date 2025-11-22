variable "cw_api_key" {
  type        = string
  description = "CoreWeave API key"
  sensitive   = true
}

variable "cluster_name" {
  type        = string
  description = "Logical name for this deployment"
}

variable "namespace" {
  type        = string
  description = "Namespace to create"
  default     = "trainer"
}

variable "trainer_image" {
  type        = string
  description = "Container image that runs train.py"
}

variable "gpu_flavor" {
  type        = string
  description = "GPU flavor (A100, H100, RTX_A6000, etc.)"
}

variable "replicas" {
  type        = number
  description = "Replica count"
  default     = 1
}
