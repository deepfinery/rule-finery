terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  release_channel {
    channel = "REGULAR"
  }

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "gpu" {
  name       = "gpu-pool"
  cluster    = google_container_cluster.primary.name
  location   = var.region
  node_count = var.node_count

  node_config {
    machine_type = var.gpu_machine_type # e.g., a2-highgpu-1g (A100) or a3-highgpu-8g (H100)
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.gpu_type # e.g., nvidia-tesla-a100 or nvidia-h100-80gb
      count = var.gpu_count
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    labels = {
      gpu = "true"
    }
  }
}

resource "null_resource" "kubeconfig" {
  provisioner "local-exec" {
    command = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id} --kubeconfig ${path.module}/kubeconfig"
  }
  depends_on = [google_container_node_pool.gpu]
}
