terraform {
  required_version = ">= 1.5.0"
  required_providers {
    coreweave = {
      source  = "coreweave/coreweave"
      version = ">= 0.2.6"
    }
  }
}

provider "coreweave" {
  api_key = var.cw_api_key
}

resource "coreweave_namespace" "ns" {
  name = var.namespace
}

resource "coreweave_gpu_deployment" "trainer" {
  metadata {
    name      = "${var.cluster_name}-trainer"
    namespace = coreweave_namespace.ns.name
  }

  spec {
    replicas = var.replicas

    template {
      metadata {
        labels = {
          app = "trainer"
        }
      }

      spec {
        dns_policy = "ClusterFirst"

        containers {
          name  = "trainer"
          image = var.trainer_image
          args  = ["--help"] # replace at runtime via the API job manifest

          resources {
            requests = {
              cpu    = "4"
              memory = "32Gi"
              gpu    = var.gpu_flavor # e.g., A100 or H100
            }
            limits = {
              cpu    = "4"
              memory = "32Gi"
              gpu    = var.gpu_flavor
            }
          }
        }
      }
    }
  }
}

output "kubeconfig" {
  value     = coreweave_namespace.ns.kubeconfig
  sensitive = true
}

resource "local_file" "kubeconfig" {
  content  = coreweave_namespace.ns.kubeconfig
  filename = "${path.module}/kubeconfig"
}
