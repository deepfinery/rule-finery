terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = var.cluster_name
  cluster_version = "1.30"
  vpc_id          = var.vpc_id
  subnet_ids      = var.private_subnet_ids

  eks_managed_node_groups = {
    gpu = {
      desired_size = var.desired_size
      max_size     = var.max_size
      min_size     = var.min_size
      instance_types = [var.gpu_instance_type] # e.g., p4d.24xlarge (A100) or p5.48xlarge (H100)
      ami_type       = "AL2_x86_64_GPU"
      capacity_type  = "ON_DEMAND"
    }
  }
}

resource "local_file" "kubeconfig" {
  content  = module.eks.kubeconfig
  filename = "${path.module}/kubeconfig"
}
