terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.50"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.cluster_name
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = "${var.cluster_name}-dns"

  default_node_pool {
    name       = "system"
    node_count = 1
    vm_size    = "Standard_D4s_v5"
    type       = "VirtualMachineScaleSets"
    mode       = "System"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size # e.g., Standard_ND96amsr_A100_v4 or Standard_ND_H100-8
  node_count            = var.node_count
  max_pods              = 30
  mode                  = "User"
}

resource "local_file" "kubeconfig" {
  content  = azurerm_kubernetes_cluster.aks.kube_config_raw
  filename = "${path.module}/kubeconfig"
}
