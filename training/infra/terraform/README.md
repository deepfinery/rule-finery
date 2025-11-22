# Terraform scaffolding for GPU Kubernetes clusters

This directory sketches minimal Terraform examples to provision GPU-ready Kubernetes clusters on major clouds. Each example:
- Creates a managed control plane (EKS, AKS, GKE) or uses CoreWeave’s API.
- Adds a GPU node pool with A100/H100-capable instance types.
- Exposes a kubeconfig file you can point the training API to via `K8S_KUBECONFIG` (and optionally `K8S_CONTEXT`).

> These are intentionally slim and omit org-specific IAM/networking. Add your own VPC/Subnet/security group settings and service accounts as needed.

## Usage pattern
```bash
cd training/infra/terraform/<provider>
terraform init
terraform apply -var 'project_id=my-project' -var 'region=us-central1' # adjust per provider
# After apply, set:
export K8S_KUBECONFIG=$(pwd)/kubeconfig
export K8S_CONTEXT=<optional context name>  # only if multiple contexts in the file
export K8S_TRAINING_IMAGE=<your trainer image with train.py entrypoint>
export K8S_APPLY=1
cd ../../training-api && npm start
```

## Provider-specific notes

### EKS (AWS)
- File: `eks/main.tf`
- Highlights: uses the official `terraform-aws-modules/eks/aws` module, provisions a GPU node group (A100/H100 via `p4d.24xlarge`, `p5.48xlarge`), writes kubeconfig locally.
- Variables you likely need: `cluster_name`, `region`, `vpc_id`, `private_subnet_ids`, `gpu_instance_type` (e.g., `p4d.24xlarge`), `desired_size`.

### GKE (Google Cloud)
- File: `gke/main.tf`
- Highlights: uses `google_container_cluster` with a node pool running `a2-highgpu-1g` (A100) or `a3-highgpu-8g` (H100). Generates kubeconfig with `gcloud container clusters get-credentials`.
- Variables: `project_id`, `region`, `cluster_name`, `gpu_machine_type` (e.g., `a2-highgpu-1g`), `min_nodes`, `max_nodes`.

### AKS (Azure)
- File: `aks/main.tf`
- Highlights: uses `azurerm_kubernetes_cluster` with a user node pool using `Standard_ND96amsr_A100_v4` (A100) or `Standard_ND_H100-8` (H100). Exports kubeconfig to `kubeconfig`.
- Variables: `resource_group_name`, `location`, `cluster_name`, `gpu_vm_size`, `node_count`.

### CoreWeave (CKS)
- File: `coreweave/main.tf`
- Highlights: uses CoreWeave’s Terraform provider to create a namespace and GPU-enabled deployment; outputs kubeconfig via the provider token. Node flavors vary; use `RTX_A6000`, `A100`, or `H100` depending on availability.
- Variables: `cw_api_key`, `cluster_name`, `namespace`, `gpu_flavor`, `replicas`.

## Switching clusters for the API
Set `K8S_KUBECONFIG` to the file emitted by Terraform (or `~/.kube/config`) and optionally `K8S_CONTEXT` if multiple contexts exist. The API’s Kubernetes backend will use these when invoking `kubectl apply -f -`.
