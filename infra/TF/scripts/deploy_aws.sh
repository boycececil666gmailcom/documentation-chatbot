#!/bin/bash
set -e

# Export system AWS CLI v2 and Python Scripts paths for Git Bash environment
export PATH="$PATH:/c/Program Files/Amazon/AWSCLIV2:/c/Program Files/Amazon/AWSCLI/bin:/c/Users/boyce/AppData/Roaming/Python/Python314/Scripts"

log_step() {

    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> $1 [$(basename "$0")] $2\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load environment variables from .env file (Source of Truth)
if [ -f "${SCRIPT_DIR}/../.env" ]; then
    set -o allexport
    source "${SCRIPT_DIR}/../.env"
    set +o allexport
fi

# Derive ECR Registry endpoint URL from AWS account ID and region
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="${ECR_REPOSITORY:-enterprise_rag_engine}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# -------------------------------------------------------------------
# [1/3] STEP 1: Build Local Docker Container Images
# -------------------------------------------------------------------
log_step "[1/3]" "Building Local Docker Container Images"
"${SCRIPT_DIR}/build-image.sh"

# -------------------------------------------------------------------
# [2/3] STEP 2: Authenticate & Push Images to Existing AWS ECR Repo
# -------------------------------------------------------------------
log_step "[2/3]" "Authenticating & Pushing to AWS ECR (${ECR_REGISTRY}/${ECR_REPOSITORY})"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Tag images using existing single repository with service prefix tags
docker tag "theme-based-rag-backend:${IMAGE_TAG}" "${ECR_REGISTRY}/${ECR_REPOSITORY}:backend-${IMAGE_TAG}"

docker push "${ECR_REGISTRY}/${ECR_REPOSITORY}:backend-${IMAGE_TAG}"

# -------------------------------------------------------------------
# [3/3] STEP 3: Apply Terraform Deployment to Managed AWS EKS
# -------------------------------------------------------------------
log_step "[3/3]" "Updating AWS EKS Deployments via Terraform"

# Update local kubeconfig with AWS EKS authentication context
aws eks update-kubeconfig --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}" || true

cd "${SCRIPT_DIR}/../infra/terraform"

terraform init
terraform apply \
  -var="use_aws_eks=true" \
  -var="aws_region=${AWS_REGION}" \
  -var="eks_cluster_name=${EKS_CLUSTER_NAME}" \
  -var="backend_image=${ECR_REGISTRY}/${ECR_REPOSITORY}:backend-${IMAGE_TAG}" \
  -auto-approve

echo -e "\n\033[1;92m>>> [$(basename "$0")] AWS CI/CD Pipeline Completed Successfully!\033[0m\n"
