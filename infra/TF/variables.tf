variable "kubeconfig_path" {
  type        = string
  description = "Path to the kubeconfig file"
  default     = "~/.kube/config"
}

variable "kubeconfig_context" {
  type        = string
  description = "Kubernetes context to use"
  default     = ""
}

variable "aws_region" {
  type        = string
  description = "AWS Region for EKS cluster deployment"
  default     = "us-east-1"
}

variable "eks_cluster_name" {
  type        = string
  description = "AWS EKS cluster name"
  default     = "theme-based-rag-eks"
}

variable "use_aws_eks" {
  type        = bool
  description = "Set to true to authenticate directly to AWS EKS cluster via AWS Provider"
  default     = false
}

variable "namespace" {
  type        = string
  description = "Kubernetes namespace for Documentation Chatbot"
  default     = "documentation-chatbot"
}

variable "gemini_api_key" {
  type        = string
  description = "Google Gemini API Key"
  sensitive   = true
  default     = "dummy_key_for_testing"
}

variable "langsmith_api_key" {
  type        = string
  description = "LangSmith API Key for tracing"
  sensitive   = true
  default     = ""
}

variable "langsmith_tracing" {
  type        = string
  description = "Enable LangSmith tracing"
  default     = "true"
}

variable "langsmith_project" {
  type        = string
  description = "LangSmith project name"
  default     = "pr-virtual-cork-53"
}

variable "gemini_model" {
  type        = string
  description = "Gemini LLM Model for routing and synthesis"
  default     = "gemini-3.1-flash-lite"
}

variable "gemini_embed_model" {
  type        = string
  description = "Gemini Embeddings Model"
  default     = "gemini-embedding-001"
}

variable "chatbot_theme" {
  type        = string
  description = "Theme boundary for chatbot routing & safeguards"
  default     = "Fintech SaaS platform"
}

variable "backend_replicas" {
  type        = number
  description = "Number of backend deployment replicas"
  default     = 2
}

variable "backend_image" {
  type        = string
  description = "Docker image for theme-based RAG backend"
  default     = "theme-based-rag-backend:latest"
}

variable "qdrant_image" {
  type        = string
  description = "Docker image for Qdrant Vector DB"
  default     = "qdrant/qdrant:v1.11.0"
}

variable "neo4j_image" {
  type        = string
  description = "Docker image for Neo4j Graph DB"
  default     = "neo4j:5.20.0"
}

variable "neo4j_username" {
  type        = string
  description = "Neo4j database username"
  default     = "neo4j"
}

variable "neo4j_password" {
  type        = string
  description = "Neo4j database password"
  sensitive   = true
  default     = "neo4jpassword123"
}

variable "qdrant_url" {
  type        = string
  description = "Qdrant Vector DB Service URL or external endpoint (leave empty for internal K8s service)"
  default     = ""
}

variable "neo4j_uri" {
  type        = string
  description = "Neo4j Graph DB Service URI or external endpoint (leave empty for internal K8s service)"
  default     = ""
}
