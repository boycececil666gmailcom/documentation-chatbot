output "namespace" {
  description = "The deployed Kubernetes namespace"
  value       = kubernetes_namespace.rag_namespace.metadata[0].name
}

output "backend_service_name" {
  description = "Name of the Chatbot Backend Kubernetes Service"
  value       = kubernetes_service.backend_service.metadata[0].name
}

output "qdrant_service_name" {
  description = "Name of the Qdrant Vector DB Service"
  value       = kubernetes_service.qdrant_service.metadata[0].name
}
