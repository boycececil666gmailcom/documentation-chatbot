resource "kubernetes_service" "backend_service" {
  metadata {
    name      = "theme-based-rag-backend-service"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
    labels = {
      app = "theme-based-rag-backend"
    }
  }

  spec {
    type = "ClusterIP"

    selector = {
      app = "theme-based-rag-backend"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 8000
    }
  }
}

resource "kubernetes_deployment" "backend" {
  wait_for_rollout = true

  metadata {
    name = "theme-based-rag-backend"

    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
    labels = {
      app = "theme-based-rag-backend"
    }
  }

  spec {
    replicas = var.backend_replicas

    selector {
      match_labels = {
        app = "theme-based-rag-backend"
      }
    }

    template {
      metadata {
        labels = {
          app = "theme-based-rag-backend"
        }
      }

      spec {
        container {
          name              = "theme-based-rag-backend"
          image             = var.backend_image
          image_pull_policy = "Always"

          port {
            name           = "http"
            container_port = 8000
          }

          #region General Environment Configuration
          env {
            name  = "POSTGRES_HOST"
            value = var.postgres_host != "" ? var.postgres_host : kubernetes_service.pgvector_service.metadata[0].name
          }
          env {
            name  = "POSTGRES_PORT"
            value = tostring(var.postgres_port)
          }
          env {
            name  = "NEO4J_URI"
            value = var.neo4j_uri
          }
          env {
            name  = "GEMINI_MODEL"
            value = var.gemini_model
          }
          env {
            name  = "GEMINI_EMBED_MODEL"
            value = var.gemini_embed_model
          }
          env {
            name  = "GEMINI_TEMPERATURE"
            value = "0.0"
          }
          env {
            name  = "CHATBOT_THEME"
            value = var.chatbot_theme
          }
          env {
            name  = "LANGSMITH_TRACING"
            value = var.langsmith_tracing
          }
          env {
            name  = "LANGSMITH_PROJECT"
            value = var.langsmith_project
          }
          #endregion

          #region Secret Credentials
          env {
            name = "POSTGRES_DB"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.postgres_secrets.metadata[0].name
                key  = "postgres-db"
              }
            }
          }
          env {
            name = "POSTGRES_USER"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.postgres_secrets.metadata[0].name
                key  = "postgres-user"
              }
            }
          }
          env {
            name = "POSTGRES_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.postgres_secrets.metadata[0].name
                key  = "postgres-password"
              }
            }
          }
          env {
            name = "GEMINI_API_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.gemini_secrets.metadata[0].name
                key  = "gemini-api-key"
              }
            }
          }
          env {
            name = "NEO4J_USERNAME"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.neo4j_secrets.metadata[0].name
                key  = "neo4j-username"
              }
            }
          }
          env {
            name = "NEO4J_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.neo4j_secrets.metadata[0].name
                key  = "neo4j-password"
              }
            }
          }
          env {
            name = "LANGSMITH_API_KEY"
            value_from {
              secret_key_ref {
                name     = kubernetes_secret.langchain_secrets.metadata[0].name
                key      = "langchain-api-key"
                optional = true
              }
            }
          }
          #endregion


          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 10
          }

          resources {
            requests = {
              memory = "512Mi"
              cpu    = "200m"
            }
            limits = {
              memory = "2Gi"
              cpu    = "1000m"
            }
          }
        }
      }
    }
  }
}
