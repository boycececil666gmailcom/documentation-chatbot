#region PGVector Kubernetes Service & StatefulSet
resource "kubernetes_service" "pgvector_service" {
  metadata {
    name      = "pgvector-service"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
    labels = {
      app = "pgvector"
    }
  }

  spec {
    cluster_ip = "None"
    selector = {
      app = "pgvector"
    }

    port {
      name        = "postgres"
      port        = 5432
      target_port = 5432
    }
  }
}

resource "kubernetes_stateful_set" "pgvector" {
  wait_for_rollout = true

  metadata {
    name      = "pgvector"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
    labels = {
      app = "pgvector"
    }
  }

  spec {
    service_name = kubernetes_service.pgvector_service.metadata[0].name
    replicas     = 1

    selector {
      match_labels = {
        app = "pgvector"
      }
    }

    template {
      metadata {
        labels = {
          app = "pgvector"
        }
      }

      spec {
        container {
          name              = "pgvector"
          image             = var.pgvector_image
          image_pull_policy = "IfNotPresent"

          port {
            name           = "postgres"
            container_port = 5432
          }

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
            name  = "PGDATA"
            value = "/var/lib/postgresql/data/pgdata"
          }

          volume_mount {
            name       = "pgvector-data"
            mount_path = "/var/lib/postgresql/data"
          }

          resources {
            requests = {
              memory = "256Mi"
              cpu    = "100m"
            }
            limits = {
              memory = "1Gi"
              cpu    = "500m"
            }
          }
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "pgvector-data"
      }

      spec {
        access_modes = ["ReadWriteOnce"]

        resources {
          requests = {
            storage = "5Gi"
          }
        }
      }
    }
  }
}
#endregion
