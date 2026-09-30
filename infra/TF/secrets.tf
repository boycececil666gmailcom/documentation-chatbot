resource "kubernetes_secret" "gemini_secrets" {
  metadata {
    name      = "gemini-secrets"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
  }

  type = "Opaque"

  data = {
    "gemini-api-key" = var.gemini_api_key
  }
}

resource "kubernetes_secret" "langchain_secrets" {
  metadata {
    name      = "langchain-secrets"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
  }

  type = "Opaque"

  data = {
    "langchain-api-key" = var.langsmith_api_key
  }

}

resource "kubernetes_secret" "neo4j_secrets" {
  metadata {
    name      = "neo4j-secrets"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
  }

  type = "Opaque"

  data = {
    "neo4j-username" = var.neo4j_username
    "neo4j-password" = var.neo4j_password
    "neo4j-auth"     = "${var.neo4j_username}/${var.neo4j_password}"
  }
}

#region Postgres Secrets
resource "kubernetes_secret" "postgres_secrets" {
  metadata {
    name      = "postgres-secrets"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
  }

  type = "Opaque"

  data = {
    "postgres-db"       = var.postgres_db
    "postgres-user"     = var.postgres_user
    "postgres-password" = var.postgres_password
  }
}
#endregion

