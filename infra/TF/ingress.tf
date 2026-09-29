resource "kubernetes_ingress_v1" "rag_ingress" {
  metadata {
    name      = "theme-based-rag-ingress"
    namespace = kubernetes_namespace.rag_namespace.metadata[0].name
    annotations = {
      "kubernetes.io/ingress.class"                = "nginx"
      "nginx.ingress.kubernetes.io/rewrite-target" = "/"
    }
  }

  spec {
    rule {
      http {
        path {
          path      = "/"
          path_type = "Prefix"

          backend {
            service {
              name = kubernetes_service.backend_service.metadata[0].name
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}
