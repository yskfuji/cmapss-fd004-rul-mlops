terraform {
  required_version = ">= 1.6.0"

  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.26"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  runtime_bucket_name                  = coalesce(var.runtime_bucket_name, "${var.project_id}-${var.service_name}-runtime")
  runtime_bucket_location              = coalesce(var.runtime_bucket_location, var.region)
  runtime_mount_path                   = "/mnt/runtime"
  model_registry_db_path               = "${local.runtime_mount_path}/registry/trained_models.db"
  model_artifacts_root                 = "${local.runtime_mount_path}/model_artifacts"
  request_audit_log_path               = "${local.runtime_mount_path}/audit/request_audit_log.jsonl"
  drift_baseline_path                  = "${local.runtime_mount_path}/monitoring/drift_baseline.json"
  model_promotion_registry_path        = "${local.runtime_mount_path}/promotion/model_promotions.json"
  cloudsql_mount_path                  = "/cloudsql"
  worker_container_image               = coalesce(var.worker_container_image, var.container_image)
  runtime_db_password_secret_id        = "${var.service_name}-db-password"
  runtime_db_dsn_secret_id             = "${var.service_name}-runtime-postgres-dsn"
  runtime_db_unix_socket_path          = "${local.cloudsql_mount_path}/${google_sql_database_instance.runtime.connection_name}"
  runtime_db_dsn                       = "postgresql://${urlencode(var.cloudsql_user_name)}:${urlencode(random_password.runtime_db_password.result)}@/${var.cloudsql_database_name}?host=${urlencode(local.runtime_db_unix_socket_path)}"
  api_plain_env = {
    RULFM_JOB_EXECUTION_BACKEND           = "worker"
    RULFM_JOB_STORE_BACKEND               = "postgres"
    RULFM_MODEL_REGISTRY_BACKEND          = "postgres"
    RULFM_METRICS_ENABLED                 = "1"
    RULFM_PROMETHEUS_MULTIPROC_DIR        = "/tmp/prometheus"
    RULFM_MODEL_REGISTRY_DB_PATH          = local.model_registry_db_path
    RULFM_MODEL_ARTIFACTS_ROOT            = local.model_artifacts_root
    RULFM_FORECASTING_API_AUDIT_LOG_PATH  = local.request_audit_log_path
    RULFM_DRIFT_BASELINE_PATH             = local.drift_baseline_path
    RULFM_MODEL_PROMOTION_REGISTRY_PATH   = local.model_promotion_registry_path
  }
  worker_plain_env = merge(
    local.api_plain_env,
    {
      RULFM_JOB_WORKER_MODE             = "batch"
      RULFM_JOB_WORKER_MAX_JOBS_PER_RUN = tostring(var.worker_max_jobs_per_run)
    }
  )
}

resource "google_artifact_registry_repository" "arcayf" {
  location      = var.region
  repository_id = var.repository_id
  description   = "ARCAYF forecasting API container images"
  format        = "DOCKER"
}

resource "google_service_account" "arcayf_runtime" {
  account_id   = "arcayf-forecasting-runtime"
  display_name = "ARCAYF Forecasting Runtime"
}

resource "google_service_account" "arcayf_scheduler" {
  account_id   = "arcayf-forecasting-scheduler"
  display_name = "ARCAYF Forecasting Worker Scheduler"
}

resource "google_secret_manager_secret_iam_member" "api_key_accessor" {
  secret_id = var.api_key_secret_name
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.arcayf_runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "bearer_token_accessor" {
  count     = var.bearer_token_secret_name == null ? 0 : 1
  secret_id = var.bearer_token_secret_name
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.arcayf_runtime.email}"
}

resource "google_project_iam_member" "runtime_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.arcayf_runtime.email}"
}

resource "google_storage_bucket" "runtime_state" {
  name                        = local.runtime_bucket_name
  location                    = local.runtime_bucket_location
  storage_class               = var.runtime_bucket_storage_class
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = var.runtime_bucket_retention_days
    }
  }
}

resource "google_storage_bucket_iam_member" "runtime_state_object_admin" {
  bucket = google_storage_bucket.runtime_state.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.arcayf_runtime.email}"
}

resource "random_password" "runtime_db_password" {
  length  = 32
  special = true
}

resource "google_sql_database_instance" "runtime" {
  name                = var.cloudsql_instance_name
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = var.cloudsql_deletion_protection

  settings {
    tier              = var.cloudsql_tier
    availability_type = var.cloudsql_availability_type
    disk_size         = var.cloudsql_disk_size_gb
    disk_type         = "PD_SSD"

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }
}

resource "google_sql_database" "runtime" {
  name     = var.cloudsql_database_name
  instance = google_sql_database_instance.runtime.name
}

resource "google_sql_user" "runtime" {
  name     = var.cloudsql_user_name
  instance = google_sql_database_instance.runtime.name
  password = random_password.runtime_db_password.result
}

resource "google_secret_manager_secret" "runtime_db_password" {
  secret_id = local.runtime_db_password_secret_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "runtime_db_password" {
  secret      = google_secret_manager_secret.runtime_db_password.id
  secret_data = random_password.runtime_db_password.result
}

resource "google_secret_manager_secret" "runtime_db_dsn" {
  secret_id = local.runtime_db_dsn_secret_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "runtime_db_dsn" {
  secret = google_secret_manager_secret.runtime_db_dsn.id
  secret_data = local.runtime_db_dsn
}

resource "google_secret_manager_secret_iam_member" "runtime_db_dsn_accessor" {
  secret_id = google_secret_manager_secret.runtime_db_dsn.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.arcayf_runtime.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  ingress  = var.run_ingress
  depends_on = [
    google_project_iam_member.runtime_cloudsql_client,
    google_secret_manager_secret_iam_member.api_key_accessor,
    google_secret_manager_secret_iam_member.runtime_db_dsn_accessor,
    google_storage_bucket_iam_member.runtime_state_object_admin,
  ]

  template {
    service_account = google_service_account.arcayf_runtime.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    timeout               = "${var.api_timeout_seconds}s"
    max_instance_request_concurrency = var.api_max_instance_request_concurrency

    volumes {
      name = "runtime-state"
      gcs {
        bucket    = google_storage_bucket.runtime_state.name
        read_only = false
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.runtime.connection_name]
      }
    }

    containers {
      image = var.container_image
      ports {
        container_port = 8000
      }
      volume_mounts {
        name       = "runtime-state"
        mount_path = local.runtime_mount_path
      }
      volume_mounts {
        name       = "cloudsql"
        mount_path = local.cloudsql_mount_path
      }
      env {
        name  = "RULFM_FORECASTING_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.api_key_secret_name
            version = var.api_key_secret_version
          }
        }
      }

      dynamic "env" {
        for_each = local.api_plain_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = var.bearer_token_secret_name == null ? {} : { bearer = var.bearer_token_secret_name }
        content {
          name = "RULFM_FORECASTING_API_BEARER_TOKEN"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = var.bearer_token_secret_version
            }
          }
        }
      }

      env {
        name = "RULFM_JOB_STORE_POSTGRES_DSN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.runtime_db_dsn.secret_id
            version = google_secret_manager_secret_version.runtime_db_dsn.version
          }
        }
      }

      env {
        name = "RULFM_MODEL_REGISTRY_POSTGRES_DSN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.runtime_db_dsn.secret_id
            version = google_secret_manager_secret_version.runtime_db_dsn.version
          }
        }
      }

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
      }
    }

    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }
  }
}

resource "google_cloud_run_v2_job" "worker" {
  name     = var.worker_job_name
  location = var.region

  template {
    task_count  = var.worker_task_count
    parallelism = var.worker_parallelism

    template {
      service_account = google_service_account.arcayf_runtime.email
      timeout         = "${var.worker_timeout_seconds}s"
      max_retries     = 0

      volumes {
        name = "runtime-state"
        gcs {
          bucket    = google_storage_bucket.runtime_state.name
          read_only = false
        }
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.runtime.connection_name]
        }
      }

      containers {
        image   = local.worker_container_image
        command = ["python", "-m", "forecasting_api.job_worker"]

        volume_mounts {
          name       = "runtime-state"
          mount_path = local.runtime_mount_path
        }
        volume_mounts {
          name       = "cloudsql"
          mount_path = local.cloudsql_mount_path
        }

        env {
          name  = "PYTHONPATH"
          value = "/app/src"
        }

        env {
          name  = "RULFM_FORECASTING_API_KEY"
          value_source {
            secret_key_ref {
              secret  = var.api_key_secret_name
              version = var.api_key_secret_version
            }
          }
        }

        dynamic "env" {
          for_each = local.worker_plain_env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = var.bearer_token_secret_name == null ? {} : { bearer = var.bearer_token_secret_name }
          content {
            name = "RULFM_FORECASTING_API_BEARER_TOKEN"
            value_source {
              secret_key_ref {
                secret  = env.value
                version = var.bearer_token_secret_version
              }
            }
          }
        }

        env {
          name = "RULFM_JOB_STORE_POSTGRES_DSN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.runtime_db_dsn.secret_id
              version = google_secret_manager_secret_version.runtime_db_dsn.version
            }
          }
        }

        env {
          name = "RULFM_MODEL_REGISTRY_POSTGRES_DSN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.runtime_db_dsn.secret_id
              version = google_secret_manager_secret_version.runtime_db_dsn.version
            }
          }
        }

        resources {
          limits = {
            cpu    = var.worker_cpu
            memory = var.worker_memory
          }
        }
      }
    }
  }
}

resource "google_cloud_run_v2_job_iam_member" "worker_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.worker.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.arcayf_scheduler.email}"
}

resource "google_cloud_scheduler_job" "worker" {
  name      = var.scheduler_job_name
  region    = var.region
  schedule  = var.worker_schedule
  time_zone = var.worker_schedule_time_zone

  http_target {
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.worker.name}:run"
    http_method = "POST"
    headers = {
      "Content-Type" = "application/json"
    }
    body = base64encode("{}")

    oauth_token {
      service_account_email = google_service_account.arcayf_scheduler.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.worker_invoker]
}
