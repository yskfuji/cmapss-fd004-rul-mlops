variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-northeast1"
}

variable "repository_id" {
  type    = string
  default = "arcayf-forecasting"
}

variable "service_name" {
  type    = string
  default = "arcayf-forecasting-api"
}

variable "run_ingress" {
  type    = string
  default = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  validation {
    condition = contains([
      "INGRESS_TRAFFIC_ALL",
      "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER",
      "INGRESS_TRAFFIC_INTERNAL_AND_CLOUD_LOAD_BALANCING",
    ], var.run_ingress)
    error_message = "run_ingress must be a supported Cloud Run ingress enum value."
  }
}

variable "container_image" {
  type = string
}

variable "worker_container_image" {
  type     = string
  default  = null
  nullable = true
}

variable "api_key_secret_name" {
  type = string
}

variable "api_key_secret_version" {
  type    = string
  default = "latest"
}

variable "bearer_token_secret_name" {
  type     = string
  default  = null
  nullable = true
}

variable "bearer_token_secret_version" {
  type    = string
  default = "latest"
}

variable "min_instance_count" {
  type    = number
  default = 1
}

variable "max_instance_count" {
  type    = number
  default = 3
}

variable "api_cpu" {
  type    = string
  default = "1"
}

variable "api_memory" {
  type    = string
  default = "1Gi"
}

variable "api_timeout_seconds" {
  type    = number
  default = 300
}

variable "api_max_instance_request_concurrency" {
  type    = number
  default = 80
}

variable "worker_job_name" {
  type    = string
  default = "arcayf-forecasting-worker"
}

variable "scheduler_job_name" {
  type    = string
  default = "arcayf-forecasting-worker-schedule"
}

variable "worker_schedule" {
  type    = string
  default = "*/5 * * * *"
}

variable "worker_schedule_time_zone" {
  type    = string
  default = "Etc/UTC"
}

variable "worker_timeout_seconds" {
  type    = number
  default = 3600
}

variable "worker_task_count" {
  type    = number
  default = 1
}

variable "worker_parallelism" {
  type    = number
  default = 1
}

variable "worker_max_jobs_per_run" {
  type    = number
  default = 32
}

variable "worker_cpu" {
  type    = string
  default = "1"
}

variable "worker_memory" {
  type    = string
  default = "1Gi"
}

variable "runtime_bucket_name" {
  type     = string
  default  = null
  nullable = true
}

variable "runtime_bucket_location" {
  type     = string
  default  = null
  nullable = true
}

variable "runtime_bucket_storage_class" {
  type    = string
  default = "STANDARD"
}

variable "runtime_bucket_retention_days" {
  type    = number
  default = 30
}

variable "cloudsql_instance_name" {
  type    = string
  default = "arcayf-forecasting-runtime"
}

variable "cloudsql_database_name" {
  type    = string
  default = "rulfm"
}

variable "cloudsql_user_name" {
  type    = string
  default = "rulfm_app"
}

variable "cloudsql_tier" {
  type    = string
  default = "db-custom-1-3840"
}

variable "cloudsql_disk_size_gb" {
  type    = number
  default = 20
}

variable "cloudsql_availability_type" {
  type    = string
  default = "ZONAL"

  validation {
    condition     = contains(["ZONAL", "REGIONAL"], var.cloudsql_availability_type)
    error_message = "cloudsql_availability_type must be ZONAL or REGIONAL."
  }
}

variable "cloudsql_deletion_protection" {
  type    = bool
  default = true
}
