output "cloud_run_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "service_account_email" {
  value = google_service_account.arcayf_runtime.email
}

output "scheduler_service_account_email" {
  value = google_service_account.arcayf_scheduler.email
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.arcayf.name
}

output "runtime_bucket_name" {
  value = google_storage_bucket.runtime_state.name
}

output "cloudsql_connection_name" {
  value = google_sql_database_instance.runtime.connection_name
}

output "cloudsql_database_name" {
  value = google_sql_database.runtime.name
}

output "worker_job_name" {
  value = google_cloud_run_v2_job.worker.name
}

output "worker_scheduler_job_name" {
  value = google_cloud_scheduler_job.worker.name
}

output "runtime_db_password_secret_name" {
  value = google_secret_manager_secret.runtime_db_password.secret_id
}

output "runtime_db_dsn_secret_name" {
  value = google_secret_manager_secret.runtime_db_dsn.secret_id
}
