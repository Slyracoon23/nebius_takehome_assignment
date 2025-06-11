terraform {
  backend "s3" {
    bucket = "tfstate-k8s-training-3a014beed862f26e788492272eaf0acc"
    key    = "k8s-training.tfstate"

    endpoints = {
      s3 = "https://storage.eu-north1.nebius.cloud:443"
    }
    region = "eu-north1"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
