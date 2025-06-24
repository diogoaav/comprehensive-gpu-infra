# JuiceFS Shared Storage for LLM Training

This project demonstrates how to set up JuiceFS shared storage backed by DigitalOcean Spaces and Redis for distributed LLM training, using the OpenWebText dataset as an example.

## Step-by-Step Setup Guide

### 1. Create a DigitalOcean Managed Redis Database

1. Go to the DigitalOcean Control Panel
2. Click on "Databases" in the left menu
3. Click "Create Database Cluster"
4. Select "Redis" as the engine
5. Choose your preferred region and plan
6. Click "Create Database Cluster"
7. Once created, copy the private connection string (you'll need this later)

### 2. Create a DigitalOcean Spaces Bucket

1. Go to the DigitalOcean Control Panel
2. Click on "Spaces" in the left menu
3. Click "Create Space"
4. Select your preferred region
5. Give your bucket a unique name
6. Click "Create Space"
7. Generate Spaces access keys if you haven't already:
   - Go to API -> Spaces access keys
   - Click "Generate New Key"
   - Save both the access key and secret key

### 3. Set Up Environment Variables

Create a `.env` file with your credentials (see `.env.example` for the required format).

### 4. Set Up Kubernetes Cluster

1. Create a Kubernetes cluster with CPU nodes using `setup_cluster.sh`
2. Get the kubeconfig using `doctl kubernetes cluster kubeconfig save training-cluster`

### 5. Install JuiceFS CSI Driver in Kubernetes

Install the JuiceFS CSI driver using Helm as described in the `install_juicefs_csi.sh` script.

### 6. Create JuiceFS Storage Class and PVC

Apply the `juicefs-storage.yaml` configuration to create the storage class and persistent volume claim.

### 7. Load OpenWebText Dataset

Apply the `load-openwebtext.yaml` job to download and save the OpenWebText dataset to the JuiceFS volume.

### 8. Tokenize the Dataset

Apply the `tokenize-openwebtext.yaml` job to tokenize the dataset in parallel across multiple pods.

### 9. Verify the Tokenized Dataset

Create a pod to check the tokenized files using the `check-tokenized.yaml` configuration.

## Dataset Structure

After tokenization, the dataset will be organized as:

```