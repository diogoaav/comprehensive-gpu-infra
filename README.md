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

### 2. Create a DigitalOcean Space

1. Go to Spaces in the DigitalOcean Control Panel
2. Create a new Space in the same region as your Redis database
3. Note the Space name and region
4. Generate API keys (Access Key and Secret Key)

### 3. Set up JuiceFS Secret

Copy the example secret file and fill in your credentials:

```bash
cp juicefs-secret.example.yaml juicefs-secret.yaml
# Edit the file with your Redis connection string and DigitalOcean Spaces credentials
```

### 4. Deploy Kubernetes Resources

```bash
# Apply the JuiceFS configuration
kubectl apply -f juicefs-secret.yaml
kubectl apply -f juicefs-storage.yaml
kubectl apply -f juicefs-pvc.yaml

# Verify the setup
kubectl get pvc
```

### 5. Load and Tokenize the Dataset

```bash
# Load the OpenWebText dataset
kubectl apply -f load-openwebtext.yaml

# Wait for completion and verify
kubectl logs -f job/load-openwebtext

# Tokenize the dataset
kubectl apply -f tokenize-job-fixed.yaml

# Monitor tokenization progress
kubectl logs -f job/tokenize-openwebtext
```

## Training

### Distributed SLURM Training

This section covers setting up a SLURM cluster for distributed training workloads with shared storage using JuiceFS.

#### Master Node Setup

Follow these steps to set up the SLURM master node:

```bash
# Connect to the master node
ssh root@<master-ip-address>

# Update and install SLURM packages
apt update
apt install -y slurmd slurmctld slurm-client

# Create necessary directories
mkdir -p /etc/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
chown slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm

# Create SLURM configuration (see configs/slurm.conf for template)
# Create GRES configuration (see configs/gres.conf for template)  
# Create cgroup configuration (see configs/cgroup.conf for template)

# Start services
systemctl enable slurmctld
systemctl start slurmctld
```

#### GPU Compute Node Setup

Follow these steps to add GPU compute nodes to your SLURM cluster:

```bash
# Connect to each GPU node using public IP
ssh root@<gpu-node-public-ip>

# Install SLURM packages and NVIDIA drivers
apt update
apt install -y slurmd slurm-client
apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Create necessary directories
mkdir -p /etc/slurm /var/spool/slurmd /var/log/slurm
chown slurm:slurm /var/spool/slurmd /var/log/slurm

# Update /etc/hosts with private IPs for all nodes
# Copy MUNGE key, slurm.conf, gres.conf, and cgroup.conf from master
# Set hostname to match SLURM configuration

# Start services
systemctl enable slurmd
systemctl start slurmd
```

#### JuiceFS Setup on All Nodes

Install and configure JuiceFS on all SLURM nodes:

```bash
# Install JuiceFS
wget https://github.com/juicedata/juicefs/releases/latest/download/juicefs-linux-amd64.tar.gz
tar -xzf juicefs-linux-amd64.tar.gz
sudo install juicefs /usr/local/bin

# Format JuiceFS (run once from master)
juicefs format \
    --storage s3 \
    --bucket https://<your-space>.nyc3.digitaloceanspaces.com \
    --access-key <your-access-key> \
    --secret-key <your-secret-key> \
    "rediss://default:<password>@<redis-host>:25061/1" \
    shared-juicefs

# Create systemd service for auto-mounting
# Mount JuiceFS on all nodes at /mnt/jfs
```

#### Running Distributed Training Jobs

With all nodes successfully connected to the SLURM cluster, you can run distributed training jobs:

```bash
# Check cluster status
sinfo

# View all available GPUs
srun -N3 nvidia-smi

# Submit distributed training job
sbatch training/train_job.sh

# Monitor training progress
squeue
tail -f /mnt/jfs/training_logs/gpt_training_*.log
```

## Performance Results

### Successful Distributed Training

The infrastructure successfully completed distributed training with the following specifications:

#### Hardware Configuration
- **Master Node**: 1 CPU-only node (SLURM controller)
- **Compute Nodes**: 3 × RTX 6000 Ada Generation GPUs
- **Network**: DigitalOcean private networking (10.10.10.0/24)
- **Storage**: JuiceFS with DigitalOcean Spaces backend

#### Training Performance
- **Model**: 162M parameter GPT model
- **Dataset**: OpenWebText (5.3M sequences, 2.7B tokens)
- **Training Time**: 4 minutes for test run (50 training steps)
- **GPU Utilization**: 100% across all nodes
- **Network Traffic**: 2.75 Gbps peak (gradient synchronization)
- **Storage Performance**: 24GB cached locally, 0% I/O wait

#### Key Performance Metrics

GPU Memory Usage: 7.2GB