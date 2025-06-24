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

## Training

### Distributed SLURM Training

This section covers how to set up a SLURM cluster for distributed training workloads with shared storage using JuiceFS.

#### Master Node Setup

Follow these steps to set up the SLURM master node:

```bash
# Connect to the master node
ssh root@<master-ip-address>

# Update package lists and install SLURM packages
apt update
apt install -y slurmd slurmctld slurm-client

# Create necessary directories
mkdir -p /etc/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
chown slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
```

Create the SLURM configuration files:
- `/etc/slurm/slurm.conf`: Main configuration file (see `configs/slurm.conf` for template)
- `/etc/slurm/cgroup.conf`: Control groups configuration (see `configs/cgroup.conf` for template)

```bash
# Restart MUNGE and SLURM services
systemctl enable munge
systemctl restart munge
systemctl enable slurmctld
systemctl restart slurmctld
systemctl enable slurmd
systemctl restart slurmd

# Check SLURM status and update node state
sinfo
scontrol update NodeName=slurm-cluster-master State=IDLE

# Verify configuration with a test job
srun hostname
```

#### GPU Compute Node Setup

Follow these steps to add a GPU compute node to your SLURM cluster:

```bash
# Connect to the GPU node
ssh root@<gpu-node-ip>

# Install SLURM packages and NVIDIA drivers
apt update
apt install -y slurmd slurm-client
apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Create necessary directories
mkdir -p /etc/slurm /var/spool/slurmd /var/log/slurm
chown slurm:slurm /var/spool/slurmd /var/log/slurm
```

Transfer configuration files from the master node:
- Copy MUNGE key from master to compute node
- Copy slurm.conf and cgroup.conf from master to compute node
- Create gres.conf for GPU resource management

```bash
# On master node, update slurm.conf to include the GPU node
cat >> /etc/slurm/slurm.conf << 'EOF'

# GPU Node Configuration
NodeName=compute-1 NodeAddr=<gpu-node-ip> CPUs=8 Sockets=1 CoresPerSocket=8 ThreadsPerCore=1 RealMemory=64000 Gres=gpu:rtx6000:1 State=UNKNOWN
EOF

# Start SLURM daemon on GPU node
systemctl enable slurmd
systemctl restart slurmd

# On master node, check if the GPU node is visible and set to IDLE
sinfo
scontrol update NodeName=compute-1 State=IDLE

# Test GPU access with a SLURM job
srun -N1 --nodelist=compute-1 nvidia-smi
```

#### Running Jobs

Basic job submission commands:

```bash
# Interactive job
srun <command>

# Batch job
sbatch job_script.sh
```

Example batch script (`example_job.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=/path/to/output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

echo "Running job on $(hostname)"
```

#### Running Distributed Training Jobs

With all three GPU nodes successfully connected to the SLURM cluster, you can now run distributed training jobs. Here are some examples:

```bash
# Check cluster status
sinfo

# View all available GPUs
srun -N3 nvidia-smi

# Run a simple distributed job across all nodes
srun -N3 hostname

# Copy the example scripts to the shared filesystem
cp examples/distributed_training.py /mnt/jfs/
cp examples/distributed_job.sh /mnt/jfs/
cp examples/simple_distributed_job.sh /mnt/jfs/

# Test with a simple distributed job first
sbatch /mnt/jfs/simple_distributed_job.sh

# Then submit the PyTorch distributed training job
sbatch /mnt/jfs/distributed_job.sh

# Monitor the jobs
squeue
tail -f /mnt/jfs/simple_dist_*.log
tail -f /mnt/jfs/dist_train_*.log
```

The distributed training example includes:
- Automatic PyTorch installation with CUDA support
- Environment setup for Python path
- Proper GRES specification for RTX 6000 GPUs

The simple test script (`examples/simple_distributed_job.sh`) provides a quick way to verify:
- SLURM environment variables are correctly set
- GPU access is working on all nodes
- Inter-node communication is functioning

#### Using JuiceFS for Distributed Checkpointing

When running distributed training, you can use the shared JuiceFS filesystem for checkpointing:

```python
# Example checkpoint saving code
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Save checkpoint to shared filesystem
save_checkpoint(model, optimizer, epoch, f"/mnt/jfs/checkpoints/model_epoch_{epoch}.pt")
```

This ensures that checkpoints are accessible from all nodes, allowing for:
- Fault tolerance and job resumption
- Model evaluation on different nodes
- Easy model deployment after training

#### Cluster Status Verification

After setting up the SLURM cluster, verify everything is working:

```bash
# Check cluster status
sinfo

# Verify GPU resources are available
scontrol show nodes | grep Gres

# Test simple distributed job
sbatch /mnt/jfs/simple_distributed_job.sh

# Test PyTorch distributed training
sbatch /mnt/jfs/distributed_job.sh

# Monitor jobs
squeue
tail -f /mnt/jfs/dist_train_*.log
```

#### Successful Setup Indicators

Your cluster is ready when you see:
- All nodes show `idle` state in `sinfo`
- GPU resources show `Gres=gpu:rtx6000:1` for compute nodes
- Distributed jobs complete successfully across all 3 GPU nodes
- PyTorch can access CUDA and create DistributedDataParallel models

#### Next Steps

With your cluster operational, you can now:
1. Load your training datasets into JuiceFS shared storage
2. Develop distributed training scripts using the examples as templates
3. Submit large-scale training jobs across the 3 RTX 6000 GPUs
4. Use JuiceFS for model checkpointing and data sharing