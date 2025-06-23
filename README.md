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

Create a `.env` file with your credentials:

```bash
# DigitalOcean Spaces Configuration
DO_SPACES_ACCESS_KEY=your_spaces_access_key
DO_SPACES_SECRET_KEY=your_spaces_secret_key
DO_SPACES_ENDPOINT=your_spaces_endpoint (e.g., nyc3.digitaloceanspaces.com)
DO_SPACES_BUCKET=your_bucket_name

# Redis Configuration
REDIS_URL=your_redis_connection_string
```

### 4. Set Up Ubuntu Droplet

1. Create a new Ubuntu droplet from the DigitalOcean control panel
2. SSH into your droplet
3. Install required dependencies:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip fuse
```

### 5. Install and Configure JuiceFS

1. Download and install JuiceFS:
```bash
curl -L https://juicefs.com/static/juicefs -o juicefs
chmod +x juicefs
sudo mv juicefs /usr/local/bin
```

2. Format the JuiceFS filesystem:
```bash
source .env
juicefs format \
    --storage s3 \
    --bucket ${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET} \
    --access-key ${DO_SPACES_ACCESS_KEY} \
    --secret-key ${DO_SPACES_SECRET_KEY} \
    "${REDIS_URL}" \
    myjfs
```

3. Mount the filesystem:
```bash
sudo juicefs mount -d \
    --cache-dir /var/jfsCache \
    --cache-size 102400 \
    "${REDIS_URL}" \
    /mnt/jfs
```

### 6. Load OpenWebText Dataset

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Load the dataset:
```bash
python load_openwebtext.py --output-dir /mnt/jfs/datasets/openwebtext
```

## Dataset Structure

After loading, your dataset will be organized as:

```
/mnt/jfs/
└── datasets/
    └── openwebtext/
        ├── 0/
        │   ├── 0
        │   ├── 1
        │   └── ...
        ├── 1/
        └── ...
```

- Each subdirectory (`0`, `1`, ...) contains a portion of the dataset
- Files are named with numeric IDs
