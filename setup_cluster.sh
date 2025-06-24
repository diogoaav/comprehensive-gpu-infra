#!/bin/bash

# Create the Kubernetes cluster with one node pool of 3 c-16 nodes
doctl kubernetes cluster create training-cluster \
    --region atl1 \
    --version latest \
    --count 3 \
    --size c-16-intel \
    --tag training

# Get the kubeconfig
doctl kubernetes cluster kubeconfig save training-cluster 