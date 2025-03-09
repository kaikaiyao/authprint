#!/bin/bash

# Specify the namespace you have access to
NAMESPACE="informatics"

# Fetch all pod information in JSON format for the specified namespace
pods=$(kubectl get pods -n "$NAMESPACE" -o json)

# Initialize an associative array to store GPU usage per user
declare -A gpu_usage

# Parse the JSON output
while read -r line; do
    # Extract the user label
    user=$(echo "$line" | jq -r '.metadata.labels."eidf/user"')
    
    # Extract the pod status phase
    pod_status=$(echo "$line" | jq -r '.status.phase')
    
    # Extract the GPU request
    gpu_request=$(echo "$line" | jq -r '.spec.containers[].resources.requests."nvidia.com/gpu"')
    
    # Check if both user and gpu_request are not null and pod is in Running state
    # Exclude Pending, Error, StartError, and Completed pods
    if [[ "$user" != "null" && "$gpu_request" != "null" && 
          "$pod_status" != "Pending" && "$pod_status" != "Error" && 
          "$pod_status" != "StartError" && "$pod_status" != "Succeeded" ]]; then
        # Add the GPU request to the user's total GPU usage
        gpu_usage["$user"]=$((${gpu_usage["$user"]:-0} + gpu_request))
    fi
done < <(echo "$pods" | jq -c '.items[]')

# Find the longest username for proper formatting
max_length=4  # Start with minimum "USER" length
for user in "${!gpu_usage[@]}"; do
    user_length=${#user}
    if [[ $user_length -gt $max_length ]]; then
        max_length=$user_length
    fi
done

# Add padding for better visual separation
column_width=$((max_length + 4))

# Output the GPU usage per user with proper alignment
printf "%-${column_width}s %s\n" "USER" "GPUs"
for user in "${!gpu_usage[@]}"; do
    printf "%-${column_width}s %s\n" "$user" "${gpu_usage[$user]}"
done