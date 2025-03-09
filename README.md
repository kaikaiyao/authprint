# 🚀 Running Kubernetes Workloads on EIDF Clusters  
**A guide to deploying Dockerized applications, managing storage, and executing batch jobs on EIDF Kubernetes clusters.**  

---

## 📦 1. Building and Pushing Your Docker Image (Windows/PowerShell)  

### 1.1 Build the Docker Image  
Navigate to your `Dockerfile` directory and run:  
```powershell
# Standard build (caches layers)
docker build -t my_app .

# Force fresh build (bypass cache for updates)
docker build --build-arg CACHEBUST=$(Get-Date -UFormat %s) -t my_app .
```

### 1.2 Tag and Push to a Registry  
```powershell
# Tag the image for your Docker Hub repository
docker tag my_app kaiyaoed/my_app:latest

# Push to Docker Hub
docker push kaiyaoed/my_app:latest
```

### 1.3 Test Locally  
Validate the image works before deployment:  
```powershell
docker run -it --rm my_app /bin/bash
```

---

## 🚀 2. Deploying a Pod on Kubernetes  

### 2.1 Create a PersistentVolumeClaim (PVC)  
PVCs provide persistent storage for your pod. Run this on your target cluster (e.g., `eidf029`):  
```bash
kubectl create -f <(PVCNAME=$USER-ws1 STORAGE=250Gi envsubst < pvc.yml)
```
*Replace `pvc.yml` with your PVC definition file.*

### 2.2 Launch the Pod  
Deploy your pod with the PVC attached:  
```bash
WORKSPACE_PVC=$USER-ws1 envsubst < my_app.yml | kubectl create -f -
```
*Ensure `my_app.yml` references the correct PVC name in `claimName`.*

---

## 🔍 3. Monitoring and Managing Pods  

- **Check Pod Status**  
  ```bash
  kubectl describe pod my-app
  ```

- **View Logs**  
  ```bash
  kubectl logs my-app
  ```

- **Delete the Pod**  
  ```bash
  kubectl delete pod my-app
  ```

---

## 🗑️ 4. PVC Management  

### 4.1 Inspect PVC Data  
1. Start a temporary sync pod:  
   ```bash
   kubectl pvcsync $USER-ws1 up
   ```
   *Wait for the "listening on port" confirmation.*

2. List or copy files:  
   ```bash
   kubectl exec -it $USER-ws1-rsync-backend -- ls /data
   kubectl cp $USER-ws1-rsync-backend:/data/ ./local-folder/
   ```

3. Clean up the sync pod:  
   ```bash
   kubectl pvcsync $USER-ws1 down
   ```

### 4.2 Delete the PVC  
```bash
kubectl delete pvc $USER-ws1
```

---

## 🔄 5. Running Batch Jobs (Ablation Studies/Parameter Search)  

For parallel tasks like hyperparameter tuning, use Kubernetes **Jobs**.  

### 5.1 Submit a Batch Job  
Create an arrayed job (similar to Slurm jobs):  
```bash
envsubst '$USER $INFK8S_QUEUE_NAME $INFK8S_NFS_SERVER_IP' < job-array.yaml | kubectl create -f -
```
*Replace `job-array.yaml` with your Job template. Uses `$INFK8S_QUEUE_NAME` for resource allocation.*

### 5.2 Monitor Job Progress  
- **Check Job Status**  
  ```bash
  kubectl describe job ${USER}-job-ablation
  ```
  *Lists all pods created by the job.*

- **Check All Individual Pods Status**
  ```bash
  kubectl get pods -l job-name=${USER}-job-ablation
  ```
  or
  ```bash
  kubectl get pods -l eidf/user=${USER}
  ```
  
- **Verify Pod Readiness**  
  Wait until all pods are ready:  
  ```bash
  kubectl wait pod --for=condition=ready -l eidf/user=${USER}
  ```

### 5.3 Useful Commands

- **Get Batched Output**  
  For example, you can use the below command to grep all GPU type prints from all Pods in this batch Job:
  ```bash
  kubectl get pods -l job-name=${USER}-job-ablation -o jsonpath='{.items[*].metadata.name}' | xargs -n1 sh -c 'echo "$0:" $(kubectl logs $0 2>/dev/null | grep "GPU(s) available: NVIDIA" | awk "{print \$4}")'
  ```
---

## 📎 Appendix  

### 🖥️ Machine Relationships & SSH Access  
- **Local Machine**: Your Windows workstation for building images.  
- **Clusters**:  
  - `eidf107`: Accessed via `ssh -J s2470447-eidf107@eidf-gateway.epcc.ed.ac.uk s2470447-eidf107@10.24.6.77`  
  - `eidf029` (infk8s): Accessed via `ssh -J s2470447-infk8s@eidf-gateway.epcc.ed.ac.uk s2470447-infk8s@10.24.5.121`  

🔐 **Two-Factor Authentication (TOTP)**: Required for SSH access. Use Microsoft Authenticator for codes.  

---

### 🛠️ Other Useful Commands  
- **Check Cluster Quotas**  
  ```bash
  kubectl describe quota
  ```

- **Check All Jobs Pods and PVCs**
  ```bash
  kubectl get jobs,pods,pvc | grep s2470447
  ```
  
- **Print Project for LLM**
  ```bash
  { find . -type d -print -o -name "*.py" -print | tree -fi --fromfile && find . -name "*.py" -print0 | while IFS= read -r -d '' file; do echo "=== $file ==="; cat "$file"; done; } > project_structure_and_codes.txt
  ```

- **To batch-delete jobs and pvcs for ablation study**
  ```bash
  kubectl delete job s2470447-infk8s-job-train-ablation-{0..7} --force --grace-period=0
  kubectl delete pvc s2470447-infk8s-ws-{0..7} --force --grace-period=0
  ```

---

✨ **Pro Tips**  
- Use MobaXTerm tabs for easy access to PowerShell, `eidf107`, and `eidf029`.  
- For multi-container workflows, explore Kubernetes `CronJobs` or `Deployments`.  
