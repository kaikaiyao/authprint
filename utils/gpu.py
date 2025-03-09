import subprocess
import torch

def get_gpu_info():
    try:
        # Run the nvidia-smi command to get GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            # Print the GPU name(s)
            gpu_info = result.stdout.strip()
            print(f"GPU(s) available: {gpu_info}")
        else:
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")


def initialize_cuda():
    try:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        _ = torch.empty(1).cuda()
        print(f"Discovered {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA initialization failed: {str(e)}")
        return torch.device("cpu")