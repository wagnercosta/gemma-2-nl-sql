import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available. Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    check_gpu()

# del model
# del trainer
torch.cuda.empty_cache()