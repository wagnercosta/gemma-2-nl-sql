#!/bin/bash
pip install -r requirements.txt

# Check CUDA device capability
CUDA_CAPABILITY=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])")

if [ "$CUDA_CAPABILITY" -lt 8 ]; then
    echo "Hardware not supported for Flash Attention"
    exit 1
fi



# Install prerequisites
pip install ninja packaging

# Install flash-attn with a limit on the number of build jobs
export MAX_JOBS=2
pip install flash-attn --no-build-isolation --upgrade
