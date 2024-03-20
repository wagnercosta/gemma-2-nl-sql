#!/bin/bash
pip install -r requirements.txt
pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade
pip install git+https://github.com/huggingface/peft@4a1559582281fc3c9283892caea8ccef1d6f5a4f --upgrade

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
pip install flash-attn --no-build-isolation
