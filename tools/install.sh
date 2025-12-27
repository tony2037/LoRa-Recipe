#!/bin/bash
set -e  # 如果任何指令失敗，立即停止腳本

WORKSPACE="/workspace"
REPO_URL="https://github.com/kohya-ss/musubi-tuner"
#REPO_URL="https://github.com/Teckt/wan-lora-trainer.git"
REPO_DIR="$WORKSPACE/wan-lora-trainer"
MODEL_DIR="$WORKSPACE/models"

echo ">>> [Init] Starting Setup for Wan2.2 LoRA Training Pipeline..."

# ==========================================
# 2. Python 環境與 Repo 設置
# ==========================================
cd $WORKSPACE

if [ ! -d "$REPO_DIR" ]; then
    echo ">>> [Repo] Cloning wan-lora-trainer..."
    git clone $REPO_URL
else
    echo ">>> [Repo] Repository already exists, pulling latest..."
    cd $REPO_DIR && git pull
fi

cd $REPO_DIR

echo ">>> [Python] Installing Python dependencies..."
# 升級 pip
pip install --upgrade pip

# 安裝核心依賴 (根據 repo 文檔)
# 注意：PyTorch 2.5.1 已包含在 Base Image 中，這裡安裝 repo 的其餘依賴
pip install torch torchvision
pip install -e .

# 安裝額外工具
pip install huggingface_hub[cli] accelerate transformers
pip install ascii-magic matplotlib tensorboard prompt-toolkit

# 安裝 Flash Attention (加速推論與訓練，選擇性但強烈建議)
echo ">>> [Python] Installing Flash Attention (This may take a while)..."
pip install flash-attn --no-build-isolation

# ==========================================
# 3. 模型權重下載 (Model Download Automation)
# ==========================================
# 建立模型目錄結構
mkdir -p $MODEL_DIR/t5
mkdir -p $MODEL_DIR/clip
mkdir -p $MODEL_DIR/vae
mkdir -p $MODEL_DIR/dit_wan21
mkdir -p $MODEL_DIR/dit_wan22

echo ">>> [Models] Starting Model Downloads. This will skip existing files."

# 設定 Hugging Face CLI 下載函數
download_file() {
    local repo_id=$1
    local filename=$2
    local local_dir=$3
    echo "Downloading $filename from $repo_id to $local_dir..."
    huggingface-cli download "$repo_id" "$filename" --local-dir "$local_dir" --local-dir-use-symlinks False
}

# ==========================================
# 4. 配置 Accelerate
# ==========================================
echo ">>> [Config] Setting up Accelerate config..."
mkdir -p /root/.cache/huggingface/accelerate
cat > /root/.cache/huggingface/accelerate/default_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

#accelerate config # configure Accelerate. Choose appropriate values for each question


echo ">>> [Done] Environment setup complete! Ready to train."
echo ">>> Useful Paths:"
echo "    Repo: $REPO_DIR"
echo "    Models: $MODEL_DIR"
