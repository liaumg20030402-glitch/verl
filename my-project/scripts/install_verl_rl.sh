#!/bin/bash
conda create -n verl_rl python=3.12 -y
source /home3/medcog/jycai6/.bashrc
conda activate verl_rl
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# 开启多线程编译，加速过程
export MAX_JOBS=64
pip install torch torchvision torchaudio
# pip install "sglang[all]==0.5.2" --no-cache-dir &&
# pip install torch-memory-saver --no-cache-dir
pip install "vllm==0.17.0"


pip install -U "transformers==5.2.0"
# 训练报错参考这个issue: https://github.com/modelscope/ms-swift/issues/8188 (已安装环境里实际生效的文件：~/miniforge3/envs/verl_rl/lib/python3.12/site-packages/transformers/modeling_rope_utils.py)

pip install accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


pip install "flash-attn==2.8.3" --no-build-isolation -v
pip install "flash-linear-attention==0.4.1" --no-build-isolation -v
# cd ~/pkgs/flash-linear-attention-main
# pip install -v . --no-build-isolation


pip install --no-build-isolation "transformer_engine[pytorch]==2.12.0"
pip install "megatron-core==0.16.0"

cd ~/pkgs/mbridge-main
pip install --no-build-isolation -e .


pip install opencv-python opencv-fixer

# avoid being overridden
# pip install nvidia-cudnn-cu12==9.10.2.21

cd ~/pkgs/apex-master
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ~/pkgs/causal-conv1d-main
pip install -v . --no-build-isolation

# 最后装verl
cd ~/pkgs/verl-main
pip install --no-deps -e .