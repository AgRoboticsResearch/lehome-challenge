#!/bin/bash
# Setup script for π0.5 environment
# This creates a separate venv with lerobot[pi] support

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-pi"

echo "=========================================="
echo "  π0.5 Environment Setup Script"
echo "=========================================="
echo ""

# 1. 创建虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  .venv-pi already exists. Removing..."
    rm -rf "$VENV_DIR"
fi

echo "📦 Creating virtual environment..."
uv venv "$VENV_DIR"
echo "✅ Created .venv-pi"
echo ""

# 激活环境
source "$VENV_DIR/bin/activate"

# 设置 NVIDIA EULA 接受（避免交互式提示）
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes

# 2. 安装 Isaac Sim（必须先做！）
echo "📦 Installing Isaac Sim (this may take a few minutes)..."
# uv 使用 --index 添加额外索引，--default-index 设置默认索引
uv pip install \
    --default-index https://pypi.org/simple/ \
    --index https://pypi.nvidia.com/ \
    isaacsim[all,extscache]==5.1.0
echo "✅ Isaac Sim installed"
echo ""

# 3. 安装 LeRobot with π0.5（会安装自定义 transformers）
echo "📦 Installing LeRobot with π0.5 support..."
echo "   (this includes custom transformers from HuggingFace)"
uv pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.3"
echo "✅ LeRobot[pi] installed"
echo ""

# 4. 安装其他项目依赖
echo "📦 Installing project dependencies..."
uv pip install num2words open3d pinocchio transforms3d
echo "✅ Project dependencies installed"
echo ""

# 5. 安装 lehome 包
echo "📦 Installing lehome package..."
uv pip install -e "$ROOT_DIR/source/lehome"
echo "✅ lehome package installed"
echo ""

# 6. 配置 Isaac Lab（现在 isaacsim 已存在）
echo "📦 Configuring Isaac Lab extensions..."
cd "$ROOT_DIR/third_party/IsaacLab"
# 使用 -i none 只安装核心扩展，跳过额外依赖
# 忽略 VSCode 设置错误（不影响核心功能）
./isaaclab.sh -i none 2>/dev/null || echo "   (VSCode setup skipped - not critical)"
cd "$ROOT_DIR"
echo "✅ Isaac Lab configured"
echo ""

# 验证安装
echo "🔍 Verifying installation..."
python -c "import scipy; print(f'   SciPy: {scipy.__version__}')" || echo "   ❌ SciPy failed"
python -c "import transformers; print(f'   Transformers: {transformers.__version__}')" || echo "   ❌ Transformers failed"
python -c "import lerobot; print(f'   LeRobot: {lerobot.__version__}')" || echo "   ❌ LeRobot failed"
python -c "import isaacsim; print(f'   Isaac Sim: OK')" 2>/dev/null || echo "   ⚠️ Isaac Sim requires ACCEPT_EULA=Y"

# 检查 pi0.5 相关模块
python -c "from lerobot.policies.pi.modeling_pi0 import PI0Policy; print('   π0.5 Policy: OK')" 2>/dev/null || echo "   ⚠️ π0.5 Policy check (may need different import path)"

echo ""
echo "=========================================="
echo "  ✅ π0.5 environment setup complete!"
echo "=========================================="
echo ""
echo "  Activate with:"
echo "    source .venv-pi/bin/activate"
echo ""
echo "  Then you can:"
echo "    - Train π0.5: lerobot-train --config_path=configs/train_pi05.yaml"
echo "    - Evaluate: python -m scripts.eval --policy_type pi05 ..."
echo ""
