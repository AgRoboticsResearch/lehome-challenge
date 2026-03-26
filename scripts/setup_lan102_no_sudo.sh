#!/bin/bash
# LeHome Challenge Training-Only Setup for lan102 (NO SUDO)
# This setup assumes system packages are already installed

set -e

echo "=========================================="
echo "LeHome Training-Only Setup (lan102)"
echo "Non-privileged user mode"
echo "=========================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="/home/hls/codes/lehome-challenge"

# Check if we're in the right place
cd "$PROJECT_DIR" || {
    echo -e "${RED}Error: Project directory not found${NC}"
    echo "Please run: ./scripts/sync.sh push lan102"
    exit 1
}

# Check Python version
echo -e "\n${GREEN}[1/4] Checking Python availability...${NC}"
PYTHON_CMD=""
for py in python3.11 python3; do
    if command -v $py &> /dev/null; then
        PYTHON_VERSION=$($py --version | awk '{print $2}')
        PYTHON_MAJOR=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "Found: $py ($PYTHON_VERSION)"
        PYTHON_CMD=$py
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3.11 not found${NC}"
    echo "Please ask admin to install Python 3.11"
    exit 1
fi

# Check if we can use Python 3.11
if [[ "$PYTHON_VERSION" != 3.11* ]]; then
    echo -e "${YELLOW}Warning: Python 3.11 is recommended, found $PYTHON_VERSION${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install uv locally (no sudo needed)
echo -e "\n${GREEN}[2/4] Installing uv (local, no sudo)...${NC}"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed: $(uv --version)"
fi

# Create venv and install dependencies
echo -e "\n${GREEN}[3/4] Creating Python environment...${NC}"
if [ -f "requirements-train.txt" ]; then
    echo "Using requirements-train.txt..."
    uv venv --python 3.11 || $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    uv pip install -r requirements-train.txt
else
    echo "requirements-train.txt not found, installing core packages..."
    uv venv --python 3.11 || $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
    uv pip install lerobot==0.4.3
    uv pip install num2words open3d transforms3d transformers
fi

# Install LeHome package
echo -e "\n${GREEN}[4/4] Installing LeHome package...${NC}"
uv pip install -e ./source/lehome

echo -e "\n${GREEN}=========================================="
echo "Setup complete!"
echo "==========================================${NC}"
echo -e "\n${YELLOW}To activate environment:${NC}"
echo "  source /home/hls/codes/lehome-challenge/.venv/bin/activate"
echo ""
echo -e "${YELLOW}To check installation:${NC}"
echo "  python -c 'import torch; import lerobot; print(\"OK\")'"
echo ""
echo "${YELLOW}Note:${NC} Isaac Sim NOT installed (requires Ubuntu 22.04+)"
echo "      Use lan103 for simulation tasks"
