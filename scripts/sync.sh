#!/bin/zsh

# Local directory
LOCAL_DIR="$(dirname "$0")/.."
LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd)"

# Server configurations
declare -A SERVER_USER
declare -A SERVER_IP
declare -A SERVER_PORT
declare -A SERVER_DIR
declare -A SERVER_NAME
declare -A SERVER_USES_PROXY

# Server 1: ZJUICI remote server
SERVER_USER[zjuici]="admin"
SERVER_IP[zjuici]="pool.zjuici.com"
SERVER_PORT[zjuici]="31329"
SERVER_DIR[zjuici]="/home/admin/codes/lehome-challenge"
SERVER_NAME[zjuici]="ZJUICI Remote"
SERVER_USES_PROXY[zjuici]="yes"

# Server 2: Local LAN server 102
SERVER_USER[lan102]="hls"
SERVER_IP[lan102]="192.168.3.102"
SERVER_PORT[lan102]="22"
SERVER_DIR[lan102]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan102]="LAN 102"
SERVER_USES_PROXY[lan102]="yes"

# Server 3: Local LAN server 103
SERVER_USER[lan103]="hls"
SERVER_IP[lan103]="192.168.3.103"
SERVER_PORT[lan103]="22"
SERVER_DIR[lan103]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan103]="LAN 103"
SERVER_USES_PROXY[lan103]="yes"

# Server 4: Local LAN server 104
SERVER_USER[lan104]="hls"
SERVER_IP[lan104]="192.168.3.104"
SERVER_PORT[lan104]="22"
SERVER_DIR[lan104]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan104]="LAN 104"
SERVER_USES_PROXY[lan104]="yes"

# Server 5: Local LAN server 206
SERVER_USER[lan206]="hls"
SERVER_IP[lan206]="192.168.3.206"
SERVER_PORT[lan206]="22"
SERVER_DIR[lan206]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan206]="LAN 206"
SERVER_USES_PROXY[lan206]="yes"

# Server 6: Local LAN server 120
SERVER_USER[lan120]="hls"
SERVER_IP[lan120]="192.168.3.120"
SERVER_PORT[lan120]="22"
SERVER_DIR[lan120]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan120]="LAN 120"
SERVER_USES_PROXY[lan120]="yes"

# Default server
DEFAULT_SERVER="zjuici"

# Exclusions for rsync
EXCLUDES=(
  '.git'
  '.venv'
  '__pycache__'
  '*.pyc'
  '*.pyo'
  '.DS_Store'
  'Datasets'
  'Assets'
  'outputs'
  'eval_videos'
  '*.egg-info'
  'third_party'
  '.idea'
  '.vscode'
)

build_excludes() {
  local excludes=""
  for item in "${EXCLUDES[@]}"; do
    excludes="$excludes --exclude=$item"
  done
  echo "$excludes"
}

# Get server identifier (first non-command arg or default)
get_server() {
  local server="$1"
  if [ -z "$server" ] || [[ "$server" =~ ^(push|pull|watch|push-all|datasets|pull-datasets|outputs|push-outputs|outputs-all|push-outputs-all|eval-videos|push-eval-videos|exec|help|-h|--help)$ ]]; then
    echo "$DEFAULT_SERVER"
  else
    echo "$server"
  fi
}

# Get command (first arg)
get_command() {
  local cmd="$1"
  local server="$2"

  if [[ "$cmd" =~ ^(push|pull|watch|push-all|datasets|pull-datasets|outputs|push-outputs|outputs-all|push-outputs-all|eval-videos|push-eval-videos|exec|help|-h|--help)$ ]]; then
    echo "$cmd"
  elif [ -z "$cmd" ]; then
    echo "help"
  else
    # If first arg is a server, second arg is command
    echo "$server"
  fi
}

# Build SSH command with port
build_ssh() {
  local server="$1"
  local user="${SERVER_USER[$server]}"
  local host="${SERVER_IP[$server]}"
  local port="${SERVER_PORT[$server]}"
  echo "ssh -p $port"
}

# Build remote path
build_remote_path() {
  local server="$1"
  local user="${SERVER_USER[$server]}"
  local host="${SERVER_IP[$server]}"
  local dir="${SERVER_DIR[$server]}"
  echo "$user@$host:$dir"
}

# Execute rsync with server config
do_rsync() {
  local server="$1"
  shift
  local ssh_cmd=$(build_ssh "$server")
  rsync -e "$ssh_cmd" "$@"
}

# Execute ssh with server config
do_ssh() {
  local server="$1"
  local user="${SERVER_USER[$server]}"
  local host="${SERVER_IP[$server]}"
  local port="${SERVER_PORT[$server]}"
  local dir="${SERVER_DIR[$server]}"
  shift
  ssh -p "$port" "$user@$host" "$@"
}

# Main script
SERVER=$(get_server "$2")
COMMAND=$(get_command "$1" "$2")

# Re-parse if server was specified first
if [[ "$1" =~ ^(zjuici|lan102|lan103|lan104|lan206|lan120)$ ]]; then
  SERVER="$1"
  COMMAND="$2"
fi

case "${COMMAND:-help}" in
  "push")
    echo "📤 Pushing to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress $(build_excludes) \
      "$LOCAL_DIR/" "$(build_remote_path $SERVER)/"
    ;;
  "pull")
    echo "📥 Pulling from ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress $(build_excludes) \
      "$(build_remote_path $SERVER)/" "$LOCAL_DIR/"
    ;;
  "push-all")
    echo "📤 Pushing ALL files to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
      "$LOCAL_DIR/" "$(build_remote_path $SERVER)/"
    ;;
  "watch")
    echo "👀 Watching for changes... (Ctrl+C to stop)"
    echo "📡 Target: ${SERVER_NAME[$SERVER]} ($SERVER)"
    if ! command -v fswatch &> /dev/null; then
      echo "Error: fswatch is not installed. Install with: brew install fswatch"
      exit 1
    fi
    while true; do
      fswatch -1 -r "$LOCAL_DIR" --exclude='.git' --exclude='.venv' --exclude='__pycache__' 2>/dev/null || sleep 2
      echo "📤 Syncing changes..."
      do_rsync "$SERVER" -avz $(build_excludes) \
        "$LOCAL_DIR/" "$(build_remote_path $SERVER)/"
    done
    ;;
  "datasets")
    echo "📦 Pushing Datasets to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$LOCAL_DIR/Datasets/" "$(build_remote_path $SERVER)/Datasets/"
    ;;
  "pull-datasets")
    echo "📦 Pulling Datasets from ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$(build_remote_path $SERVER)/Datasets/" "$LOCAL_DIR/Datasets/"
    ;;
  "outputs")
    echo "📊 Pulling outputs/train from ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$(build_remote_path $SERVER)/outputs/train/" "$LOCAL_DIR/outputs/train/"
    ;;
  "push-outputs")
    echo "📊 Pushing outputs/train to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$LOCAL_DIR/outputs/train/" "$(build_remote_path $SERVER)/outputs/train/"
    ;;
  "outputs-all")
    echo "📊 Pulling ALL outputs from ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$(build_remote_path $SERVER)/outputs/" "$LOCAL_DIR/outputs/"
    ;;
  "push-outputs-all")
    echo "📊 Pushing ALL outputs to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$LOCAL_DIR/outputs/" "$(build_remote_path $SERVER)/outputs/"
    ;;
  "eval-videos")
    echo "🎥 Pulling eval_videos from ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$(build_remote_path $SERVER)/outputs/eval_videos/" "$LOCAL_DIR/outputs/eval_videos/"
    ;;
  "push-eval-videos")
    echo "🎥 Pushing eval_videos to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    do_rsync "$SERVER" -avz --progress --exclude='.DS_Store' \
      "$LOCAL_DIR/outputs/eval_videos/" "$(build_remote_path $SERVER)/outputs/eval_videos/"
    ;;
  "exec")
    echo "🔐 Connecting to ${SERVER_NAME[$SERVER]} ($SERVER)..."
    local dir="${SERVER_DIR[$SERVER]}"
    do_ssh "$SERVER" "cd $dir && bash --login"
    ;;
  "proxy-on")
    local target_server="${2:-$SERVER}"
    echo "🌐 Enabling mihomo proxy on ${SERVER_NAME[$target_server]} ($target_server)..."
    do_ssh "$target_server" "cd ${SERVER_DIR[$target_server]} && source proxy_on && echo 'Proxy enabled'"
    ;;
  "proxy-off")
    local target_server="${2:-$SERVER}"
    echo "🔌 Disabling mihomo proxy on ${SERVER_NAME[$target_server]} ($target_server)..."
    do_ssh "$target_server" "cd ${SERVER_DIR[$target_server]} && source proxy_off && echo 'Proxy disabled'"
    ;;
  "proxy-status")
    local target_server="${2:-$SERVER}"
    echo "📊 Checking mihomo status on ${SERVER_NAME[$target_server]} ($target_server)..."
    do_ssh "$target_server" "systemctl status mihomo --no-pager 2>/dev/null || echo 'Mihomo not installed'"
    ;;
  "help"|"-h"|"--help"|*)
    cat << 'EOF'
LeHome Challenge Multi-Server Sync Tool

Usage: ./scripts/sync.sh [command] [server]
       ./scripts/sync.sh [server] [command]

Servers:
  zjuici  - ZJUICI remote server (default) [🔐 has mihomo proxy]
  lan102  - Local LAN 192.168.3.102 [🔐 has mihomo proxy]
  lan103  - Local LAN 192.168.3.103 [🔐 has mihomo proxy]
  lan104  - Local LAN 192.168.3.104 [🔐 has mihomo proxy]
  lan206  - Local LAN 192.168.3.206 [🔐 has mihomo proxy]
  lan120  - Local LAN 192.168.3.120 [🔐 has mihomo proxy]

Commands:
  push           - Push code to remote (excludes large dirs)
  pull           - Pull code from remote
  push-all       - Push ALL files including Assets/Datasets
  watch          - Auto-push on file changes (requires fswatch)
  datasets       - Push Datasets/ to remote
  pull-datasets  - Pull Datasets/ from remote
  outputs          - Pull outputs/train from remote
  push-outputs     - Push outputs/train to remote
  outputs-all      - Pull ALL outputs from remote
  push-outputs-all - Push ALL outputs to remote
  eval-videos    - Pull eval_videos from remote
  push-eval-videos - Push eval_videos to remote
  exec           - Run interactive shell on remote
  proxy-on       - Enable mihomo proxy on server (for pip, git, etc.)
  proxy-off      - Disable mihomo proxy on server
  proxy-status   - Check mihomo service status on server
  help           - Show this help message

Examples:
  ./scripts/sync.sh push              # Push to default server (zjuici)
  ./scripts/sync.sh push lan102       # Push to lan102
  ./scripts/sync.sh lan103 pull       # Pull from lan103
  ./scripts/sync.sh push-outputs      # Push outputs/train/ to default server (zjuici)
  ./scripts/sync.sh push-outputs lan102  # Push outputs/train/ to lan102
  ./scripts/sync.sh push-outputs-all  # Push ALL outputs/ to default server
  ./scripts/sync.sh eval-videos       # Pull eval_videos/ from default server (zjuici)
  ./scripts/sync.sh eval-videos lan102  # Pull eval_videos/ from lan102
  ./scripts/sync.sh exec lan102       # SSH into lan102
  ./scripts/sync.sh proxy-on lan102   # Enable mihomo proxy on lan102
  ./scripts/sync.sh proxy-status      # Check mihomo status on default server

Mihomo Integration:
  - Each server runs mihomo locally on 127.0.0.1:7897
  - Use 'proxy-on' to enable proxy for pip, git, curl, etc.
  - Each server has independent proxy access to external services
  - Deploy mihomo: ./scripts/mihomo-setup/deploy_to_server.sh all

Server Configurations:
  zjuici  → admin@pool.zjuici.com:31329 → /home/admin/codes/lehome-challenge
  lan102  → hls@192.168.3.102:22        → /home/hls/codes/lehome-challenge
  lan103  → hls@192.168.3.103:22        → /home/hls/codes/lehome-challenge
  lan104  → hls@192.168.3.104:22        → /home/hls/codes/lehome-challenge
  lan206  → hls@192.168.3.206:22        → /home/hls/codes/lehome-challenge
  lan120  → hls@192.168.3.120:22        → /home/hls/codes/lehome-challenge

To change default server, edit DEFAULT_SERVER variable in this script.
EOF
    ;;
esac
