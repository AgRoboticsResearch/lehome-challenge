#!/bin/zsh

##############################################################
# Deploy Mihomo to Multiple Servers
# This script copies the config and setup files to your servers
##############################################################

set -e

LOCAL_DIR="/Users/moky/codes/lehome-challenge/scripts/mihomo-setup"

# Server configurations (mirroring sync.sh)
declare -A SERVER_USER
declare -A SERVER_IP
declare -A SERVER_PORT
declare -A SERVER_NAME
declare -A SERVER_NEEDS_PROXY
declare -A SERVER_HAS_SUDO

# Server 1: ZJUICI remote server (has sudo)
SERVER_USER[zjuici]="admin"
SERVER_IP[zjuici]="pool.zjuici.com"
SERVER_PORT[zjuici]="31329"
SERVER_NAME[zjuici]="ZJUICI Remote"
SERVER_NEEDS_PROXY[zjuici]="yes"
SERVER_HAS_SUDO[zjuici]="yes"

# Server 2: Local LAN server 102 (NO sudo)
SERVER_USER[lan102]="hls"
SERVER_IP[lan102]="192.168.3.102"
SERVER_PORT[lan102]="22"
SERVER_NAME[lan102]="LAN 102"
SERVER_NEEDS_PROXY[lan102]="yes"
SERVER_HAS_SUDO[lan102]="no"

# Server 3: Local LAN server 103 (NO sudo)
SERVER_USER[lan103]="hls"
SERVER_IP[lan103]="192.168.3.103"
SERVER_PORT[lan103]="22"
SERVER_NAME[lan103]="LAN 103"
SERVER_NEEDS_PROXY[lan103]="yes"
SERVER_HAS_SUDO[lan103]="no"

# Server 4: Local LAN server 104 (NO sudo)
SERVER_USER[lan104]="hls"
SERVER_IP[lan104]="192.168.3.104"
SERVER_PORT[lan104]="22"
SERVER_NAME[lan104]="LAN 104"
SERVER_NEEDS_PROXY[lan104]="yes"
SERVER_HAS_SUDO[lan104]="no"

# Server 5: Local LAN server 206 (NO sudo)
SERVER_USER[lan206]="hls"
SERVER_IP[lan206]="192.168.3.206"
SERVER_PORT[lan206]="22"
SERVER_NAME[lan206]="LAN 206"
SERVER_NEEDS_PROXY[lan206]="yes"
SERVER_HAS_SUDO[lan206]="no"

# Server 6: Local LAN server 120 (NO sudo)
SERVER_USER[lan120]="hls"
SERVER_IP[lan120]="192.168.3.120"
SERVER_PORT[lan120]="22"
SERVER_NAME[lan120]="LAN 120"
SERVER_NEEDS_PROXY[lan120]="yes"
SERVER_HAS_SUDO[lan120]="no"

# Server 7: Local LAN server 153 - lan153 (has sudo)
SERVER_USER[lan153]="brl"
SERVER_IP[lan153]="192.168.3.153"
SERVER_PORT[lan153]="22"
SERVER_NAME[lan153]="lan153"
SERVER_NEEDS_PROXY[lan153]="yes"
SERVER_HAS_SUDO[lan153]="yes"

# Available servers list
SERVERS=(zjuici lan102 lan103 lan104 lan206 lan120 lan153)
DEFAULT_SERVER="zjuici"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

# Show available servers
show_servers() {
    echo "Available servers:"
    echo ""
    for server in "${SERVERS[@]}"; do
        local user="${SERVER_USER[$server]}"
        local host="${SERVER_IP[$server]}"
        local port="${SERVER_PORT[$server]}"
        local name="${SERVER_NAME[$server]}"
        local needs_proxy="${SERVER_NEEDS_PROXY[$server]}"
        local has_sudo="${SERVER_HAS_SUDO[$server]}"

        printf "  ${BLUE}%-8s${NC} %s\n" "$server" "$name"
        printf "           ${user}@${host}:${port}"
        if [[ "$needs_proxy" == "yes" ]]; then
            printf " ${GREEN}[needs mihomo]${NC}"
        fi
        if [[ "$has_sudo" == "no" ]]; then
            printf " ${YELLOW}[no sudo, user install]${NC}"
        fi
        printf "\n"
    done
    echo ""
    printf "Default: ${GREEN}%s${NC}\n" "$DEFAULT_SERVER"
    echo ""
}

# Deploy to a single server
deploy_to_server() {
    local server="$1"
    local user="${SERVER_USER[$server]}"
    local host="${SERVER_IP[$server]}"
    local port="${SERVER_PORT[$server]}"
    local name="${SERVER_NAME[$server]}"

    print_header "Deploying to ${name} ($server)"

    # Check if config exists
    if [[ ! -f "$LOCAL_DIR/config.yaml" ]]; then
        log_error "config.yaml not found at $LOCAL_DIR/config.yaml"
        exit 1
    fi

    log_info "📦 Copying setup script and config to server..."

    local has_sudo="${SERVER_HAS_SUDO[$server]}"
    local setup_script="setup_mihomo.sh"

    if [[ "$has_sudo" == "no" ]]; then
        setup_script="setup_mihomo_nosudo.sh"
        log_info "📋 Using non-sudo installation script"
    else
        setup_script="setup_mihomo.sh"
        log_info "📋 Using sudo installation script"
    fi

    # Download mihomo binary locally (for servers without external access)
    local arch="amd64"
    local mihomo_version="v1.19.20"
    local mihomo_url="https://github.com/MetaCubeX/mihomo/releases/download/${mihomo_version}/mihomo-linux-${arch}-compatible-${mihomo_version}.gz"
    local local_binary="$LOCAL_DIR/.mihomo_binary_${arch}"
    local local_mmdb="$LOCAL_DIR/.country.mmdb"

    log_info "📥 Downloading mihomo ${mihomo_version} binary locally..."
    if [[ ! -f "$local_binary" ]]; then
        log_info "  URL: $mihomo_url"
        if curl -L "$mihomo_url" -o "${local_binary}.gz"; then
            gunzip -c "${local_binary}.gz" > "$local_binary"
            rm -f "${local_binary}.gz"
            chmod +x "$local_binary"
            log_info "✅ Downloaded to $local_binary"
        else
            log_error "Failed to download mihomo binary"
            exit 1
        fi
    else
        log_info "✅ Using cached binary: $local_binary"
    fi

    # Download MMDB database locally (required for mihomo to start)
    log_info "📥 Downloading GeoIP MMDB database locally..."
    if [[ ! -f "$local_mmdb" ]]; then
        if curl -L "https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/country.mmdb" -o "$local_mmdb"; then
            log_info "✅ Downloaded to $local_mmdb"
        else
            log_warn "Failed to download MMDB, will try to download on server"
        fi
    else
        log_info "✅ Using cached MMDB: $local_mmdb"
    fi

    # Copy files to server
    scp -P "$port" "$LOCAL_DIR/$setup_script" "${user}@${host}:/tmp/"
    scp -P "$port" "$LOCAL_DIR/config.yaml" "${user}@${host}:/tmp/mihomo_config.yaml"
    scp -P "$port" "$local_binary" "${user}@${host}:/tmp/mihomo_${arch}"
    if [[ -f "$local_mmdb" ]]; then
        scp -P "$port" "$local_mmdb" "${user}@${host}:/tmp/country.mmdb"
    fi

    log_info "🚀 Running setup on server..."

    if [[ "$has_sudo" == "no" ]]; then
        # Non-sudo installation
        ssh -p "$port" "${user}@${host}" "bash /tmp/${setup_script} /tmp/mihomo_config.yaml"
    else
        # Sudo installation
        log_info "🔐 You may be asked for your sudo password..."
        ssh -t -p "$port" "${user}@${host}" "sudo bash /tmp/${setup_script} /tmp/mihomo_config.yaml"
    fi

    log_info "✅ Deployment complete!"

    # Copy and run test script
    log_info "🧪 Running proxy tests..."
    scp -P "$port" "$LOCAL_DIR/test_mihomo.sh" "${user}@${host}:/tmp/" 2>/dev/null || true
    ssh -p "$port" "${user}@${host}" "bash /tmp/test_mihomo.sh" 2>/dev/null || log_warn "Test script failed - you can run it manually"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Next Steps for $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1️⃣  Test proxy on server:"
    echo "   ssh -p $port ${user}@${host} 'bash /tmp/test_mihomo.sh'"
    echo ""
    echo "2️⃣  Set up SSH tunnel for web UI:"
    echo "   ssh -L 9099:127.0.0.1:9090 -L 7897:127.0.0.1:7897 -p $port ${user}@${host}"
    echo ""
    echo "3️⃣  Open web dashboard:"
    echo "   http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099"
    echo ""
    echo "4️⃣  Use on server terminal:"
    echo "   ssh -p $port ${user}@${host}"
    echo "   source proxy_on"
    echo "   curl https://api.ipify.org"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Deploy to all servers
deploy_to_all() {
    print_header "Deploying Mihomo to ALL Servers"

    for server in "${SERVERS[@]}"; do
        echo ""
        deploy_to_server "$server"
        echo ""
        read -p "Press Enter to continue to next server (or Ctrl+C to stop)..."
    done

    print_header "All Deployments Complete!"

    echo "Summary of deployments:"
    for server in "${SERVERS[@]}"; do
        local user="${SERVER_USER[$server]}"
        local host="${SERVER_IP[$server]}"
        local port="${SERVER_PORT[$server]}"
        local name="${SERVER_NAME[$server]}"

        printf "  ✅ ${GREEN}%-8s${NC} %s (${user}@${host}:${port})\n" "$server" "$name"
    done
    echo ""
}

# Show help
show_help() {
    cat << 'EOF'
Mihomo Multi-Server Deployment Script

Deploy mihomo proxy service to multiple servers for external access.

Usage: ./deploy_to_server.sh [server] [command]
       ./deploy_to_server.sh [command] [server]

Servers:
  zjuici  - ZJUICI remote server (admin@pool.zjuici.com:31329) [has sudo]
  lan102  - Local LAN 192.168.3.102 (hls@192.168.3.102:22) [no sudo]
  lan103  - Local LAN 192.168.3.103 (hls@192.168.3.103:22) [no sudo]
  lan104  - Local LAN 192.168.3.104 (hls@192.168.3.104:22) [no sudo]
  lan206  - Local LAN 192.168.3.206 (hls@192.168.3.206:22) [no sudo]
  lan120  - Local LAN 192.168.3.120 (hls@192.168.3.120:22) [no sudo]
  lan153  - Local LAN 192.168.3.153 (brl@192.168.3.153:22) [has sudo]

Commands:
  deploy  - Deploy mihomo to specified server (default)
  all     - Deploy to ALL servers
  list    - List all servers with their configurations
  help    - Show this help message

Examples:
  ./deploy_to_server.sh                    # Deploy to default server (zjuici)
  ./deploy_to_server.sh deploy lan102      # Deploy to lan102
  ./deploy_to_server.sh lan103 deploy      # Deploy to lan103
  ./deploy_to_server.sh all                # Deploy to all servers
  ./deploy_to_server.sh list               # List all servers

What Mihomo Does:
  - Provides HTTP/SOCKS5 proxy on port 7897
  - Enables external access for your LAN servers
  - Allows pip, git, and other tools to work behind NAT/firewall
  - Web UI dashboard for managing proxy rules

After Deployment:
  1. Test proxy: ssh server 'curl -x http://127.0.0.1:7897 https://api.ipify.org'
  2. Enable proxy: ssh server 'source proxy_on'
  3. Use in sync: Edit sync.sh to enable proxy for each server

EOF
}

# Main script
SERVER=""
COMMAND=""

# Parse arguments
if [[ $# -eq 0 ]]; then
    # No arguments - use default server
    SERVER="$DEFAULT_SERVER"
    COMMAND="deploy"
elif [[ $# -eq 1 ]]; then
    # One argument
    if [[ "$1" =~ ^(zjuici|lan102|lan103|lan104|lan206|lan120|lan153)$ ]]; then
        SERVER="$1"
        COMMAND="deploy"
    elif [[ "$1" == "all" ]]; then
        COMMAND="all"
    elif [[ "$1" == "list" ]]; then
        COMMAND="list"
    elif [[ "$1" == "help" ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        COMMAND="help"
    else
        COMMAND="deploy"
        SERVER="$1"
    fi
elif [[ $# -eq 2 ]]; then
    # Two arguments - could be "server command" or "command server"
    if [[ "$1" =~ ^(zjuici|lan102|lan103|lan104|lan206|lan120|lan153)$ ]]; then
        SERVER="$1"
        COMMAND="$2"
    elif [[ "$2" =~ ^(zjuici|lan102|lan103|lan104|lan206|lan120|lan153)$ ]]; then
        COMMAND="$1"
        SERVER="$2"
    else
        log_error "Invalid arguments"
        show_help
        exit 1
    fi
fi

# Execute command
case "${COMMAND:-help}" in
    "deploy")
        if [[ -z "$SERVER" ]]; then
            log_error "No server specified"
            show_servers
            exit 1
        fi
        deploy_to_server "$SERVER"
        ;;
    "all")
        deploy_to_all
        ;;
    "list")
        print_header "Mihomo Server Configurations"
        show_servers
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
