#!/bin/zsh

##############################################################
# Check Mihomo Status on All Servers
# This script checks the status of mihomo on all configured servers
##############################################################

set -e

# Server configurations (mirroring sync.sh and deploy_to_server.sh)
declare -A SERVER_USER
declare -A SERVER_IP
declare -A SERVER_PORT
declare -A SERVER_NAME
declare -A SERVER_NEEDS_PROXY

# Server 1: ZJUICI remote server
SERVER_USER[zjuici]="admin"
SERVER_IP[zjuici]="pool.zjuici.com"
SERVER_PORT[zjuici]="31329"
SERVER_NAME[zjuici]="ZJUICI Remote"
SERVER_NEEDS_PROXY[zjuici]="yes"

# Server 2: Local LAN server 102
SERVER_USER[lan102]="hls"
SERVER_IP[lan102]="192.168.3.102"
SERVER_PORT[lan102]="22"
SERVER_NAME[lan102]="LAN 102"
SERVER_NEEDS_PROXY[lan102]="yes"

# Server 3: Local LAN server 103
SERVER_USER[lan103]="hls"
SERVER_IP[lan103]="192.168.3.103"
SERVER_PORT[lan103]="22"
SERVER_NAME[lan103]="LAN 103"
SERVER_NEEDS_PROXY[lan103]="yes"

# Server 4: Local LAN server 104
SERVER_USER[lan104]="hls"
SERVER_IP[lan104]="192.168.3.104"
SERVER_PORT[lan104]="22"
SERVER_NAME[lan104]="LAN 104"
SERVER_NEEDS_PROXY[lan104]="yes"

# Server 5: Local LAN server 206
SERVER_USER[lan206]="hls"
SERVER_IP[lan206]="192.168.3.206"
SERVER_PORT[lan206]="22"
SERVER_NAME[lan206]="LAN 206"
SERVER_NEEDS_PROXY[lan206]="yes"

# Server 6: Local LAN server 120
SERVER_USER[lan120]="hls"
SERVER_IP[lan120]="192.168.3.120"
SERVER_PORT[lan120]="22"
SERVER_NAME[lan120]="LAN 120"
SERVER_NEEDS_PROXY[lan120]="yes"

SERVERS=(zjuici lan102 lan103 lan104 lan206 lan120)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Check status on a single server
check_server() {
    local server="$1"
    local user="${SERVER_USER[$server]}"
    local host="${SERVER_IP[$server]}"
    local port="${SERVER_PORT[$server]}"
    local name="${SERVER_NAME[$server]}"

    printf "${BLUE}%-8s${NC} %s " "$server" "$name"

    # Check if SSH connection works
    if ! ssh -p "$port" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${user}@${host}" "echo -n" 2>/dev/null; then
        printf "${RED}[OFFLINE]${NC}\n"
        return 1
    fi

    # Check if mihomo is installed
    local mihomo_path=$(ssh -p "$port" "${user}@${host}" "test -f ~/.mihomo/mihomo && echo ~/.mihomo/mihomo || test -f /opt/mihomo/mihomo && echo /opt/mihomo/mihomo || echo ''" 2>/dev/null)

    if [[ -z "$mihomo_path" ]]; then
        printf "${YELLOW}[NOT INSTALLED]${NC}\n"
        return 0
    fi

    # Check if service is running (user or system service)
    local svc_status=$(ssh -p "$port" "${user}@${host}" "systemctl --user is-active mihomo 2>/dev/null || systemctl is-active mihomo 2>/dev/null || echo 'unknown'" 2>/dev/null)

    if [[ "$svc_status" == "active" ]]; then
        printf "${GREEN}[RUNNING]${NC} "

        # Check if proxy is responding
        local ip=$(ssh -p "$port" "${user}@${host}" "curl -s -x http://127.0.0.1:7897 --connect-timeout 3 https://api.ipify.org 2>/dev/null" 2>/dev/null)

        if [[ -n "$ip" ]]; then
            printf "${GREEN}→ $ip${NC}\n"
        else
            printf "${GRAY}(no external IP)${NC}\n"
        fi
    elif [[ "$status" == "inactive" ]] || [[ "$status" == "unknown" ]]; then
        printf "${YELLOW}[STOPPED]${NC}\n"
    else
        printf "${RED}[$status]${NC}\n"
    fi

    return 0
}

# Show detailed status for a server
show_details() {
    local server="$1"
    local user="${SERVER_USER[$server]}"
    local host="${SERVER_IP[$server]}"
    local port="${SERVER_PORT[$server]}"
    local name="${SERVER_NAME[$server]}"

    echo ""
    echo "=========================================="
    echo "  $name ($server)"
    echo "=========================================="
    echo ""

    echo "Server: ${user}@${host}:${port}"
    echo ""

    # Check service status
    echo "Service Status:"
    ssh -p "$port" "${user}@${host}" "systemctl --user status mihomo --no-pager 2>/dev/null || systemctl status mihomo --no-pager 2>/dev/null || echo 'Service not found'" 2>/dev/null
    echo ""

    # Check recent logs
    echo "Recent Logs:"
    ssh -p "$port" "${user}@${host}" "journalctl --user -u mihomo -n 10 --no-pager 2>/dev/null || journalctl -u mihomo -n 10 --no-pager 2>/dev/null || echo 'No logs available'" 2>/dev/null
    echo ""

    # Test proxy
    echo "Proxy Test:"
    local test_result=$(ssh -p "$port" "${user}@${host}" "curl -s -x http://127.0.0.1:7897 --connect-timeout 5 -w '\nHTTP: %{http_code}\nTime: %{time_total}s\n' https://api.ipify.org 2>&1" 2>/dev/null)
    if [[ -n "$test_result" ]]; then
        echo "$test_result"
    else
        echo "Proxy not responding"
    fi
    echo ""
}

# Main
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Mihomo Status Checker"
        echo ""
        echo "Usage: ./check_status.sh [server] [--details]"
        echo ""
        echo "Examples:"
        echo "  ./check_status.sh              # Check all servers"
        echo "  ./check_status.sh zjuici       # Check zjuici only"
        echo "  ./check_status.sh lan102 --details  # Show detailed info for lan102"
        echo ""
        exit 0
        ;;
    *)
        # Parse arguments
        server=""
        show_details_flag=""

        for arg in "$@"; do
            if [[ "$arg" =~ ^(zjuici|lan102|lan103|lan104|lan206|lan120)$ ]]; then
                server="$arg"
            elif [[ "$arg" == "--details" ]]; then
                show_details_flag="yes"
            fi
        done

        if [[ -n "$server" ]]; then
            if [[ -n "$show_details_flag" ]]; then
                show_details "$server"
            else
                check_server "$server"
            fi
        else
            echo "Mihomo Status Check"
            echo ""
            echo "Checking all servers..."
            echo ""

            for s in "${SERVERS[@]}"; do
                check_server "$s"
            done

            echo ""
            echo "Use --details to see more information:"
            echo "  ./check_status.sh zjuici --details"
        fi
        ;;
esac
