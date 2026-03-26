#!/bin/zsh

##############################################################
# Multi-Server Disk Usage Checker
# Works without sudo/ncdu - uses basic df/du commands
#
# Usage: ./scripts/disk_usage.sh [server] [--detail] [--top]
##############################################################

set -e

# Server configurations (mirroring sync.sh)
declare -A SERVER_USER
declare -A SERVER_IP
declare -A SERVER_PORT
declare -A SERVER_NAME
declare -A SERVER_DIR

# Server 1: ZJUICI remote server
SERVER_USER[zjuici]="admin"
SERVER_IP[zjuici]="pool.zjuici.com"
SERVER_PORT[zjuici]="31329"
SERVER_DIR[zjuici]="/home/admin/codes/lehome-challenge"
SERVER_NAME[zjuici]="ZJUICI Remote"

# Server 2: Local LAN server 102
SERVER_USER[lan102]="hls"
SERVER_IP[lan102]="192.168.3.102"
SERVER_PORT[lan102]="22"
SERVER_DIR[lan102]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan102]="LAN 102"

# Server 3: Local LAN server 103
SERVER_USER[lan103]="hls"
SERVER_IP[lan103]="192.168.3.103"
SERVER_PORT[lan103]="22"
SERVER_DIR[lan103]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan103]="LAN 103"

# Server 4: Local LAN server 104
SERVER_USER[lan104]="hls"
SERVER_IP[lan104]="192.168.3.104"
SERVER_PORT[lan104]="22"
SERVER_DIR[lan104]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan104]="LAN 104"

# Server 5: Local LAN server 206
SERVER_USER[lan206]="hls"
SERVER_IP[lan206]="192.168.3.206"
SERVER_PORT[lan206]="22"
SERVER_DIR[lan206]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan206]="LAN 206"

# Server 6: Local LAN server 120
SERVER_USER[lan120]="hls"
SERVER_IP[lan120]="192.168.3.120"
SERVER_PORT[lan120]="22"
SERVER_DIR[lan120]="/home/hls/codes/lehome-challenge"
SERVER_NAME[lan120]="LAN 120"

SERVERS=(zjuici lan102 lan103 lan104 lan206 lan120)
DEFAULT_SERVER="zjuici"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

# Format bytes to human readable
format_bytes() {
    local bytes=$1
    if [[ $bytes -lt 1024 ]]; then
        echo "${bytes}B"
    elif [[ $bytes -lt 1048576 ]]; then
        echo "$((bytes / 1024))K"
    elif [[ $bytes -lt 1073741824 ]]; then
        echo "$((bytes / 1048576))M"
    elif [[ $bytes -lt 1099511627776 ]]; then
        echo "$((bytes / 1073741824))G"
    else
        echo "$((bytes / 1099511627776))T"
    fi
}

# Check disk usage on a single server
check_server() {
    local server="$1"
    local show_detail="$2"
    local show_top="$3"
    local user="${SERVER_USER[$server]}"
    local host="${SERVER_IP[$server]}"
    local port="${SERVER_PORT[$server]}"
    local name="${SERVER_NAME[$server]}"
    local dir="${SERVER_DIR[$server]}"

    printf "${BLUE}%-8s${NC} %s " "$server" "$name"

    # Check SSH connectivity
    if ! ssh -p "$port" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${user}@${host}" "echo -n" 2>/dev/null; then
        printf "${RED}[OFFLINE]${NC}\n"
        return 1
    fi

    # Get disk usage
    local df_output=$(ssh -p "$port" "${user}@${host}" "df -h ~ 2>/dev/null | tail -1" 2>/dev/null)

    if [[ -n "$df_output" ]]; then
        local usage_percent=$(echo "$df_output" | awk '{print $5}' | sed 's/%//')
        local available=$(echo "$df_output" | awk '{print $4}')
        local used=$(echo "$df_output" | awk '{print $3}')
        local total=$(echo "$df_output" | awk '{print $2}')

        if [[ $usage_percent -ge 90 ]]; then
            printf "${RED}[%s%% used]${NC} " "$usage_percent"
        elif [[ $usage_percent -ge 70 ]]; then
            printf "${YELLOW}[%s%% used]${NC} " "$usage_percent"
        else
            printf "${GREEN}[%s%% used]${NC} " "$usage_percent"
        fi

        printf "%s available\n" "$available"
    else
        printf "${GRAY}[No data]${NC}\n"
    fi

    # Show detailed info if requested
    if [[ "$show_detail" == "yes" ]]; then
        echo ""
        printf "  ${CYAN}Disk Usage Details:${NC}\n"

        # Home directory breakdown
        local du_output=$(ssh -p "$port" "${user}@${host}" "du -sh ~/* 2>/dev/null | sort -rh | head -20" 2>/dev/null)
        if [[ -n "$du_output" ]]; then
            printf "  ${GRAY}Top directories in home:${NC}\n"
            echo "$du_output" | sed 's/^/    /'
        fi

        # Project directory breakdown
        if ssh -p "$port" "${user}@${host}" "test -d $dir" 2>/dev/null; then
            local project_du=$(ssh -p "$port" "${user}@${host}" "du -sh $dir/* 2>/dev/null | sort -rh | head -15" 2>/dev/null)
            if [[ -n "$project_du" ]]; then
                printf "\n  ${CYAN}Project directories ($dir):${NC}\n"
                echo "$project_du" | sed 's/^/    /'
            fi
        fi

        # System disk info
        local df_all=$(ssh -p "$port" "${user}@${host}" "df -h 2>/dev/null | grep -v tmpfs" 2>/dev/null)
        if [[ -n "$df_all" ]]; then
            printf "\n  ${CYAN}All filesystems:${NC}\n"
            echo "$df_all" | sed 's/^/    /'
        fi
        echo ""
    fi

    # Show top disk consumers if requested
    if [[ "$show_top" == "yes" ]]; then
        echo ""
        printf "  ${CYAN}Top 10 disk consumers in home:${NC}\n"
        local top_dirs=$(ssh -p "$port" "${user}@${host}" "du -sh ~/* 2>/dev/null | sort -rh | head -10" 2>/dev/null)
        if [[ -n "$top_dirs" ]]; then
            local rank=1
            echo "$top_dirs" | while read -r size path; do
                # Get basename using parameter expansion instead of basename command
                local filename="${path:t}"
                printf "    ${GREEN}%2d${NC}. ${GRAY}%10s${NC} %s\n" "$rank" "$size" "$filename"
                rank=$((rank + 1))
            done
        fi
        echo ""
    fi

    return 0
}

# Show help
show_help() {
    cat << 'EOF'
Multi-Server Disk Usage Checker

Check disk usage across all LeHome Challenge servers without sudo/ncdu.

Usage: ./scripts/disk_usage.sh [server] [options]

Servers:
  zjuici  - ZJUICI remote server (default)
  lan102  - Local LAN 192.168.3.102
  lan103  - Local LAN 192.168.3.103
  lan104  - Local LAN 192.168.3.104
  lan206  - Local LAN 192.168.3.206
  lan120  - Local LAN 192.168.3.120

Options:
  --detail, -d    Show detailed disk usage breakdown
  --top, -t      Show top disk consumers
  --help, -h     Show this help message

Examples:
  ./scripts/disk_usage.sh                 # Check all servers
  ./scripts/disk_usage.sh lan102          # Check lan102 only
  ./scripts/disk_usage.sh lan102 --detail # Detailed info for lan102
  ./scripts/disk_usage.sh --top           # Show top consumers on all servers

Output Legend:
  [GREEN]   Disk usage < 70%
  [YELLOW]  Disk usage 70-90%
  [RED]     Disk usage > 90%
EOF
}

# Main script
SERVER=""
SHOW_DETAIL="no"
SHOW_TOP="no"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --detail|-d)
            SHOW_DETAIL="yes"
            ;;
        --top|-t)
            SHOW_TOP="yes"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        zjuici|lan102|lan103|lan104|lan206|lan120)
            SERVER="$arg"
            ;;
    esac
done

# Use default server if none specified
if [[ -z "$SERVER" ]]; then
    # Check all servers
    echo "=========================================="
    echo "  Disk Usage Check - All Servers"
    echo "=========================================="
    echo ""

    for s in "${SERVERS[@]}"; do
        check_server "$s" "$SHOW_DETAIL" "$SHOW_TOP"
    done

    echo ""
    echo "Use --detail for more information or --top for top consumers"
else
    # Check specific server
    echo "=========================================="
    echo "  Disk Usage Check - ${SERVER_NAME[$SERVER]}"
    echo "=========================================="
    echo ""
    check_server "$SERVER" "$SHOW_DETAIL" "$SHOW_TOP"
fi
