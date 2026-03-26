#!/bin/bash

##############################################################
# Test Mihomo on Remote Servers
# Usage: ./test_remote.sh [server]
##############################################################

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Get server info
get_server_info() {
    local server="$1"
    case "$server" in
        lan102)
            echo "hls 192.168.3.102 22 LAN 102"
            ;;
        lan103)
            echo "hls 192.168.3.103 22 LAN 103"
            ;;
        lan104)
            echo "hls 192.168.3.104 22 LAN 104"
            ;;
        lan206)
            echo "hls 192.168.3.206 22 LAN 206"
            ;;
        lan120)
            echo "hls 192.168.3.120 22 LAN 120"
            ;;
        lan153)
            echo "brl 192.168.3.153 22 lan153"
            ;;
        zjuici)
            echo "admin pool.zjuici.com 31329 ZJUICI Remote"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Test a single server
test_server() {
    local server="$1"
    local info=$(get_server_info "$server")

    if [[ -z "$info" ]]; then
        echo "Unknown server: $server"
        echo "Available: lan102, lan103, lan104, lan206, lan120, lan153, zjuici"
        exit 1
    fi

    local user=$(echo "$info" | cut -d' ' -f1)
    local host=$(echo "$info" | cut -d' ' -f2)
    local port=$(echo "$info" | cut -d' ' -f3)
    local name=$(echo "$info" | cut -d' ' -f4-)

    echo ""
    echo "=========================================="
    echo "  Testing ${name} ($server)"
    echo "=========================================="
    echo ""
    echo "Server: ${user}@${host}:${port}"
    echo ""

    # Copy test script
    scp -P "$port" "$LOCAL_DIR/test_mihomo.sh" "${user}@${host}:/tmp/" 2>/dev/null

    # Run test
    ssh -p "$port" "${user}@${host}" "bash /tmp/test_mihomo.sh"
}

# Test all servers
test_all() {
    echo "=========================================="
    echo "  Testing ALL Servers"
    echo "=========================================="

    for server in lan102 lan103 lan104 lan206 lan120 lan153 zjuici; do
        echo ""
        test_server "$server"
        echo ""
        echo "----------------------------------------"
    done
}

# Show help
show_help() {
    cat << 'EOF'
Mihomo Remote Test Script

Usage: ./test_remote.sh [server]
       ./test_remote.sh all

Servers:
  lan102, lan103, lan104, lan206, lan120, lan153, zjuici

Examples:
  ./test_remote.sh lan102     # Test lan102
  ./test_remote.sh all        # Test all servers

EOF
}

# Main
case "${1:-help}" in
    all)
        test_all
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        test_server "$1"
        ;;
esac
