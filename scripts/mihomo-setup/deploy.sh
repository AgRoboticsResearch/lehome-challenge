#!/bin/bash

##############################################################
# Quick Deploy Script
# Copies setup script and config to remote server and runs it
##############################################################

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo "Usage: $0 <server> <config_file> [ssh_port]"
    echo ""
    echo "Example: $0 admin@192.168.1.100 my-config.yaml 22"
    echo ""
    echo "Arguments:"
    echo "  server      SSH connection string (user@host)"
    echo "  config_file Path to your config.yaml"
    echo "  ssh_port    SSH port (default: 22)"
    exit 1
}

# Check arguments
if [[ $# -lt 2 ]]; then
    usage
fi

SERVER="$1"
CONFIG_FILE="$2"
SSH_PORT="${3:-22}"

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo -e "${GREEN}Mihomo Quick Deploy${NC}"
echo "Server: $SERVER"
echo "Config: $CONFIG_FILE"
echo "SSH Port: $SSH_PORT"
echo ""

# Copy files to server
echo "Copying files to server..."
scp -P "$SSH_PORT" "$0" "${SERVER}:/tmp/setup_mihomo.sh"
scp -P "$SSH_PORT" "$CONFIG_FILE" "${SERVER}:/tmp/mihomo_config.yaml"

# Run setup on server
echo ""
echo "Running setup on server..."
ssh -p "$SSH_PORT" "$SERVER" "sudo bash /tmp/setup_mihomo.sh /tmp/mihomo_config.yaml"

echo ""
echo -e "${GREEN}Deploy completed!${NC}"
echo ""
echo "To test the proxy:"
echo "  ssh -p $SSH_PORT $SERVER 'curl -x http://127.0.0.1:7897 https://api.ipify.org'"
echo ""
echo "To set up SSH tunnel for web UI:"
echo "  ssh -L 9099:127.0.0.1:9090 -p $SSH_PORT $SERVER"
echo "  Then open: http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099"
