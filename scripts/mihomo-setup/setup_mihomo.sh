#!/bin/bash

##############################################################
# Mihomo (Clash Meta) Automated Setup Script
# For Ubuntu/Debian systems
#
# Usage: sudo bash setup_mihomo.sh [config_file]
#   config_file: Optional path to custom config.yaml
##############################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/mihomo"
SERVICE_NAME="mihomo"
GITHUB_REPO="MetaCubeX/mihomo"
VERSION="v1.19.20"  # Specific version, can be updated

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

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

detect_architecture() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            echo "amd64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        armv7l)
            echo "armv7"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
}

install_dependencies() {
    log_info "Installing required dependencies..."
    apt-get update -qq
    apt-get install -y curl wget gzip systemd
}

download_mihomo() {
    local arch=$1
    local filename="mihomo-linux-${arch}"

    log_info "Downloading Mihomo ${VERSION} for ${arch}..."

    local download_url="https://github.com/${GITHUB_REPO}/releases/download/${VERSION}/${filename}.gz"

    # Download to temp file
    local tmp_file="/tmp/mihomo.gz"

    if ! curl -L "$download_url" -o "$tmp_file"; then
        log_error "Failed to download Mihomo from GitHub"
        log_info "Please check your internet connection and try again"
        exit 1
    fi

    # Extract and install
    log_info "Extracting and installing..."
    gunzip -c "$tmp_file" > "${INSTALL_DIR}/mihomo"
    chmod +x "${INSTALL_DIR}/mihomo"
    rm -f "$tmp_file"

    log_info "Mihomo installed to ${INSTALL_DIR}/mihomo"
}

create_directories() {
    log_info "Creating directory structure..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "${INSTALL_DIR}/configs"
    mkdir -p "${INSTALL_DIR}/logs"
}

setup_config() {
    local config_source="${1:-}"

    if [[ -n "$config_source" && -f "$config_source" ]]; then
        log_info "Using provided config file: $config_source"
        cp "$config_source" "${INSTALL_DIR}/config.yaml"
    elif [[ -f "${INSTALL_DIR}/config.yaml" ]]; then
        log_info "Using existing config at ${INSTALL_DIR}/config.yaml"
    else
        log_error "No config.yaml found!"
        log_info "Please provide a config file: sudo bash $0 /path/to/config.yaml"
        log_info "Or place config.yaml at ${INSTALL_DIR}/config.yaml"
        exit 1
    fi

    # Ensure localhost-only external controller for security
    log_info "Configuring external-controller for localhost only..."
    sed -i 's/external-controller:.*$/external-controller: 127.0.0.1:9090/g' "${INSTALL_DIR}/config.yaml" 2>/dev/null || true

    # Ensure mixed-port is set to 7897
    log_info "Configuring mixed-port to 7897..."
    sed -i 's/mixed-port:.*$/mixed-port: 7897/g' "${INSTALL_DIR}/config.yaml" 2>/dev/null || true

    # Validate YAML syntax (basic check)
    if ! grep -q "mixed-port:" "${INSTALL_DIR}/config.yaml"; then
        log_warn "Config validation warning: mixed-port not found in config.yaml"
    fi

    log_info "Configuration setup complete"
}

install_systemd_service() {
    log_info "Installing systemd service..."

    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << 'EOF'
[Unit]
Description=Mihomo Proxy Service
After=network.target

[Service]
Type=simple
User=nobody
Group=nogroup
ExecStart=/opt/mihomo/mihomo -d /opt/mihomo -f /opt/mihomo/config.yaml
Restart=on-failure
RestartSec=5s
StandardOutput=append:/opt/mihomo/logs/mihomo.log
StandardError=append:/opt/mihomo/logs/mihomo.log

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"

    log_info "Systemd service installed and enabled"
}

install_management_scripts() {
    log_info "Installing management scripts..."

    # Install mihomoctl
    cat > "${INSTALL_DIR}/mihomoctl" << 'EOF'
#!/bin/bash
# Mihomo management script

case "${1:-}" in
    start)
        sudo systemctl start mihomo
        echo "Mihomo started"
        ;;
    stop)
        sudo systemctl stop mihomo
        echo "Mihomo stopped"
        ;;
    restart)
        sudo systemctl restart mihomo
        echo "Mihomo restarted"
        ;;
    status)
        systemctl status mihomo
        ;;
    reload)
        sudo systemctl reload mihomo 2>/dev/null || sudo systemctl restart mihomo
        echo "Mihomo configuration reloaded"
        ;;
    logs)
        sudo journalctl -u mihomo -f
        ;;
    *)
        echo "Usage: mihomoctl {start|stop|restart|status|reload|logs}"
        exit 1
        ;;
esac
EOF
    chmod +x "${INSTALL_DIR}/mihomoctl"

    # Install proxy_on.sh
    cat > "${INSTALL_DIR}/proxy_on.sh" << 'EOF'
#!/bin/bash
# Enable proxy in current terminal

export http_proxy="http://127.0.0.1:7897"
export https_proxy="http://127.0.0.1:7897"
export all_proxy="socks5://127.0.0.1:7897"
export no_proxy="localhost,127.0.0.1,::1"

echo "Proxy enabled for current terminal session"
echo "HTTP_PROXY=$http_proxy"
echo "HTTPS_PROXY=$https_proxy"
EOF
    chmod +x "${INSTALL_DIR}/proxy_on.sh"

    # Install proxy_off.sh
    cat > "${INSTALL_DIR}/proxy_off.sh" << 'EOF'
#!/bin/bash
# Disable proxy in current terminal

unset http_proxy
unset https_proxy
unset all_proxy
unset no_proxy

echo "Proxy disabled for current terminal session"
EOF
    chmod +x "${INSTALL_DIR}/proxy_off.sh"

    # Create symlinks for easy access
    ln -sf "${INSTALL_DIR}/mihomoctl" "/usr/local/bin/mihomoctl"
    ln -sf "${INSTALL_DIR}/proxy_on.sh" "/usr/local/bin/proxy_on"
    ln -sf "${INSTALL_DIR}/proxy_off.sh" "/usr/local/bin/proxy_off"

    log_info "Management scripts installed"
    log_info "  - mihomoctl: Service management"
    log_info "  - proxy_on: Enable terminal proxy"
    log_info "  - proxy_off: Disable terminal proxy"
}

start_service() {
    log_info "Starting Mihomo service..."
    systemctl start "$SERVICE_NAME"

    # Wait a moment for service to start
    sleep 2

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Mihomo service started successfully!"
    else
        log_error "Failed to start Mihomo service"
        log_info "Check logs with: journalctl -u mihomo -n 50"
        exit 1
    fi
}

print_success_message() {
    cat << 'SUCCESS'

╔════════════════════════════════════════════════════════════╗
║           Mihomo Installation Completed Successfully!      ║
╚════════════════════════════════════════════════════════════╝

Service Commands:
  mihomoctl start    - Start Mihomo service
  mihomoctl stop     - Stop Mihomo service
  mihomoctl restart  - Restart Mihomo service
  mihomoctl status   - Check service status
  mihomoctl logs     - View service logs

Terminal Proxy:
  proxy_on           - Enable proxy for current terminal
  proxy_off          - Disable proxy for current terminal

Configuration:
  Config file:       /opt/mihomo/config.yaml
  Edit and reload:   mihomoctl reload

Web Dashboard (requires SSH tunnel):
  Local terminal:    ssh -L 9099:127.0.0.1:9090 user@server
  Dashboard URL:     http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099

Ports:
  Proxy (HTTP/SOCKS): 127.0.0.1:7897
  API:                127.0.0.1:9090

For troubleshooting, check:
  journalctl -u mihomo -f

SUCCESS
}

# Main execution
main() {
    local config_file="${1:-}"

    log_info "Starting Mihomo setup..."
    log_info "Installation directory: $INSTALL_DIR"

    # Run installation steps
    check_root
    install_dependencies
    create_directories
    download_mihomo "$(detect_architecture)"
    setup_config "$config_file"
    install_systemd_service
    install_management_scripts
    start_service

    print_success_message
}

# Run main function
main "$@"
