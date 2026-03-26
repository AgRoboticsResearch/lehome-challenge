#!/bin/bash

##############################################################
# Mihomo (Clash Meta) Setup Script - NO SUDO REQUIRED
# Installs mihomo in user directory with user service
#
# Usage: bash setup_mihomo_nosudo.sh [config_file]
##############################################################

set -e
set -u

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
INSTALL_DIR="$HOME/.mihomo"
SERVICE_NAME="mihomo"
GITHUB_REPO="MetaCubeX/mihomo"
VERSION="v1.19.20"

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

install_mihomo() {
    local arch=$1
    local filename="mihomo-linux-${arch}"

    # Check if binary was already copied (by deploy script)
    if [[ -f "/tmp/mihomo_${arch}" ]]; then
        log_info "Using pre-downloaded mihomo binary..."
        mkdir -p "$INSTALL_DIR"
        cp "/tmp/mihomo_${arch}" "${INSTALL_DIR}/mihomo"
        chmod +x "${INSTALL_DIR}/mihomo"
        rm -f "/tmp/mihomo_${arch}"
        log_info "Mihomo installed to ${INSTALL_DIR}/mihomo"
        return
    fi

    log_info "Downloading Mihomo ${VERSION} for ${arch}..."

    local download_url="https://github.com/${GITHUB_REPO}/releases/download/${VERSION}/${filename}.gz"
    local tmp_file="/tmp/mihomo_${arch}.gz"

    # Download to temp file
    if ! curl -L "$download_url" -o "$tmp_file"; then
        log_error "Failed to download Mihomo from GitHub"
        log_warn "If you're behind a firewall, download manually:"
        log_warn "  curl -L '$download_url' -o mihomo.gz"
        log_warn "Then copy to server and run this script again"
        exit 1
    fi

    # Check if file is actually a gzip file
    if ! file "$tmp_file" | grep -q gzip; then
        log_error "Downloaded file is not a valid gzip archive"
        log_error "The server may not have external access or GitHub may be blocked"
        log_warn "Please download mihomo manually and copy it to the server"
        rm -f "$tmp_file"
        exit 1
    fi

    # Create install directory
    mkdir -p "$INSTALL_DIR"

    # Extract and install
    log_info "Extracting and installing to $INSTALL_DIR..."
    gunzip -c "$tmp_file" > "${INSTALL_DIR}/mihomo"
    chmod +x "${INSTALL_DIR}/mihomo"
    rm -f "$tmp_file"

    log_info "Mihomo installed to ${INSTALL_DIR}/mihomo"
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
        log_info "Please provide a config file: bash $0 /path/to/config.yaml"
        exit 1
    fi

    # Ensure localhost-only external controller for security
    log_info "Configuring external-controller for localhost only..."
    sed -i 's/external-controller:.*$/external-controller: 127.0.0.1:9090/g' "${INSTALL_DIR}/config.yaml" 2>/dev/null || true

    # Ensure mixed-port is set to 7897
    log_info "Configuring mixed-port to 7897..."
    sed -i 's/mixed-port:.*$/mixed-port: 7897/g' "${INSTALL_DIR}/config.yaml" 2>/dev/null || true

    # Change port to 7897 if mixed-port is not set (for older configs)
    if ! grep -q "mixed-port:" "${INSTALL_DIR}/config.yaml"; then
        sed -i 's/^port:.*$/port: 7897/g' "${INSTALL_DIR}/config.yaml" 2>/dev/null || true
    fi

    # Copy MMDB database if it was provided (by deploy script)
    if [[ -f "/tmp/country.mmdb" ]]; then
        log_info "Installing GeoIP MMDB database..."
        cp "/tmp/country.mmdb" "${INSTALL_DIR}/country.mmdb"
        rm -f "/tmp/country.mmdb"
    fi

    log_info "Configuration setup complete"
}

install_user_service() {
    log_info "Installing user systemd service..."

    # Create user systemd directory if it doesn't exist
    mkdir -p "$HOME/.config/systemd/user"

    # Create service file
    cat > "$HOME/.config/systemd/user/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Mihomo Proxy Service
After=network.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/mihomo -d $INSTALL_DIR -f $INSTALL_DIR/config.yaml
Restart=on-failure
RestartSec=5s
StandardOutput=append:$INSTALL_DIR/mihomo.log
StandardError=append=$INSTALL_DIR/mihomo.log

[Install]
WantedBy=default.target
EOF

    # Reload systemd for user
    log_info "Reloading user systemd..."
    systemctl --user daemon-reload 2>/dev/null || true

    # Enable service
    systemctl --user enable "$SERVICE_NAME" 2>/dev/null || true

    log_info "User systemd service installed"
}

install_management_scripts() {
    log_info "Installing management scripts..."

    # Create bin directory
    mkdir -p "$HOME/.local/bin"

    # Install mihomoctl
    cat > "$HOME/.local/bin/mihomoctl" << 'EOF'
#!/bin/bash
# Mihomo management script

case "${1:-}" in
    start)
        systemctl --user start mihomo
        echo "Mihomo started"
        ;;
    stop)
        systemctl --user stop mihomo
        echo "Mihomo stopped"
        ;;
    restart)
        systemctl --user restart mihomo
        echo "Mihomo restarted"
        ;;
    status)
        systemctl --user status mihomo
        ;;
    reload)
        systemctl --user restart mihomo
        echo "Mihomo configuration reloaded"
        ;;
    logs)
        journalctl --user -u mihomo -f
        ;;
    *)
        echo "Usage: mihomoctl {start|stop|restart|status|reload|logs}"
        exit 1
        ;;
esac
EOF
    chmod +x "$HOME/.local/bin/mihomoctl"

    # Install proxy_on.sh
    cat > "$HOME/.local/bin/proxy_on" << 'EOF'
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
    chmod +x "$HOME/.local/bin/proxy_on"

    # Install proxy_off.sh
    cat > "$HOME/.local/bin/proxy_off" << 'EOF'
#!/bin/bash
# Disable proxy in current terminal

unset http_proxy
unset https_proxy
unset all_proxy
unset no_proxy

echo "Proxy disabled for current terminal session"
EOF
    chmod +x "$HOME/.local/bin/proxy_off"

    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        echo "" >> "$HOME/.bashrc"
        echo "# Add ~/.local/bin to PATH" >> "$HOME/.bashrc"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.bashrc"
        log_info "Added ~/.local/bin to PATH in .bashrc"
        log_info "Run 'source ~/.bashrc' or restart your shell to use commands"
    fi

    log_info "Management scripts installed to ~/.local/bin/"
}

start_service() {
    log_info "Starting Mihomo service..."

    # Check if DBUS_SESSION_BUS_ADDRESS is set
    if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
        log_warn "DBUS_SESSION_BUS_ADDRESS not set"
        log_info "You may need to run: export $(dbus-launch)"
    fi

    systemctl --user start "$SERVICE_NAME"

    # Wait a moment for service to start
    sleep 2

    if systemctl --user is-active --quiet "$SERVICE_NAME"; then
        log_info "Mihomo service started successfully!"
    else
        log_error "Failed to start Mihomo service"
        log_info "Check logs with: journalctl --user -u mihomo -n 50"
        exit 1
    fi
}

print_success_message() {
    cat << SUCCESS

╔════════════════════════════════════════════════════════════╗
║     Mihomo Installation Completed (No Sudo Required!)     ║
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
  Config file:       $INSTALL_DIR/config.yaml
  Edit and reload:   mihomoctl reload

Web Dashboard (requires SSH tunnel):
  Local terminal:    ssh -L 9099:127.0.0.1:9090 user@server
  Dashboard URL:     http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099

Ports:
  Proxy (HTTP/SOCKS): 127.0.0.1:7897
  API:                127.0.0.1:9090

Installation Directory:
  $INSTALL_DIR

For troubleshooting, check:
  journalctl --user -u mihomo -f
  tail -f $INSTALL_DIR/mihomo.log

SUCCESS
}

# Main execution
main() {
    local config_file="${1:-}"

    echo "=========================================="
    echo "  Mihomo Setup (No Sudo Required)"
    echo "=========================================="
    echo ""
    log_info "Installation directory: $INSTALL_DIR"
    log_info "Version: $VERSION"
    echo ""

    # Run installation steps
    install_mihomo "$(detect_architecture)"
    setup_config "$config_file"
    install_user_service
    install_management_scripts
    start_service

    print_success_message
}

# Run main function
main "$@"
