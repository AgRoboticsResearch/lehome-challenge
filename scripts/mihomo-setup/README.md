# Mihomo Multi-Server Setup

Automated setup scripts for deploying [Mihomo (Clash Meta)](https://github.com/MetaCubeX/mihomo) on multiple Ubuntu/Debian servers with systemd integration.

**What it does:** Enables external access for your LAN servers by deploying a proxy service that routes through your remote server.

## Features

- **Automated installation** - One-command setup with dependency management
- **Systemd integration** - Auto-start on boot, automatic restart on failure
- **Security focused** - Localhost-only API by default, proper permissions
- **Management tools** - Easy-to-use control commands
- **Terminal proxy** - Quick enable/disable for shell sessions

## Architecture

### Multi-Server Setup

Each server runs its own mihomo instance for local external access:

```
┌─────────────┐
│   Your Mac  │
└─────┬───────┘
      │
      ├───SSH──▶ zjuici (pool.zjuici.com:31329)
      │            └── Mihomo: 127.0.0.1:7897
      │
      ├───SSH──▶ lan102 (192.168.3.102:22)
      │            └── Mihomo: 127.0.0.1:7897
      │
      ├───SSH──▶ lan103 (192.168.3.103:22)
      │            └── Mihomo: 127.0.0.1:7897
      │
      ├───SSH──▶ lan104 (192.168.3.104:22)
      │            └── Mihomo: 127.0.0.1:7897
      │
      ├───SSH──▶ lan206 (192.168.3.206:22)
      │            └── Mihomo: 127.0.0.1:7897
      │
      └───SSH──▶ lan120 (192.168.3.120:22)
                   └── Mihomo: 127.0.0.1:7897
```

Each server uses mihomo for its own external access (pip, git, curl, etc.).

### Supported Servers

| Server ID | Name | Host | Port | User |
|-----------|------|------|------|------|
| `zjuici` | ZJUICI Remote | pool.zjuici.com | 31329 | admin |
| `lan102` | LAN 102 | 192.168.3.102 | 22 | hls |
| `lan103` | LAN 103 | 192.168.3.103 | 22 | hls |
| `lan104` | LAN 104 | 192.168.3.104 | 22 | hls |
| `lan206` | LAN 206 | 192.168.3.206 | 22 | hls |
| `lan120` | LAN 120 | 192.168.3.120 | 22 | hls |

## Quick Start

### 1. List Available Servers

```bash
cd scripts/mihomo-setup
./deploy_to_server.sh list
```

### 2. Deploy to All Servers

```bash
./deploy_to_server.sh all
```

### 3. Deploy to Specific Server

```bash
./deploy_to_server.sh lan102
./deploy_to_server.sh zjuici
```

### 4. Verify Installation

```bash
# Check status on all servers
./check_status.sh

# Check detailed status for specific server
./check_status.sh lan102 --details
```

## Usage

### Service Management

```bash
mihomoctl start    # Start service
mihomoctl stop     # Stop service
mihomoctl restart  # Restart service
mihomoctl status   # Check status
mihomoctl reload   # Reload config
mihomoctl logs     # View logs
```

## Sync Script Integration

The LeHome sync script (`../sync.sh`) integrates with mihomo for easy proxy management:

```bash
# Push to any server
../sync.sh push lan102

# Enable proxy on server for pip/git access
../sync.sh proxy-on lan102

# Disable proxy
../sync.sh proxy-off lan102

# Check mihomo status
../sync.sh proxy-status lan103

# SSH to server
../sync.sh exec lan102
```

### Terminal Proxy

```bash
# Enable proxy for current terminal session
source proxy_on

# Disable proxy
source proxy_off
```

Or use the global commands:
```bash
proxy_on    # Enable
proxy_off   # Disable
```

### Update Config

```bash
# Edit config
sudo nano /opt/mihomo/config.yaml

# Reload service
mihomoctl reload
```

## Web Dashboard

Mihomo's web UI (external-controller) is configured for **localhost only** for security. Access it via SSH tunnel:

### From Mac/Linux:

```bash
ssh -L 9099:127.0.0.1:9090 -p 22 admin@your-server
```

### From Windows (PowerShell):

```powershell
ssh -L 9099:127.0.0.1:9090 admin@your-server
```

Then open: [http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099](http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099)

### Alternative Web UIs:

- [Yacd](https://github.com/haishanh/yacd) - Modern dashboard
- [Clash Dashboard](https://dashboard.dash.elfhosted.com) - Classic interface

## Ports

| Port | Type | Access | Description |
|------|------|--------|-------------|
| 7897 | HTTP/SOCKS5 | 127.0.0.1 | Proxy server |
| 9090 | REST API | 127.0.0.1 | Web UI control |

## Directory Structure

```
/opt/mihomo/
├── mihomo              # Mihomo binary
├── config.yaml         # Configuration file
├── logs/
│   └── mihomo.log      # Service logs
├── mihomoctl           # Management script
├── proxy_on.sh         # Terminal proxy enable
└── proxy_off.sh        # Terminal proxy disable
```

## Troubleshooting

### Service Won't Start

```bash
# Check status and logs
mihomoctl status
mihomoctl logs

# Validate config
sudo /opt/mihomo/mihomo -d /opt/mihomo -t -f /opt/mihomo/config.yaml
```

### Connection Refused

```bash
# Check if port is listening
ss -tlnp | grep 7897

# Check firewall
sudo ufw status
```

### Permission Errors

```bash
# Ensure proper ownership
sudo chown -R nobody:nogroup /opt/mihomo

# Fix permissions
sudo chmod 644 /opt/mihomo/config.yaml
sudo chmod +x /opt/mihomo/mihomo
```

### Can't Access Web UI

Remember the web UI is **localhost only**. You must use an SSH tunnel:

```bash
ssh -L 9099:127.0.0.1:9090 admin@your-server
```

Then open http://127.0.0.1:9099/ui (or your preferred web UI).

## Security Notes

1. **Localhost-only API** - External controller only binds to 127.0.0.1
2. **Firewall** - Consider configuring UFW to restrict access:
   ```bash
   sudo ufw allow 22/tcp    # SSH only
   sudo ufw enable
   ```
3. **Secret** - Add a secret to config.yaml for web UI authentication:
   ```yaml
   secret: "your-strong-password"
   ```

## Updating Mihomo

```bash
# Download new version
sudo systemctl stop mihomo
sudo curl -L https://github.com/MetaCubeX/mihomo/releases/download/v1.18.10/mihomo-linux-amd64.gz -o /tmp/mihomo.gz
sudo gunzip -c /tmp/mihomo.gz > /opt/mihomo/mihomo
sudo chmod +x /opt/mihomo/mihomo
sudo systemctl start mihomo
```

## Uninstall

```bash
# Stop and disable service
sudo systemctl stop mihomo
sudo systemctl disable mihomo

# Remove files
sudo rm -rf /opt/mihomo
sudo rm /etc/systemd/system/mihomo.service
sudo systemctl daemon-reload

# Remove symlinks
sudo rm /usr/local/bin/mihomoctl
sudo rm /usr/local/bin/proxy_on
sudo rm /usr/local/bin/proxy_off
```

## License

This setup script is provided as-is for use with Mihomo (Clash Meta).
Mihomo is licensed under the GNU General Public License v3.0.

## Resources

- [Mihomo GitHub](https://github.com/MetaCubeX/mihomo)
- [Mihomo Documentation](https://wiki.metacubex.one/)
- [Clash Config Documentation](https://lancellc.gitbook.io/clash/)
- [Yacd Dashboard](https://github.com/haishanh/yacd)

## Support

For issues with:
- **Setup script**: Check logs with `journalctl -u mihomo -n 50`
- **Mihomo**: See [Mihomo issues](https://github.com/MetaCubeX/mihomo/issues)
- **Configuration**: Reference [Mihomo wiki](https://wiki.metacubex.one/)
