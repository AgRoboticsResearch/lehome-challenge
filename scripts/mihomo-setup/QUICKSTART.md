# Quick Start: Multi-Server Mihomo Deployment

## Overview

Deploy mihomo proxy service to all your LeHome Challenge servers. Each server runs its own mihomo instance for local external access (pip, git, curl, etc.).

### Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Your Mac  │ ──SSH──▶ zjuici    │ Mihomo:7897│ (local proxy)
└─────────────┘                    └─────────────┘
     │
     ├──SSH──▶ lan102              ┌─────────────┐
     │                              │ Mihomo:7897│ (local proxy)
     ├──SSH──▶ lan103              └─────────────┘
     │                              ┌─────────────┐
     ├──SSH──▶ lan104              │ Mihomo:7897│ (local proxy)
     │                              └─────────────┘
     ├──SSH──▶ lan206              ┌─────────────┐
     │                              │ Mihomo:7897│ (local proxy)
     │                              └─────────────┘
     └──SSH──▶ lan120              ┌─────────────┐
                                    │ Mihomo:7897│ (local proxy)
                                    └─────────────┘
```

Each server has mihomo running locally on `127.0.0.1:7897` for that server's own use.

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

### 1. Deploy to All Servers

```bash
cd /Users/moky/codes/lehome-challenge/scripts/mihomo-setup
./deploy_to_server.sh all
```

This will deploy mihomo to all servers sequentially.

### 2. Deploy to Specific Server

```bash
./deploy_to_server.sh lan102      # Deploy to LAN 102
./deploy_to_server.sh zjuici      # Deploy to ZJUICI (default)
```

### 3. List All Servers

```bash
./deploy_to_server.sh list
```

### 4. Check Status on All Servers

```bash
./check_status.sh                 # Quick status check
./check_status.sh lan102 --details  # Detailed status for lan102
```

## After Deployment

### Test Proxy on Server

```bash
# SSH to server
ssh -p 31329 admin@pool.zjuici.com       # For zjuici
ssh -p 22 hls@192.168.3.102              # For LAN servers

# Test proxy
curl -x http://127.0.0.1:7897 https://api.ipify.org
```

### Enable Proxy for Terminal Session

```bash
# On the server
source proxy_on
curl https://api.ipify.org  # Should return external IP
```

### Use with pip, git, etc.

```bash
# On the server, after enabling proxy
source proxy_on

# Install Python packages
pip install torch

# Clone from GitHub
git clone https://github.com/...

# Download files
wget https://...
```

### Sync Script Integration

```bash
# Push to any server
cd /Users/moky/codes/lehome-challenge
./scripts/sync.sh push lan102

# Enable proxy on server for pip/git access
./scripts/sync.sh proxy-on lan102

# Check mihomo status
./scripts/sync.sh proxy-status lan103

# SSH to server
./scripts/sync.sh exec lan102
```

## Web UI Dashboard

### For ZJUICI Server

```bash
# SSH tunnel from Mac
ssh -L 9099:127.0.0.1:9090 -L 7897:127.0.0.1:7897 -p 31329 admin@pool.zjuici.com

# Open dashboard
open http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099
```

### For LAN Servers

```bash
# SSH tunnel from Mac to LAN server
ssh -L 9099:127.0.0.1:9090 -L 7897:127.0.0.1:7897 -p 22 hls@192.168.3.102

# Open dashboard
open http://yacd.haishan.me/#/setup?hostname=127.0.0.1&port=9099
```

## Service Management

On each server:

```bash
mihomoctl status    # Check status
mihomoctl logs      # View logs
mihomoctl restart   # Restart service
```

## Update Config

If you need to update your subscription config:

```bash
# Download updated config
curl -s "https://qic0q.no-mad-world.club/link/F0lRt3K9MSyETv94?clash=3" -o config.yaml
sed -i '' 's/^port: 7890/port: 7897/' config.yaml

# Deploy to all servers
./deploy_to_server.sh all
```

Or update on specific server:

```bash
# On the server
sudo nano /opt/mihomo/config.yaml
mihomoctl reload
```

## Troubleshooting

### Check Mihomo Status

```bash
# Check all servers
./check_status.sh

# Check specific server
./check_status.sh lan102 --details

# Or via sync script
./scripts/sync.sh proxy-status lan102
```

### Proxy Not Working

```bash
# SSH to server
ssh -p 22 hls@192.168.3.102

# Check service
sudo systemctl status mihomo

# Check logs
sudo journalctl -u mihomo -n 50

# Test proxy directly
curl -x http://127.0.0.1:7897 https://api.ipify.org
```

### Config Validation

```bash
# SSH to server
ssh -p 22 hls@192.168.3.102

# Test config
sudo /opt/mihomo/mihomo -d /opt/mihomo -t -f /opt/mihomo/config.yaml
```

## Config Details

Each server uses the same config:

| Setting | Value |
|---------|-------|
| Proxy Port | 7897 (HTTP/SOCKS5) |
| API Port | 9090 (localhost only) |
| Config File | /opt/mihomo/config.yaml |
| Log File | /opt/mihomo/logs/mihomo.log |

## Related Scripts

- `deploy_to_server.sh` - Deploy mihomo to servers
- `check_status.sh` - Check mihomo status on all servers
- `../sync.sh` - Multi-server sync with proxy commands
