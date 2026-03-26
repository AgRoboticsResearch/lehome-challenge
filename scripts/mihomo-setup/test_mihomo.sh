#!/usr/bin/env bash

##############################################################
# Mihomo Proxy Test Script
# Run this after deployment to verify the proxy is working
##############################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default proxy settings
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-7897}"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
TIMEOUT="${TIMEOUT:-15}"

# Test results
PASSED=0
FAILED=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=$((FAILED + 1))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

# Test 1: Check if mihomo process is running
test_mihomo_running() {
    log_test "Checking if mihomo process is running..."

    if pgrep -f "mihomo" > /dev/null 2>&1; then
        local pid=$(pgrep -f "mihomo" | head -1)
        log_pass "Mihomo is running (PID: $pid)"
        return 0
    else
        log_fail "Mihomo is NOT running"
        return 1
    fi
}

# Test 2: Check if proxy port is listening
test_proxy_port() {
    log_test "Checking if proxy port ${PROXY_PORT} is listening..."

    if nc -z ${PROXY_HOST} ${PROXY_PORT} 2>/dev/null || \
       (echo > /dev/tcp/${PROXY_HOST}/${PROXY_PORT}) 2>/dev/null; then
        log_pass "Port ${PROXY_PORT} is open"
        return 0
    else
        # Try with ss or netstat
        if ss -tln 2>/dev/null | grep -q ":${PROXY_PORT}" || \
           netstat -tln 2>/dev/null | grep -q ":${PROXY_PORT}"; then
            log_pass "Port ${PROXY_PORT} is open"
            return 0
        fi
        log_fail "Port ${PROXY_PORT} is NOT listening"
        return 1
    fi
}

# Test 3: Check RESTful API
test_api() {
    log_test "Checking mihomo RESTful API..."

    local response
    response=$(curl -s --connect-timeout 5 "http://${PROXY_HOST}:9090/version" 2>/dev/null)

    if [[ -n "$response" ]]; then
        log_pass "API is responding: $response"
        return 0
    else
        log_fail "API is NOT responding"
        return 1
    fi
}

# Test 4: Test basic HTTP proxy
test_http_proxy() {
    log_test "Testing HTTP proxy with ipify..."

    local ip
    ip=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s https://api.ipify.org 2>/dev/null)

    if [[ -n "$ip" && "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_pass "HTTP proxy works! Exit IP: $ip"
        return 0
    else
        log_fail "HTTP proxy NOT working"
        return 1
    fi
}

# Test 5: Test with ipinfo (shows location)
test_ipinfo() {
    log_test "Testing proxy location with ipinfo..."

    local info
    info=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s https://ipinfo.io/json 2>/dev/null)

    if [[ -n "$info" ]]; then
        local country=$(echo "$info" | grep -o '"country": *"[^"]*"' | cut -d'"' -f4)
        local city=$(echo "$info" | grep -o '"city": *"[^"]*"' | cut -d'"' -f4)
        log_pass "Proxy location: $city, $country"
        return 0
    else
        log_fail "Could not get proxy location"
        return 1
    fi
}

# Test 6: Test wandb connectivity
test_wandb() {
    log_test "Testing wandb connectivity..."

    local status
    status=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s -o /dev/null -w "%{http_code}" https://api.wandb.ai/healthz 2>/dev/null)

    if [[ "$status" == "200" ]]; then
        log_pass "wandb API is accessible (HTTP $status)"
        return 0
    else
        log_fail "wandb API returned HTTP $status"
        return 1
    fi
}

# Test 7: Test GitHub connectivity
test_github() {
    log_test "Testing GitHub connectivity..."

    local status
    status=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s -o /dev/null -w "%{http_code}" https://api.github.com/zen 2>/dev/null)

    if [[ "$status" == "200" ]]; then
        log_pass "GitHub API is accessible (HTTP $status)"
        return 0
    else
        log_fail "GitHub API returned HTTP $status"
        return 1
    fi
}

# Test 8: Test Google connectivity
test_google() {
    log_test "Testing Google connectivity..."

    local status
    status=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s -o /dev/null -w "%{http_code}" https://www.google.com/generate_204 2>/dev/null)

    if [[ "$status" == "204" ]]; then
        log_pass "Google is accessible (HTTP $status)"
        return 0
    else
        log_fail "Google returned HTTP $status"
        return 1
    fi
}

# Test 9: Check environment variables
test_env_vars() {
    log_test "Checking proxy environment variables..."

    local has_env=0

    if [[ -n "$http_proxy" ]]; then
        log_info "http_proxy=$http_proxy"
        has_env=1
    fi

    if [[ -n "$https_proxy" ]]; then
        log_info "https_proxy=$https_proxy"
        has_env=1
    fi

    if [[ -n "$all_proxy" ]]; then
        log_info "all_proxy=$all_proxy"
        has_env=1
    fi

    if [[ $has_env -eq 1 ]]; then
        log_pass "Environment variables are set"
        return 0
    else
        log_warn "No proxy environment variables set"
        log_info "To set them, run: export http_proxy=${PROXY_URL} https_proxy=${PROXY_URL}"
        return 0
    fi
}

# Test 10: Test HuggingFace connectivity
test_huggingface() {
    log_test "Testing HuggingFace connectivity..."

    local status
    status=$(curl -x ${PROXY_URL} --connect-timeout ${TIMEOUT} -s -o /dev/null -w "%{http_code}" https://huggingface.co/healthcheck 2>/dev/null)

    if [[ "$status" == "200" ]]; then
        log_pass "HuggingFace is accessible (HTTP $status)"
        return 0
    else
        log_fail "HuggingFace returned HTTP $status"
        return 1
    fi
}

# Print summary
print_summary() {
    print_header "Test Summary"

    echo -e "  ${GREEN}Passed:${NC} $PASSED"
    echo -e "  ${RED}Failed:${NC} $FAILED"
    echo ""

    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed! Proxy is working correctly.${NC}"
        echo ""
        echo "To use the proxy in your current shell:"
        echo "  export http_proxy=${PROXY_URL}"
        echo "  export https_proxy=${PROXY_URL}"
        echo ""
        echo "Or add to ~/.bashrc for persistence:"
        echo "  echo 'export http_proxy=${PROXY_URL}' >> ~/.bashrc"
        echo "  echo 'export https_proxy=${PROXY_URL}' >> ~/.bashrc"
        return 0
    else
        echo -e "${RED}Some tests failed. Check the output above for details.${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check if mihomo is running: pgrep -a mihomo"
        echo "  2. Check logs: tail -50 ~/.mihomo/mihomo.log"
        echo "  3. Restart mihomo: pkill -f mihomo && nohup ~/.mihomo/mihomo -d ~/.mihomo -f ~/.mihomo/config.yaml > ~/.mihomo/mihomo.log 2>&1 &"
        echo "  4. Update config from subscription: curl -o ~/.mihomo/config.yaml 'SUBSCRIPTION_URL'"
        return 1
    fi
}

# Main
main() {
    print_header "Mihomo Proxy Test Suite"

    echo "Configuration:"
    echo "  Proxy URL: ${PROXY_URL}"
    echo "  Timeout: ${TIMEOUT}s"
    echo ""

    # Run all tests
    test_mihomo_running
    test_proxy_port
    test_api
    test_http_proxy
    test_ipinfo
    test_wandb
    test_github
    test_google
    test_huggingface
    test_env_vars

    # Print summary
    print_summary
}

# Run main
main "$@"
