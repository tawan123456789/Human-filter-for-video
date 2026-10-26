#!/usr/bin/env bash
# ==============================================================================
# Jetson Nano Performance Booster
# Unlocks 10W MAXN power mode and locks CPU/GPU clocks at maximum frequency.
# Run on Jetson Nano with: sudo bash jetson_boost.sh
# ==============================================================================

set -e

echo "=========================================================="
echo " Jetson Nano Maximum Performance Configurator"
echo "=========================================================="

# Check root
if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run as root (use: sudo bash jetson_boost.sh)"
  exit 1
fi

echo "[+] Setting Power Mode to 0 (10W MAXN)..."
if command -v nvpmodel &> /dev/null; then
  nvpmodel -m 0
  echo "    Current mode: $(nvpmodel -q | grep -i "NV Power Mode")"
else
  echo "    [WARN] nvpmodel command not found."
fi

echo "[+] Locking CPU, GPU, and EMC memory clocks to MAX frequency..."
if command -v jetson_clocks &> /dev/null; then
  jetson_clocks
  jetson_clocks --show
else
  echo "    [WARN] jetson_clocks command not found."
fi

echo "[+] Checking SWAP memory..."
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
if [ "$SWAP_TOTAL" -lt 2000 ]; then
  echo "    [!] Swap is less than 2GB (current: ${SWAP_TOTAL}MB)."
  echo "    To prevent out-of-memory errors on Jetson Nano, create a 4GB swapfile:"
  echo "      sudo fallocate -l 4G /swapfile"
  echo "      sudo chmod 600 /swapfile"
  echo "      sudo mkswap /swapfile"
  echo "      sudo swapon /swapfile"
  echo "      echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
else
  echo "    Swap is healthy: ${SWAP_TOTAL} MB"
fi

echo ""
echo "[V] Jetson Nano is now configured for maximum FPS!"
echo "=========================================================="
