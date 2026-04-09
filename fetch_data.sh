#!/usr/bin/env bash
# fetch_data.sh — Download CryoPPP dataset for training
set -euo pipefail

EMPIAR_ID="${1:-10017}"
DATA_DIR="data/cryoppp/lite/${EMPIAR_ID}"
URL="http://calla.rnet.missouri.edu/cryoppp_lite/${EMPIAR_ID}.tar.gz"

if [ -d "${DATA_DIR}/micrographs" ]; then
    echo "Data already present at ${DATA_DIR}"
    MICS=$(find "${DATA_DIR}/micrographs" -type f | wc -l)
    COORDS=$(find "${DATA_DIR}/ground_truth/particle_coordinates" -type f -name "*.csv" 2>/dev/null | wc -l)
    echo "  Micrographs: ${MICS}, Coordinate files: ${COORDS}"
    exit 0
fi

echo "Downloading CryoPPP EMPIAR-${EMPIAR_ID} lite..."
mkdir -p "data/cryoppp/lite"
TARBALL="/tmp/${EMPIAR_ID}.tar.gz"

if command -v wget &> /dev/null; then
    wget -q --show-progress "${URL}" -O "${TARBALL}"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar "${URL}" -o "${TARBALL}"
fi

echo "Extracting..."
mkdir -p "${DATA_DIR}"
tar xzf "${TARBALL}" --no-same-owner -C "${DATA_DIR}" 2>/dev/null || tar xzf "${TARBALL}" -C "${DATA_DIR}"
rm -f "${TARBALL}"

# Fix: tar may create a nested directory (10017/10017/) — flatten it
if [ -d "${DATA_DIR}/${EMPIAR_ID}/micrographs" ]; then
    echo "Fixing nested directory..."
    mv "${DATA_DIR}/${EMPIAR_ID}"/* "${DATA_DIR}/"
    rmdir "${DATA_DIR}/${EMPIAR_ID}"
fi

MICS=$(find "${DATA_DIR}" -type f \( -name "*.mrc" -o -name "*.jpg" \) | wc -l)
COORDS=$(find "${DATA_DIR}" -type f -name "*.csv" | wc -l)
SIZE=$(du -sh "${DATA_DIR}" | cut -f1)
echo "Done: ${MICS} micrographs, ${COORDS} coordinate files, ${SIZE}"
