#!/bin/bash
# =============================================================================
# Extract CLAP Embeddings
#
# Pre-extracts 512-d CLAP embeddings from wav files for frozen CLAP mode.
# Fill in the two paths below, then run:  bash scripts/extract_clap_embeddings.sh
# =============================================================================

# ---- FILL THESE IN ----
WAV_ROOT=""          # Path to directory containing wav files (e.g., /data/Sorted)
OUT_ROOT=""          # Path to save extracted embeddings (e.g., /data/clap_embeddings)
# ------------------------

# Optional: set to --enable_fusion if you want CLAP fusion mode
FUSION_FLAG=""

# ---- Validation ----
if [ -z "$WAV_ROOT" ]; then
    echo "ERROR: WAV_ROOT is not set. Edit this script and fill in the path to your wav files."
    exit 1
fi

if [ -z "$OUT_ROOT" ]; then
    echo "ERROR: OUT_ROOT is not set. Edit this script and fill in the output path."
    exit 1
fi

if [ ! -d "$WAV_ROOT" ]; then
    echo "ERROR: WAV_ROOT directory does not exist: $WAV_ROOT"
    exit 1
fi

# ---- Run extraction ----
echo "Extracting CLAP embeddings..."
echo "  WAV_ROOT: $WAV_ROOT"
echo "  OUT_ROOT: $OUT_ROOT"
echo ""

python "$(dirname "$0")/extract_clap_embeddings.py" \
    --wav_root "$WAV_ROOT" \
    --out_root "$OUT_ROOT" \
    $FUSION_FLAG

echo ""
echo "Done. Use these settings in your YAML config:"
echo "  is_clap: True"
echo "  clap_data_mode: embeddings"
echo "  clap_emb_root: $OUT_ROOT"
