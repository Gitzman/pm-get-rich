#!/usr/bin/env bash
set -euo pipefail

FLYCTL="/home/gitzman/.fly/bin/flyctl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VIZ_DIR="$REPO_DIR/viz"

echo "=== PM Get Rich Viz Deploy ==="

# Step 1: Export Parquet data
echo ""
echo "--- Exporting data to Parquet ---"
cd "$REPO_DIR"
uv run python -m scripts.export_parquet

# Step 2: Verify data files exist
echo ""
echo "--- Checking data files ---"
for f in weather_resolved weather_markets whale_leaderboard; do
    if [ ! -f "$VIZ_DIR/data/$f.parquet" ]; then
        echo "ERROR: Missing $VIZ_DIR/data/$f.parquet"
        exit 1
    fi
    size=$(stat -f%z "$VIZ_DIR/data/$f.parquet" 2>/dev/null || stat -c%s "$VIZ_DIR/data/$f.parquet")
    echo "  $f.parquet: $(( size / 1024 )) KB"
done

# Step 3: Create Fly app if it doesn't exist
echo ""
echo "--- Checking Fly.io app ---"
cd "$VIZ_DIR"
if ! $FLYCTL apps list 2>/dev/null | grep -q pmgetrich-viz; then
    echo "Creating app pmgetrich-viz..."
    $FLYCTL apps create pmgetrich-viz --org personal 2>/dev/null || true
fi

# Step 4: Deploy
echo ""
echo "--- Deploying to Fly.io ---"
$FLYCTL deploy --ha=false

echo ""
echo "=== Deploy complete ==="
$FLYCTL status
echo ""
echo "Site: https://pmgetrich-viz.fly.dev"
