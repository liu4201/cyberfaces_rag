#!/bin/bash
set -euo pipefail

# --- Configuration ---
DB_NAME="${DB_NAME:-cybertraining}"
TABLE="${TABLE:-course_units}"
DATA_DIR="${DATA_DIR:-/data}"
CURRENT_DIR="$DATA_DIR/current"
VERSIONS_DIR="$DATA_DIR/versions"
MAX_VERSIONS=10

CURRENT_FILE="$CURRENT_DIR/data.jsonl"
CURRENT_MD5="$CURRENT_DIR/data.jsonl.md5"
RELOAD_SIGNAL="$CURRENT_DIR/.reload"
CHROMADB_DIR="${CHROMADB_DIR:-/data/chromaDB}"
TMP_FILE="$DATA_DIR/data.jsonl.tmp"
# ---------------------

mkdir -p "$CURRENT_DIR" "$VERSIONS_DIR"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting data sync..."

# 1. Fetch data from MySQL
echo "Fetching data from MySQL ($DB_HOST/$DB_NAME.$TABLE)..."

# Test TCP connectivity first
echo "Testing TCP connectivity to $DB_HOST:3306..."
if ! bash -c "echo > /dev/tcp/$DB_HOST/3306" 2>/dev/null; then
    echo "ERROR: Cannot reach $DB_HOST:3306. Check DB_HOST env var and network."
    exit 1
fi
echo "TCP connection OK."

MYSQLSH_EXIT=0
MYSQLSH_ERR_FILE="$DATA_DIR/mysqlsh.err"

MYSQLSH_TERM_COLOR_MODE=nocolor \
timeout 60 mysqlsh \
    --no-defaults \
    --no-wizard \
    --mysql \
    --sql \
    -u "$DB_USER" \
    "--password=$DB_PASSWORD" \
    -h "$DB_HOST" \
    --database "$DB_NAME" \
    --result-format=ndjson \
    -e "SELECT * FROM \`$TABLE\`" 2>"$MYSQLSH_ERR_FILE" \
    | grep '^{' > "$TMP_FILE" || MYSQLSH_EXIT=$?

echo "mysqlsh exit code: $MYSQLSH_EXIT"
if [ -s "$MYSQLSH_ERR_FILE" ]; then
    echo "--- mysqlsh stderr ---"
    cat "$MYSQLSH_ERR_FILE"
    echo "----------------------"
fi
rm -f "$MYSQLSH_ERR_FILE"

if [ $MYSQLSH_EXIT -ne 0 ]; then
    rm -f "$TMP_FILE"
    exit 1
fi

# 2. Guard: abort if result is empty
if [ ! -s "$TMP_FILE" ]; then
    echo "ERROR: MySQL returned empty result. Aborting to prevent data loss."
    rm -f "$TMP_FILE"
    exit 1
fi

# 3. Compute checksum of new data
NEW_MD5=$(md5sum "$TMP_FILE" | awk '{print $1}')
echo "New data MD5: $NEW_MD5"

# 4. Compare with existing checksum
if [ -f "$CURRENT_MD5" ]; then
    OLD_MD5=$(cat "$CURRENT_MD5")
    if [ "$NEW_MD5" = "$OLD_MD5" ]; then
        echo "No changes detected. Exiting."
        rm -f "$TMP_FILE"
        exit 0
    fi
    echo "Change detected (old: $OLD_MD5, new: $NEW_MD5)"
else
    echo "No existing checksum found. Treating as new data."
fi

# 5. Archive current version before replacing
if [ -f "$CURRENT_FILE" ]; then
    TIMESTAMP=$(date -u '+%Y%m%d-%H%M%S')
    cp "$CURRENT_FILE" "$VERSIONS_DIR/data.jsonl.$TIMESTAMP"
    echo "Archived current file as data.jsonl.$TIMESTAMP"

    # Keep only the latest MAX_VERSIONS versions
    VERSION_COUNT=$(ls "$VERSIONS_DIR"/data.jsonl.* 2>/dev/null | wc -l)
    if [ "$VERSION_COUNT" -gt "$MAX_VERSIONS" ]; then
        EXCESS=$((VERSION_COUNT - MAX_VERSIONS))
        ls "$VERSIONS_DIR"/data.jsonl.* | sort | head -n "$EXCESS" | xargs rm -f
        echo "Pruned $EXCESS old version(s), keeping $MAX_VERSIONS."
    fi
fi

# 6. Replace current file and update checksum
mv "$TMP_FILE" "$CURRENT_FILE"
echo "$NEW_MD5" > "$CURRENT_MD5"
echo "Replaced $CURRENT_FILE with new data."

# 7. Clear chromaDB so it gets rebuilt on next app start
if [ -d "$CHROMADB_DIR" ]; then
    rm -rf "${CHROMADB_DIR:?}"/*
    echo "Cleared chromaDB directory: $CHROMADB_DIR"
fi

# 8. Write reload signal for hot-reload watcher
touch "$RELOAD_SIGNAL"
echo "Reload signal written to $RELOAD_SIGNAL"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Data sync complete."
