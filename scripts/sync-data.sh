#!/bin/bash
set -euo pipefail

# --- Configuration ---
DB_NAME="${DB_NAME:-cybertraining}"
TABLE="${TABLE:-course_units}"             # main corpus  -> data.jsonl
MAP_TABLE="${MAP_TABLE:-course_unit_map}"  # unit->course  -> course_unit_map.jsonl
DATA_DIR="${DATA_DIR:-/data}"
CURRENT_DIR="$DATA_DIR/current"
VERSIONS_DIR="$DATA_DIR/versions"
MAX_VERSIONS=10

CURRENT_FILE="$CURRENT_DIR/data.jsonl"
CURRENT_MD5="$CURRENT_DIR/data.jsonl.md5"
MAP_FILE="$CURRENT_DIR/course_unit_map.jsonl"
MAP_MD5="$CURRENT_DIR/course_unit_map.jsonl.md5"
RELOAD_SIGNAL="$CURRENT_DIR/.reload"
CHROMADB_DIR="${CHROMADB_DIR:-/data/chromaDB}"
TMP_FILE="$DATA_DIR/data.jsonl.tmp"
MAP_TMP_FILE="$DATA_DIR/course_unit_map.jsonl.tmp"
# ---------------------

mkdir -p "$CURRENT_DIR" "$VERSIONS_DIR"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting data sync..."

# Test TCP connectivity first (shared by both table fetches)
echo "Testing TCP connectivity to $DB_HOST:3306..."
if ! bash -c "echo > /dev/tcp/$DB_HOST/3306" 2>/dev/null; then
    echo "ERROR: Cannot reach $DB_HOST:3306. Check DB_HOST env var and network."
    exit 1
fi
echo "TCP connection OK."

# fetch_table <table_name> <output_tmp_file>
# Dumps a full table as NDJSON into <output_tmp_file>. Aborts the whole sync on
# failure or empty result, so a bad fetch never clobbers good data downstream.
fetch_table() {
    local table="$1"
    local out_tmp="$2"
    local err_file="$DATA_DIR/mysqlsh.err"
    local exit_code=0

    echo "Fetching data from MySQL ($DB_HOST/$DB_NAME.$table)..."

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
        -e "SELECT * FROM \`$table\`" 2>"$err_file" \
        | grep '^{' > "$out_tmp" || exit_code=$?

    echo "mysqlsh exit code (table $table): $exit_code"
    if [ -s "$err_file" ]; then
        echo "--- mysqlsh stderr ---"
        cat "$err_file"
        echo "----------------------"
    fi
    rm -f "$err_file"

    if [ "$exit_code" -ne 0 ]; then
        rm -f "$out_tmp"
        echo "ERROR: mysqlsh failed for table $table."
        exit 1
    fi

    # Guard: abort if result is empty
    if [ ! -s "$out_tmp" ]; then
        echo "ERROR: MySQL returned empty result for $table. Aborting to prevent data loss."
        rm -f "$out_tmp"
        exit 1
    fi
}

# apply_if_changed <tmp_file> <current_file> <current_md5> <label>
# Compares checksums; if changed, archives the old file (pruning to MAX_VERSIONS)
# and swaps in the new one. Returns 0 when it applied a change, 1 when unchanged.
# Always call inside an `if` so `set -e` does not treat "unchanged" as a failure.
apply_if_changed() {
    local tmp="$1"
    local current="$2"
    local md5file="$3"
    local label="$4"
    local new_md5 old_md5 ts base count excess

    new_md5=$(md5sum "$tmp" | awk '{print $1}')
    echo "[$label] New data MD5: $new_md5"

    if [ -f "$md5file" ]; then
        old_md5=$(cat "$md5file")
        if [ "$new_md5" = "$old_md5" ]; then
            echo "[$label] No changes detected."
            rm -f "$tmp"
            return 1
        fi
        echo "[$label] Change detected (old: $old_md5, new: $new_md5)"
    else
        echo "[$label] No existing checksum found. Treating as new data."
    fi

    # Archive current version before replacing
    if [ -f "$current" ]; then
        ts=$(date -u '+%Y%m%d-%H%M%S')
        base=$(basename "$current")
        cp "$current" "$VERSIONS_DIR/$base.$ts"
        echo "[$label] Archived current file as $base.$ts"

        # Keep only the latest MAX_VERSIONS versions of THIS file
        count=$(ls "$VERSIONS_DIR/$base."* 2>/dev/null | wc -l)
        if [ "$count" -gt "$MAX_VERSIONS" ]; then
            excess=$((count - MAX_VERSIONS))
            ls "$VERSIONS_DIR/$base."* | sort | head -n "$excess" | xargs rm -f
            echo "[$label] Pruned $excess old version(s), keeping $MAX_VERSIONS."
        fi
    fi

    # Replace current file and update checksum
    mv "$tmp" "$current"
    echo "$new_md5" > "$md5file"
    echo "[$label] Replaced $current with new data."
    return 0
}

# 1. Fetch both tables from MySQL
fetch_table "$TABLE" "$TMP_FILE"
fetch_table "$MAP_TABLE" "$MAP_TMP_FILE"

# 2. Apply each independently, tracking what actually changed
DATA_CHANGED=0
if apply_if_changed "$TMP_FILE" "$CURRENT_FILE" "$CURRENT_MD5" "course_units"; then
    DATA_CHANGED=1
fi

MAP_CHANGED=0
if apply_if_changed "$MAP_TMP_FILE" "$MAP_FILE" "$MAP_MD5" "course_unit_map"; then
    MAP_CHANGED=1
fi

# 3. Clear chromaDB ONLY when the embedded corpus (data.jsonl) changed, so it
#    gets rebuilt on next reload. A mapping-only change is reloaded without
#    re-embedding (the parent-course lookup does not touch the vector store).
if [ "$DATA_CHANGED" -eq 1 ] && [ -d "$CHROMADB_DIR" ]; then
    rm -rf "${CHROMADB_DIR:?}"/*
    echo "Cleared chromaDB directory: $CHROMADB_DIR"
fi

# 4. Signal the app to hot-reload if EITHER dataset changed
if [ "$DATA_CHANGED" -eq 1 ] || [ "$MAP_CHANGED" -eq 1 ]; then
    touch "$RELOAD_SIGNAL"
    echo "Reload signal written to $RELOAD_SIGNAL"
else
    echo "No changes in either table. Nothing to reload."
fi

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Data sync complete."
