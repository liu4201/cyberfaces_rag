#!/bin/bash

source ./cyberfaces-rag/.env
# --- Configuration ---
DB_NAME="cybertraining"
TABLE="course_units"
OUTPUT_FILE="/Users/xiaoliu/Work/Project/SmartSearch/data.jsonl"
# ---------------------

echo "Starting export of table: $TABLE to $OUTPUT_FILE..."

# Execute MySQL Shell in SQL mode and pass commands via a Here Document
mysqlsh -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" --password="$DB_SECOND" --database "$DB_NAME" --result-format=ndjson -e "SELECT * FROM \`$TABLE\`" > "$OUTPUT_FILE"

echo "--- Script execution complete. ---"
echo "Check the file at: $OUTPUT_FILE"