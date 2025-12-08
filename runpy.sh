#!/bin/bash
# Wrapper to run Python scripts with project root on PYTHONPATH

ROOT_PY="development/03_validate_deserialize_boards/trainRL_view_pred.py"
# Create logs directory if it doesn't exist
mkdir -p logs

# Get the base name without .py extension and create timestamped log file
TIMESTAMP=$(date +%y-%m-%d_%H_%M)

BASENAME=$(basename "$ROOT_PY" .py)
# BASENAME=$(basename "$1" .py)
LOGFILE="logs/${BASENAME}-${TIMESTAMP}.log"

# Run with unbuffered output, save to file and display on screen
PYTHONPATH=. python -u "$ROOT_PY" | tee "$LOGFILE"

# Remove ANSI color codes from the log file after execution
sed -i 's/\x1b\[[0-9;]*m//g' "$LOGFILE"
echo "Log saved to: $LOGFILE (colors stripped)"