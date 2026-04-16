#!/bin/bash
# Wrapper to run Python scripts with project root on PYTHONPATH

ROOT_PY="trainRL.py"
ECHO_OUTPUT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no_echo)
            ECHO_OUTPUT=false
            shift
            ;;
        *)
            ROOT_PY="$1"
            shift
            ;;
    esac
done

# Use default if no script provided
if [ -z "$ROOT_PY" ]; then
    ROOT_PY="trainRL.py"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the base name without .py extension and create timestamped log file
TIMESTAMP=$(date +%y-%m-%d_%H_%M)

BASENAME=$(basename "$ROOT_PY" .py)
LOGFILE="logs/${BASENAME}-${TIMESTAMP}.log"

# Run with unbuffered output
if [ "$ECHO_OUTPUT" = true ]; then
    # Display on screen and save to file
    PYTHONPATH=. python -u "$ROOT_PY" | tee "$LOGFILE"
else
    # Only save to file, no terminal output
    PYTHONPATH=. python "$ROOT_PY" > "$LOGFILE" 2>&1
fi

# Remove ANSI color codes from the log file after execution
sed -i 's/\x1b\[[0-9;]*m//g' "$LOGFILE"
echo "Log saved to: $LOGFILE (colors stripped)"