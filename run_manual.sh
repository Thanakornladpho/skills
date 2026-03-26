#!/bin/bash
# Helper script for manual background removal (No AI, No Hangs)
# Usage: ./run_manual.sh [all|filename.png]

if [ -z "$1" ]; then
    echo "Usage: $0 [filename.png OR directory]"
    echo "Example: $0 photo.jpg"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="manual_cleaned_$(date +%s).png"

# Check if directory
if [ -d "$INPUT_PATH" ]; then
    OUTPUT_PATH="manual_cleaned_output"
fi

echo "Running Manual Background Removal on: $INPUT_PATH"
echo "Removing Color: BLACK | Tolerance: 20 | Removing Watermark: YES"

# Run Python script strictly in manual mode
./venv/bin/python remove_bg.py "$INPUT_PATH" -o "$OUTPUT_PATH" --remove-color white --tolerance 20 --remove-watermark

echo "Done! Saved to: $OUTPUT_PATH"
