#!/bin/bash

# Define paths
DOWNLOADS_DIR="$HOME/Downloads"
PROJECT_DIR="$HOME/Downloads/MDS Git Sem 2/wqd7005-project"
MODELS_ZIP="$DOWNLOADS_DIR/images.zip"
MODELS_DIR="$PROJECT_DIR/images"

# Check if models.zip exists in Downloads
if [ ! -f "$MODELS_ZIP" ]; then
    echo "Error: $MODELS_ZIP not found!"
    exit 1
fi

echo "Found images.zip in Downloads folder"

# Ensure models directory exists in project
if [ ! -d "$MODELS_DIR" ]; then
    echo "Creating models directory..."
    mkdir -p "$MODELS_DIR"
fi

# Unzip and overwrite existing content
echo "Extracting images.zip to $MODELS_DIR..."
unzip -o "$MODELS_ZIP" -d "$PROJECT_DIR"

echo "Images successfully updated!"