#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Project root directory (assuming the script is in the project root)
PROJECT_ROOT=$(pwd)

# Documentation directory
DOCS_DIR="$PROJECT_ROOT/docs"

# Source code directory
SRC_DIR="$PROJECT_ROOT/src/detoxai"
# Change to the docs directory
cd "$DOCS_DIR"

# Clean the build
echo "Cleaning previous builds..."
make clean

# Generate API documentation (explicitly specify output path)
echo "Generating API documentation..."
sphinx-apidoc -o "$DOCS_DIR" "$SRC_DIR"

# Build the HTML documentation
echo "Building HTML documentation..."
make html

echo "Documentation build complete."