#!/bin/bash

# Create auxiliary directory if it doesn't exist
mkdir -p aux

# Move check/test files to aux directory
mv check-*.yaml aux/ 2>/dev/null || true
mv test-*.yaml aux/ 2>/dev/null || true

# Remove any temporary or unnecessary files
rm -f *.log *.tmp

echo "Repository cleaned up!" 