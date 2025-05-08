#!/bin/bash

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 is not installed. Please install it using:"
    echo "brew install python@3.9"
    exit 1
fi

# Create virtual environment with Python 3.9
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r voice/requirements.txt

# Add project root to PYTHONPATH
export PYTHONPATH="/Users/dhogan/DevelopmentProjects/home_assistant:$PYTHONPATH"

# Run the voice service
echo "Starting voice service..."
python voice/app/main.py 