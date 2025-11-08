#!/bin/bash

# Quick start script for the ML Model Server

echo "Starting ML Model Server..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not detected."
    echo "   Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Virtual environment not found. Please create one first:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        exit 1
    fi
fi

# Check if flask-cors is installed
python -c "import flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing flask-cors..."
    pip install flask-cors
fi

# Start the server
echo "Starting server with default settings..."
echo "   Experiment: House_Price_Prediction"
echo "   Port: 5000"
echo ""
python main.py --experiment "House_Price_Prediction" --port 5000

