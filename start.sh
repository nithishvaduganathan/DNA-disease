#!/bin/bash
# Quick start script for DNA Disease Prediction System

echo "=================================="
echo "DNA Disease Prediction System"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -q -r requirements.txt

# Check if model exists
if [ ! -f "model/disease_model.pkl" ] || [ ! -f "model/vectorizer.pkl" ]; then
    echo ""
    echo "Model not found. Generating dataset and training model..."
    
    # Generate dataset
    python3 data/generate_dataset.py
    
    # Train model
    python3 model/train_model.py
fi

echo ""
echo "=================================="
echo "Starting Flask application..."
echo "=================================="
echo ""
echo "Open your browser and navigate to: http://localhost:5000"
echo ""

# Start Flask app
python3 app.py
