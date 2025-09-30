#!/bin/bash
# CT Scan Classification Project Installation Script

echo "🏥 CT Scan Classification Project Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. Python 3.7+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 detected"

# Create virtual environment (optional)
read -p "🤔 Do you want to create a virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv ct_classifier_env
    source ct_classifier_env/bin/activate
    echo "✅ Virtual environment created and activated"
    echo "💡 To activate in the future, run: source ct_classifier_env/bin/activate"
fi

# Install requirements
echo "📥 Installing requirements..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Requirements installed successfully"
else
    echo "❌ Failed to install requirements"
    exit 1
fi

# Check if model file exists
if [ ! -f "resnet_50_23dataset.pth" ]; then
    echo "⚠️  Model file 'resnet_50_23dataset.pth' not found in current directory"
    echo "📝 Please place your trained model file in this directory"
    echo "💡 You can download it or copy it from your training setup"
fi

# Make scripts executable
chmod +x classify_ct_scans.py
chmod +x example_usage.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "🚀 Quick start:"
echo "   python classify_ct_scans.py --help"
echo ""
echo "📖 For detailed usage, see README.md"
echo ""
echo "🧪 Run examples:"
echo "   python example_usage.py"
echo ""
