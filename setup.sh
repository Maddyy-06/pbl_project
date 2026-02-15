#!/bin/bash
echo "🏥 Setting up Kidney Tumor Detection System..."
echo ""

# Remove old venv
rm -rf venv

# Create fresh virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python3 extract_kits19_samples.py"
echo "3. streamlit run app.py"