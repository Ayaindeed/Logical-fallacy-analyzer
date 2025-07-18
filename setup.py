#!/usr/bin/env python3
"""
Setup script for the Logical Fallacy News Analyzer
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("Installing required packages...")
        # Try different Python commands for Windows compatibility
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--no-warn-script-location"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to py command on Windows
            subprocess.check_call(["py", "-m", "pip", "install", "-r", "requirements.txt", "--no-warn-script-location"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("💡 Try running manually: py -m pip install -r requirements.txt")
        return False
    return True

def check_files():
    """Check if required files exist"""
    required_files = ["fallacies.csv", "requirements.txt"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    """Main setup function"""
    print("🔍 Setting up Logical Fallacy News Analyzer...")
    
    # Check files
    if not check_files():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("1. Basic version: py -m streamlit run streamlit_app.py")
    print("2. Advanced version: py -m streamlit run advanced_streamlit_app.py")
    print("\nAlternatively, if streamlit is in PATH:")
    print("1. streamlit run streamlit_app.py")
    print("2. streamlit run advanced_streamlit_app.py")
    print("\nMake sure to have your OpenAI and Serper API keys ready!")

if __name__ == "__main__":
    main()
