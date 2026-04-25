#!/usr/bin/env python3
"""
Quick Start Guide - Interactive setup for the gesture recognition project
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a shell command and show output"""
    print(f"➤ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"  ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name} - Installed")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing.append(name)
    
    return len(missing) == 0, missing

def create_directories():
    """Create necessary project directories"""
    print_header("Creating Project Directories")
    
    dirs = [
        'data/train/l',
        'data/train/peace',
        'data/train/stop',
        'data/train/thumbs_up',
        'data/val/l',
        'data/val/peace',
        'data/val/stop',
        'data/val/thumbs_up',
        'model'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def print_next_steps():
    """Print next steps for the user"""
    print_header("Next Steps")
    
    print("📋 Follow these steps to complete the setup:\n")
    
    print("1️⃣  COLLECT DATA:")
    print("    python collect_data.py")
    print("    - Collect 500+ images per gesture")
    print("    - Use SPACE to capture, Q for next gesture\n")
    
    print("2️⃣  TRAIN MODEL:")
    print("    python train.py")
    print("    - This will take 10-30 minutes")
    print("    - Watch for training progress\n")
    
    print("3️⃣  EVALUATE MODEL:")
    print("    python evaluate.py")
    print("    - Check accuracy and confusion matrix\n")
    
    print("4️⃣  RUN PREDICTIONS:")
    print("    Option A - Real-time webcam:")
    print("    python predict.py\n")
    print("    Option B - Web interface:")
    print("    streamlit run app.py\n")
    
    print("📖 For detailed instructions, see README.md")

def main():
    print_header("🤚 Hand Gesture Recognition - Quick Start")
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\n📦 Install them with:")
        print("   pip install -r requirements.txt\n")
        response = input("Would you like to install now? (y/n): ").lower()
        
        if response == 'y':
            success = run_command(
                f"{sys.executable} -m pip install -r requirements.txt",
                "Installing dependencies"
            )
            if not success:
                print("\n❌ Installation failed. Please install manually.")
                return
        else:
            print("\n⚠️  Please install dependencies before continuing.")
            return
    
    # Create directories
    create_directories()
    
    # Check if data exists
    print_header("Checking Project Status")
    
    train_data = any(os.listdir(f'data/train/{g}') 
                    for g in ['l', 'peace', 'stop', 'thumbs_up'] 
                    if os.path.exists(f'data/train/{g}'))
    
    if train_data:
        print("✅ Training data found")
    else:
        print("⚠️  No training data found - run collect_data.py")
    
    if os.path.exists('model/gesture_model.h5'):
        print("✅ Trained model found")
    else:
        print("⚠️  No trained model found - run train.py")
    
    # Print next steps
    print_next_steps()
    
    # Offer to start collection
    if not train_data:
        print("\n" + "="*70)
        response = input("\nStart data collection now? (y/n): ").lower()
        if response == 'y':
            print("\nStarting data collection...")
            os.system("python collect_data.py")
    
    print("\n" + "="*70)
    print("  Setup Complete! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
