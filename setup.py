#!/usr/bin/env python
"""
Setup Script - Initializes project directories and installs dependencies
Run this once to set up the entire project
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
END = '\033[0m'


def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text:^60}{END}")
    print(f"{BLUE}{'='*60}{END}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{END}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{END}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{END}")


def create_directories():
    """Create required project directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        'data/fer2013/train',
        'data/fer2013/validation',
        'data/ravdess/train',
        'data/ravdess/test',
        'models',
        'outputs/sessions',
        'src'
    ]
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for emotion in emotions:
        directories.append(f'data/fer2013/train/{emotion}')
        directories.append(f'data/fer2013/validation/{emotion}')
        directories.append(f'data/ravdess/train/{emotion}')
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory}")
        except Exception as e:
            print_error(f"Failed to create {directory}: {e}")
    
    print()


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    print("This may take 5-10 minutes depending on your internet connection...\n")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists('requirements.txt'):
            print_error("requirements.txt not found!")
            return False
        
        # Install using pip
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        
        print_success("All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def check_installations():
    """Check if all required packages are installed"""
    print_header("Verifying Installations")
    
    required_packages = [
        'tensorflow',
        'keras',
        'cv2',
        'fer',
        'librosa',
        'sounddevice',
        'streamlit',
        'numpy',
        'sklearn'
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            all_good = False
    
    print()
    return all_good


def create_config_template():
    """Create local config file if needed"""
    print_header("Configuration")
    
    if os.path.exists('src/config.py'):
        print_success("config.py already exists")
    else:
        print_warning("config.py not found. Please ensure it's in src/")
    
    print()


def print_next_steps():
    """Print next steps for user"""
    print_header("Next Steps")
    
    print(f"""
{BLUE}1. Download Datasets:{END}
   
   {YELLOW}FER-2013 (Facial Images):{END}
   - Go to: https://www.kaggle.com/msambare/fer2013
   - Download the CSV file
   - Extract images to: data/fer2013/train and data/fer2013/validation
   
   {YELLOW}RAVDESS (Speech Audio):{END}
   - Go to: https://zenodo.org/record/1188976
   - Download all actor folders
   - Extract audio files to: data/ravdess/train/[emotion]/

{BLUE}2. (Optional) Train LSTM Model:{END}
   python train_lstm.py

{BLUE}3. Run the Application:{END}
   streamlit run app.py

{BLUE}4. Open in Browser:{END}
   http://localhost:8501

{BLUE}5. Start Analyzing Emotions!{END}

    """)


def main():
    """Main setup function"""
    print(f"{BLUE}\n{'='*60}{END}")
    print(f"{BLUE}{'Real-Time Multimodal Emotion & Stress Detection':^60}{END}")
    print(f"{BLUE}{'System Setup Script':^60}{END}")
    print(f"{BLUE}{'='*60}{END}\n")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install dependencies
    print_header("Installation")
    proceed = input(f"{YELLOW}Install Python dependencies? (y/n): {END}")
    
    if proceed.lower() == 'y':
        if not install_dependencies():
            print_error("Installation failed!")
            sys.exit(1)
    
    # Step 3: Verify installations
    if not check_installations():
        print_warning("Some packages may not be installed correctly")
        print_warning("Please run: pip install -r requirements.txt")
    
    # Step 4: Create config
    create_config_template()
    
    # Step 5: Print next steps
    print_next_steps()
    
    print(f"{GREEN}✓ Setup completed successfully!{END}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Setup cancelled by user{END}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{END}\n")
        sys.exit(1)
