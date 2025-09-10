"""
Animal Detection System

A real-time animal detection system using YOLOv8n for COCO animal classes.
Supports 7 animal types: horse, sheep, cow, elephant, bear, zebra, giraffe.
"""

__version__ = "1.0.0"
__author__ = "Animal Detection Team"

# Core modules
from .detector import SimpleAnimalDetector
from .database import SimpleInfluxDB, create_database, test_database_connection

# Configuration
import os
import yaml

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_animal_classes():
    """Get list of supported animal classes"""
    config = load_config()
    animals = []
    for animal in config['animals']['classes']:
        animals.append({
            'name': animal['name'],
            'thai_name': animal['thai_name'],
            'coco_id': animal['coco_id']
        })
    return animals

def get_version():
    """Get current version"""
    return __version__

# System information
SUPPORTED_ANIMALS = [
    'horse', 'sheep', 'cow', 'elephant', 
    'bear', 'zebra', 'giraffe'
]

COCO_IDS = [17, 18, 19, 20, 21, 22, 23]

# Convenience functions
def create_detection_system(config_path="config/config.yaml"):
    """Create a complete detection system with database"""
    detector = SimpleAnimalDetector(config_path)
    database = SimpleInfluxDB(config_path)
    return detector, database

def check_system_requirements():
    """Check if all required packages are available"""
    required_packages = [
        'ultralytics', 'cv2', 'yaml', 'influxdb_client', 
        'streamlit', 'plotly', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            elif package == 'influxdb_client':
                import influxdb_client
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("All required packages are installed")
        return True

# Export main classes and functions
__all__ = [
    'SimpleAnimalDetector',
    'SimpleInfluxDB', 
    'create_database',
    'test_database_connection',
    'load_config',
    'get_animal_classes',
    'get_version',
    'create_detection_system',
    'check_system_requirements',
    'SUPPORTED_ANIMALS',
    'COCO_IDS'
]