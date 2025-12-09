import os
import sys
import argparse

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train
from test import test

def main():
    """Main execution script"""
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognition System')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'both'],
                       help='Mode: train, test, or both')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("Deep Learning Based Approach using CNN")
    print("="*60 + "\n")
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'both':
        train()
        test()

if __name__ == "__main__":
    main()
