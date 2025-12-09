Handwritten Digit Recognition System

## Project Overview
A deep learning-based system for recognizing handwritten digits using Convolutional Neural Networks (CNN) trained on the MNIST dataset.

## Project Structure
```
handwritten-digit-recognition/
│
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Data loading and preprocessing
│
├── models/
│   ├── __init__.py
│   └── cnn_model.py            # CNN model architecture
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Visualization utilities
│   └── evaluation.py           # Model evaluation functions
│
├── config/
│   ├── __init__.py
│   └── config.py               # Configuration parameters
│
├── saved_models/               # Trained models (auto-created)
├── results/                    # Output visualizations (auto-created)
│
├── main.py                     # Main execution script
├── train.py                    # Training script
├── test.py                     # Testing script
├── requirements.txt            # Required packages
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python train.py
```

### Testing the Model
```bash
python test.py
```

### Running Both Training and Testing
```bash
python main.py --mode both
```

## Model Architecture
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax output for 10 digit classes (0-9)

## Results
- Expected accuracy: ~99% on MNIST test set
- Training visualizations saved in `results/` directory
- Trained model saved in `saved_models/` directory

## Future Enhancements
- Data augmentation
- Real-time digit drawing interface
- Feature map visualization
- Model deployment (web/mobile)
- Support for custom datasets

## Technologies Used
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn