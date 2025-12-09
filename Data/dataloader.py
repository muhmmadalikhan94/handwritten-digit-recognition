import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load_data(self):
        """Load MNIST dataset"""
        print("Loading MNIST dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.x_train)}")
        print(f"Test samples: {len(self.x_test)}")
    
    def preprocess_data(self):
        """Preprocess the data"""
        # Normalize pixel values
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Reshape data
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)
        
        # Convert labels to categorical
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)
        
        print("Data preprocessing completed!")
    
    def get_data(self):
        """Return preprocessed data"""
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
