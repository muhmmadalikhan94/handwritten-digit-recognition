import numpy as np
import ssl
import certifi
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
        # Fix SSL certificate issues
        self._fix_ssl()
    
    def _fix_ssl(self):
        """Fix SSL certificate verification issues"""
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print("SSL certificate verification disabled for downloading dataset")
        except Exception as e:
            print(f"Note: Could not modify SSL settings: {e}")
    
    def load_data(self):
        """Load MNIST dataset"""
        print("Loading MNIST dataset...")
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            print(f"Dataset loaded successfully!")
            print(f"Training samples: {len(self.x_train)}")
            print(f"Test samples: {len(self.x_test)}")
        except Exception as e:
            print(f"\nError loading dataset: {e}")
            print("\nTrying alternative download method...")
            self._download_manually()
    
    def _download_manually(self):
        """Fallback method to download MNIST if automatic download fails"""
        import urllib.request
        import os
        from pathlib import Path
        
        # Create cache directory
        cache_dir = Path.home() / '.keras' / 'datasets'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        mnist_path = cache_dir / 'mnist.npz'
        
        if not mnist_path.exists():
            print("Downloading MNIST dataset manually...")
            url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
            
            # Disable SSL verification for download
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            try:
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    with open(mnist_path, 'wb') as f:
                        f.write(response.read())
                print("Download complete!")
            except Exception as e:
                print(f"Manual download also failed: {e}")
                raise
        
        # Load from downloaded file
        with np.load(mnist_path, allow_pickle=True) as f:
            self.x_train, self.y_train = f['x_train'], f['y_train']
            self.x_test, self.y_test = f['x_test'], f['y_test']
        
        print(f"Dataset loaded from cache!")
        print(f"Training samples: {len(self.x_train)}")
        print(f"Test samples: {len(self.x_test)}")
    
    def preprocess_data(self):
        """Preprocess the data"""
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
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
        if self.x_train is None or self.y_train is None:
            raise ValueError("Data not loaded or preprocessed.")
        return (self.x_train, self.y_train), (self.x_test, self.y_test)