from tensorflow import keras
from tensorflow.keras import layers

class DigitRecognitionModel:
    """CNN Model for Digit Recognition"""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build CNN architecture"""
        self.model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
            
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
            
            # Third Convolutional Block
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
            
            # Flatten and Dense Layers
            layers.Flatten(name='flatten'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        print("Model architecture built successfully!")
        return self.model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy'):
        """Compile the model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        print("Model compiled successfully!")
    
    def get_model_summary(self):
        """Display model summary"""
        return self.model.summary()
