import os
from data.data_loader import DataLoader
from models.cnn_model import DigitRecognitionModel
from utils.visualization import Visualizer
from config.config import *

def train():
    """Training pipeline"""
    print("\n" + "="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - TRAINING")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    (x_train, y_train), (x_test, y_test) = data_loader.get_data()
    
    # Build model
    model_builder = DigitRecognitionModel(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS),
        num_classes=NUM_CLASSES
    )
    model = model_builder.build_model()
    model_builder.compile_model()
    model_builder.get_model_summary()
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_training_history(history)
    
    return model, history

if __name__ == "__main__":
    model, history = train()
