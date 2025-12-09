from tensorflow import keras
from data.data_loader import DataLoader
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from config.config import *

def test():
    """Testing pipeline"""
    print("\n" + "="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - TESTING")
    print("="*60 + "\n")
    
    # Load data
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    (x_train, y_train), (x_test, y_test) = data_loader.get_data()
    
    # Load trained model
    print(f"Loading model from: {MODEL_SAVE_PATH}")
    model = keras.models.load_model(MODEL_SAVE_PATH)
    print("Model loaded successfully!\n")
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(x_test, y_test)
    evaluator.generate_classification_report(x_test, y_test)
    evaluator.plot_confusion_matrix(x_test, y_test)
    
    # Visualize predictions
    visualizer = Visualizer()
    visualizer.plot_sample_predictions(model, x_test, y_test, num_samples=10)
    
    print("\nTesting completed! Results saved in 'results/' directory.")

if __name__ == "__main__":
    test()
