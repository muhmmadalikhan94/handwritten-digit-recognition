import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Class for model evaluation"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, x_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy
    
    def generate_classification_report(self, x_test, y_test):
        """Generate detailed classification report"""
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, 
                                   target_names=[str(i) for i in range(10)]))
    
    def plot_confusion_matrix(self, x_test, y_test):
        """Plot confusion matrix"""
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
