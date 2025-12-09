import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """Class for visualization utilities"""
    
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss', marker='o')
        ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(model, x_test, y_test, num_samples=10):
        """Visualize sample predictions"""
        predictions = model.predict(x_test[:num_samples])
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(x_test[i].squeeze(), cmap='gray')
            pred_label = np.argmax(predictions[i])
            true_label = np.argmax(y_test[i])
            confidence = predictions[i][pred_label] * 100
            
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(f'Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}', 
                            color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

