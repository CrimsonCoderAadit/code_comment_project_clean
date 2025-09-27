import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gnn_model import GNNModel
from gnn_dataset import GNNDataset
from torch_geometric.loader import DataLoader

def load_trained_model(model_path, input_dim=53, hidden_dim=128, output_dim=2):
    """Load the trained GNN model"""
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"✅ Loaded trained model from {model_path}")
    return model

def evaluate_on_synthetic_data(model, synthetic_data_path):
    """Evaluate trained model on synthetic test data"""
    
    print("📊 Loading synthetic test data...")
    
    # Load the synthetic dataset
    dataset = GNNDataset(synthetic_data_path)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"📝 Synthetic dataset size: {len(dataset)}")
    
    # Run inference
    all_predictions = []
    all_labels = []
    
    device = torch.device('cpu')  # Using CPU for inference
    model.to(device)
    
    print("🔍 Running inference on synthetic data...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    return all_predictions, all_labels

def analyze_performance(predictions, true_labels):
    """Analyze model performance on synthetic data"""
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*50)
    print("🎯 SYNTHETIC DATA EVALUATION RESULTS")
    print("="*50)
    print(f"🎯 Test Accuracy on Synthetic Data: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    class_names = ['Not Useful', 'Useful']
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # Performance breakdown
    print("\n📈 Performance Analysis:")
    tn, fp, fn, tp = cm.ravel()
    
    print(f"✅ True Negatives (Correctly identified 'Not Useful'): {tn}")
    print(f"❌ False Positives (Incorrectly labeled as 'Useful'): {fp}")  
    print(f"❌ False Negatives (Incorrectly labeled as 'Not Useful'): {fn}")
    print(f"✅ True Positives (Correctly identified 'Useful'): {tp}")
    
    # Calculate precision and recall for each class
    if tp + fp > 0:
        useful_precision = tp / (tp + fp)
        print(f"🎯 'Useful' Precision: {useful_precision:.3f}")
    
    if tp + fn > 0:
        useful_recall = tp / (tp + fn)
        print(f"🎯 'Useful' Recall: {useful_recall:.3f}")
    
    if tn + fp > 0:
        not_useful_precision = tn / (tn + fn)  
        print(f"🎯 'Not Useful' Precision: {not_useful_precision:.3f}")
    
    return accuracy

def compare_with_original_performance():
    """Compare synthetic data performance with original test performance"""
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    print("🏆 Original Test Set Accuracy: 80.8%")
    print("🤖 Synthetic Test Set Accuracy: [Will be calculated above]")
    print("\n💡 Analysis:")
    print("- If synthetic accuracy ≈ original: Model generalizes well")
    print("- If synthetic accuracy < original: Model may overfit to training data")
    print("- If synthetic accuracy > original: Synthetic data might be easier/different")

def main():
    print("🚀 Starting Model Analysis on Synthetic Data")
    print("="*50)
    
    # Paths
    model_path = "best_upsampled_model.pth"  # Your trained model
    synthetic_data_path = "data/synthetic_test_data.csv"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you have trained your model first!")
        return
    
    if not os.path.exists(synthetic_data_path):
        print(f"❌ Synthetic data not found: {synthetic_data_path}")
        print("💡 Run generate_test_data.py first to create synthetic data!")
        return
    
    try:
        # Load trained model
        model = load_trained_model(model_path)
        
        # Evaluate on synthetic data  
        predictions, true_labels = evaluate_on_synthetic_data(model, synthetic_data_path)
        
        # Analyze performance
        synthetic_accuracy = analyze_performance(predictions, true_labels)
        
        # Compare with original performance
        compare_with_original_performance()
        
        print("\n✨ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Make sure your model and data files are properly formatted")

if __name__ == "__main__":
    main()