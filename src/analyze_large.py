import os
import torch
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gnn_model import create_model
from torch_geometric.loader import DataLoader

def load_trained_model(model_path, input_dim=53, hidden_dim=64, num_classes=2):
    """Load the trained GNN model"""
    model = create_model(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    print(f"✅ Loaded trained model from {model_path}")
    return model

def evaluate_on_large_synthetic_data(model, large_synthetic_graphs_dir):
    """Evaluate trained model on large synthetic test data"""
    
    print("📊 Loading large synthetic test graphs...")
    
    # Load large synthetic graph files directly with weights_only=False
    graph_files = glob.glob(os.path.join(large_synthetic_graphs_dir, "large_synthetic_graph_*.pt"))
    
    if not graph_files:
        print(f"❌ No large synthetic graphs found in {large_synthetic_graphs_dir}")
        return [], []
    
    print(f"📁 Found {len(graph_files)} large synthetic graph files")
    
    # Load all synthetic graphs with proper torch.load settings
    synthetic_graphs = []
    failed_loads = 0
    
    for graph_file in sorted(graph_files):
        try:
            # Fix: Use weights_only=False for PyTorch Geometric data objects
            graph = torch.load(graph_file, map_location='cpu', weights_only=False)
            synthetic_graphs.append(graph)
        except Exception as e:
            print(f"❌ Error loading {graph_file}: {e}")
            failed_loads += 1
    
    if len(synthetic_graphs) == 0:
        print("❌ No graphs could be loaded successfully!")
        return [], []
    
    print(f"✅ Successfully loaded {len(synthetic_graphs)} large synthetic graphs")
    if failed_loads > 0:
        print(f"⚠️ Failed to load {failed_loads} graphs")
    
    test_loader = DataLoader(synthetic_graphs, batch_size=32, shuffle=False)
    
    print(f"📝 Large synthetic dataset size: {len(synthetic_graphs)}")
    
    # Run inference
    all_predictions = []
    all_labels = []
    
    device = torch.device('cpu')  # Using CPU for inference
    model.to(device)
    
    print("🔍 Running inference on large synthetic data...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch.x.float(), batch.edge_index, batch.batch)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    return all_predictions, all_labels

def analyze_large_performance(predictions, true_labels):
    """Analyze model performance on large synthetic data"""
    
    if len(predictions) == 0 or len(true_labels) == 0:
        print("❌ No predictions or labels to analyze!")
        return 0.0
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*60)
    print("🎯 LARGE SYNTHETIC DATA EVALUATION RESULTS (100 samples)")
    print("="*60)
    print(f"🎯 Test Accuracy on Large Synthetic Data: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Check if we have both classes in the results
    unique_labels = set(true_labels)
    unique_preds = set(predictions)
    
    print(f"\n📊 Label Distribution:")
    print(f"True labels: {dict(zip(*np.unique(true_labels, return_counts=True)))}")
    print(f"Predictions: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    # Detailed classification report with error handling
    print("\n📋 Detailed Classification Report:")
    try:
        class_names = ['Not Useful', 'Useful']
        if len(unique_labels) >= 2:
            print(classification_report(true_labels, predictions, 
                                      target_names=class_names, 
                                      zero_division=0))
        else:
            print("⚠️ Only one class present in true labels - limited evaluation possible")
            for label in unique_labels:
                class_name = class_names[label] if label < len(class_names) else f"Class {label}"
                count = sum(1 for l in true_labels if l == label)
                correct = sum(1 for t, p in zip(true_labels, predictions) if t == label and t == p)
                print(f"{class_name}: {correct}/{count} correct")
    except Exception as e:
        print(f"❌ Error in classification report: {e}")
    
    # Confusion matrix with error handling
    print("\n🔢 Confusion Matrix:")
    try:
        cm = confusion_matrix(true_labels, predictions)
        print(cm)
        
        # Performance breakdown if we have a 2x2 matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n📈 Performance Analysis:")
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
            
            if tn + fn > 0:
                not_useful_precision = tn / (tn + fn)  
                print(f"🎯 'Not Useful' Precision: {not_useful_precision:.3f}")
        
    except Exception as e:
        print(f"❌ Error in confusion matrix: {e}")
    
    return accuracy

def compare_all_performance(large_synthetic_accuracy):
    """Compare all test performances"""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*60)
    print("🏆 Original Test Set Accuracy (Real Data):     80.8%")
    print("🤖 Small Synthetic Test Set (16 samples):     62.5%")
    print(f"🚀 Large Synthetic Test Set (100 samples):    {large_synthetic_accuracy*100:.1f}%")
    
    # Compare with small synthetic test
    small_synthetic_acc = 0.625
    difference_from_small = large_synthetic_accuracy - small_synthetic_acc
    difference_from_original = large_synthetic_accuracy - 0.808
    
    print(f"\n💡 Analysis:")
    print(f"Difference from original: {difference_from_original*100:+.1f} percentage points")
    print(f"Difference from small synthetic: {difference_from_small*100:+.1f} percentage points")
    
    if abs(difference_from_original) < 0.05:  # Within 5%
        print("✅ Model generalizes excellently - similar performance across data types")
    elif difference_from_original < -0.1:
        print("⚠️ Significant performance drop on synthetic data indicates:")
        print("   - Model overfits to training data patterns")
        print("   - Synthetic data exposes model weaknesses")
        print("   - Need for better regularization or data augmentation")
    elif difference_from_original < -0.05:
        print("⚠️ Moderate performance drop suggests limited generalization")
    else:
        print("📈 Model performs better on synthetic data - may indicate:")
        print("   - Synthetic examples are clearer/more structured")
        print("   - LLM generates prototypical examples")
    
    # Stability analysis
    if abs(difference_from_small) < 0.05:
        print(f"\n🔒 Consistent performance across synthetic data sizes (stable)")
    else:
        print(f"\n📊 Performance varies with dataset size - may indicate sampling effects")

def main():
    print("🚀 Starting Comprehensive Model Analysis on Large Synthetic Data (100 samples)")
    print("="*80)
    
    # Paths
    model_path = "best_upsampled_model.pth"  # Your trained model
    large_synthetic_graphs_dir = "data/large_synthetic_graphs"  # Large synthetic graph directory
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you have trained your model first!")
        return
    
    if not os.path.exists(large_synthetic_graphs_dir):
        print(f"❌ Large synthetic graphs directory not found: {large_synthetic_graphs_dir}")
        print("💡 Run build_large_synthetic_graphs.py first to create large synthetic graphs!")
        return
    
    try:
        # Load trained model
        model = load_trained_model(model_path)
        
        # Evaluate on large synthetic data  
        predictions, true_labels = evaluate_on_large_synthetic_data(model, large_synthetic_graphs_dir)
        
        if len(predictions) == 0:
            print("❌ No predictions generated - check graph loading issues")
            return
        
        # Analyze performance
        large_synthetic_accuracy = analyze_large_performance(predictions, true_labels)
        
        # Compare with all previous performance
        compare_all_performance(large_synthetic_accuracy)
        
        print("\n✨ Comprehensive analysis complete!")
        
        # Save results
        results = {
            'large_synthetic_accuracy': float(large_synthetic_accuracy),
            'original_accuracy': 0.808,
            'small_synthetic_accuracy': 0.625,
            'num_large_synthetic_samples': len(predictions),
            'predictions': predictions,
            'true_labels': true_labels,
            'balanced_dataset': True,
            'sample_size': 100
        }
        
        import json
        with open('large_synthetic_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: large_synthetic_evaluation_results.json")
        
        # Statistical significance note
        print(f"\n📈 Statistical Note:")
        print(f"With 100 balanced samples, this evaluation provides more reliable")
        print(f"statistical significance than the 16-sample test.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("💡 Check that all dependencies are installed and files are accessible")

if __name__ == "__main__":
    main()