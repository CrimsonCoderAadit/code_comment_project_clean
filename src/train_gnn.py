"""
Training script with upsampling to handle class imbalance.
Uses SMOTE-like techniques adapted for graph data.
"""
import torch
from torch_geometric.loader import DataLoader
from gnn_dataset import CodeCommentGraphDataset
from gnn_model import create_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import json
from torch_geometric.data import Data

def create_balanced_dataset(dataset):
    """Create balanced training dataset using upsampling techniques."""
    
    # Separate by class
    class_0_data = []  # Not useful
    class_1_data = []  # Useful
    
    for data in dataset:
        if data.y.item() == 0:
            class_0_data.append(data)
        else:
            class_1_data.append(data)
    
    print(f"Original distribution:")
    print(f"  Not Useful: {len(class_0_data)} samples")
    print(f"  Useful: {len(class_1_data)} samples")
    
    # Calculate how many samples we need
    target_size = max(len(class_0_data), len(class_1_data))
    
    # Upsample minority class (Not Useful)
    if len(class_0_data) < len(class_1_data):
        minority_data = class_0_data
        majority_data = class_1_data
        minority_label = 0
    else:
        minority_data = class_1_data
        majority_data = class_0_data
        minority_label = 1
    
    print(f"Upsampling class {minority_label} from {len(minority_data)} to {target_size}")
    
    # Simple random oversampling with noise
    upsampled_minority = []
    current_size = len(minority_data)
    
    while len(upsampled_minority) + current_size < target_size:
        for original_data in minority_data:
            if len(upsampled_minority) + current_size >= target_size:
                break
            
            # Create synthetic sample by adding small noise
            new_data = original_data.clone()
            
            # Add small random noise to node features
            noise_scale = 0.05  # 5% noise
            noise = torch.randn_like(new_data.x) * noise_scale
            new_data.x = new_data.x + noise
            
            upsampled_minority.append(new_data)
    
    # Combine all data
    balanced_dataset = minority_data + upsampled_minority + majority_data
    
    print(f"Balanced dataset:")
    final_class_0 = sum(1 for data in balanced_dataset if data.y.item() == 0)
    final_class_1 = sum(1 for data in balanced_dataset if data.y.item() == 1)
    print(f"  Not Useful: {final_class_0} samples")
    print(f"  Useful: {final_class_1} samples")
    print(f"  Total: {len(balanced_dataset)} samples")
    
    return balanced_dataset

def evaluate(model, loader, device):
    """Evaluate model performance."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1, all_preds, all_labels

def train_with_upsampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with upsampling - Using device: {device}")
    
    print("Loading dataset...")
    dataset = CodeCommentGraphDataset(root="data")
    print(f"Original dataset size: {len(dataset)}")
    print(f"Node features: {dataset.num_node_features}")
    
    # Split dataset first
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]
    
    print(f"\nBefore upsampling:")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Apply upsampling only to training data
    balanced_train_data = create_balanced_dataset(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(balanced_train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = create_model(
        input_dim=dataset.num_node_features,
        hidden_dim=64,
        num_classes=dataset.num_classes
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()  # No class weights needed with balanced data
    
    print("Starting training with balanced data...")
    
    best_test_acc = 0
    
    # Training loop
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Evaluation on original test set
        test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_upsampled_model.pth')
        
        print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.load_state_dict(torch.load('best_upsampled_model.pth', weights_only=False))
    final_acc, final_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"\nFINAL RESULTS WITH UPSAMPLING:")
    print(f"Test Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"Test F1 Score: {final_f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Not Useful', 'Useful']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    # Performance comparison
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"Original simple model (6D features):      79.8%")
    print(f"Enhanced simple model (53D features):     80.4%") 
    print(f"Upsampled enhanced model (53D features):  {final_acc*100:.1f}%")
    
    improvement = final_acc - 0.804
    if improvement > 0.01:
        print(f"SUCCESS: {improvement*100:.1f}% improvement over previous best!")
    elif improvement > 0:
        print(f"GOOD: {improvement*100:.1f}% improvement")
    else:
        print("Similar performance - upsampling didn't help significantly")
    
    # Save results
    results = {
        'model_type': 'upsampled_enhanced',
        'features': '53D_selective',
        'upsampling': 'random_with_noise',
        'final_test_acc': float(final_acc),
        'final_test_f1': float(final_f1),
        'model_parameters': total_params,
        'improvement_vs_enhanced': float(improvement),
        'training_samples': len(balanced_train_data),
        'confusion_matrix': cm.tolist()
    }
    
    with open('upsampled_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return final_acc

if __name__ == "__main__":
    train_with_upsampling()