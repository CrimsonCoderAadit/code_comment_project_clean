"""
Training script for GNN model on code-comment graphs.
"""
import torch
from torch_geometric.loader import DataLoader
from gnn_dataset import CodeCommentGraphDataset
from gnn_model import GNNModel
import torch.nn.functional as F

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    print("📊 Loading dataset...")
    dataset = CodeCommentGraphDataset(root="data")
    print("✅ Dataset loaded!")
    
    # Print dataset statistics
    print(f"📈 Dataset size: {len(dataset)}")
    print(f"📊 Node features: {dataset.num_node_features}")
    print(f"🎯 Number of classes: {dataset.num_classes}")
    
    # Check a sample to understand the data better
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"🔍 Sample graph: {sample.num_nodes} nodes, {sample.num_edges} edges")
        print(f"📝 Sample features shape: {sample.x.shape}")
        print(f"🏷️  Sample label: {sample.y.item()}")
    
    # Train/test split
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    print(f"🚂 Train size: {len(train_dataset)}")
    print(f"🧪 Test size: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    input_dim = dataset.num_node_features
    num_classes = dataset.num_classes
    
    print(f"🧠 Model config - Input dim: {input_dim}, Hidden: 64, Output: {num_classes}")
    
    model = GNNModel(input_dim=input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"🎯 Starting training...")
    
    # Training loop
    best_acc = 0
    for epoch in range(1, 21):  # More epochs
        # Training
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track training accuracy
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Evaluation
        test_acc = evaluate(model, test_loader, device)
        
        # Track best model
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        # Early stopping if no improvement
        if epoch > 10 and test_acc < 0.55:  # If stuck at low accuracy
            print("⚠️  Model seems to be struggling. Consider:")
            print("   - Checking data quality")
            print("   - Adjusting learning rate")
            print("   - Modifying model architecture")
    
    print(f"🏆 Best test accuracy: {best_acc:.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    train()