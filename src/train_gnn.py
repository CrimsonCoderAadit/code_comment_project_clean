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

    print("Processing...")
    dataset = CodeCommentGraphDataset(root="data")
    print("✅ Dataset loaded!")

    # Train/test split
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    input_dim = dataset.num_node_features
    num_classes = len(torch.unique(dataset.data.y))
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")

    model = GNNModel(input_dim=input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        # Evaluation each epoch
        acc = evaluate(model, test_loader, device)
        print(f"Validation Accuracy after epoch {epoch}: {acc:.4f}")

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
