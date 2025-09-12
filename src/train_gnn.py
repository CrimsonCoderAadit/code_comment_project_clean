import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from gnn_dataset import CodeCommentGraphDataset
from gnn_model import GCNClassifier

def train():
    dataset = CodeCommentGraphDataset(root="data")
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNClassifier(input_dim=1, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = torch.nn.functional.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss {total_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
    acc = correct / len(test_dataset)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
