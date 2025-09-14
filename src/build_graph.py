import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import re
import numpy as np
from embeddings import AdvancedTokenEmbeddings

# ------------------------------
# Safe label conversion
# ------------------------------
def safe_label(label):
    """Convert labels to integers safely."""
    if pd.isna(label):
        return None
    label_str = str(label).strip().lower()
    mapping = {
        "not useful": 0,
        "useful": 1
    }
    if label_str in mapping:
        return mapping[label_str]
    try:
        return int(label)
    except Exception:
        return None

# ------------------------------
# Advanced tokenization
# ------------------------------
def advanced_tokenize_code(code_text):
    """Enhanced code tokenization with better preprocessing."""
    if not isinstance(code_text, str):
        return []
    
    # Normalize whitespace
    code_text = re.sub(r'\s+', ' ', code_text.strip())
    
    # Better tokenization for code
    tokens = re.findall(r'\w+|[^\w\s]', code_text)
    tokens = [token.strip().lower() for token in tokens if token.strip()]
    
    return tokens

def advanced_tokenize_comment(comment_text):
    """Enhanced comment tokenization."""
    if not isinstance(comment_text, str):
        return []
    
    # Clean comment markers
    comment_text = re.sub(r'[/*#]+', ' ', comment_text)
    comment_text = re.sub(r'\s+', ' ', comment_text.strip())
    
    # Extract meaningful words
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', comment_text.lower())
    tokens = [token for token in tokens if len(token) >= 2]
    
    return tokens

# ------------------------------
# Enhanced graph builder with better structure
# ------------------------------
def build_enhanced_graph(code, comment, label, embedder):
    """
    Build enhanced graph with better structure and richer features.
    
    Graph structure:
    1. Code tokens in sequence with skip connections
    2. Comment tokens in sequence
    3. Strategic bridges between code and comment
    4. Attention-like connections for important tokens
    """
    if not isinstance(code, str) or not isinstance(comment, str):
        return None
    
    # Tokenize with advanced methods
    code_tokens = advanced_tokenize_code(code)
    comment_tokens = advanced_tokenize_comment(comment)
    
    if len(code_tokens) == 0 and len(comment_tokens) == 0:
        return None
    
    # Create graph with enhanced structure
    G = nx.Graph()
    node_features = []
    node_id = 0
    
    # Add code nodes with skip connections
    code_node_ids = []
    for i, token in enumerate(code_tokens):
        # Get rich embeddings with context
        features = embedder.get_token_embedding(token, 0, code_tokens)
        node_features.append(features)
        G.add_node(node_id, token=token, type='code', pos=i)
        code_node_ids.append(node_id)
        
        # Sequential connections
        if i > 0:
            G.add_edge(node_id - 1, node_id)
        
        # Skip connections for better information flow
        if i > 1:
            G.add_edge(node_id - 2, node_id, weight=0.5)  # Skip-1 connection
        if i > 2:
            G.add_edge(node_id - 3, node_id, weight=0.3)  # Skip-2 connection
        
        node_id += 1
    
    # Add comment nodes with skip connections
    comment_node_ids = []
    for i, token in enumerate(comment_tokens):
        # Get rich embeddings with context
        features = embedder.get_token_embedding(token, 1, comment_tokens)
        node_features.append(features)
        G.add_node(node_id, token=token, type='comment', pos=i)
        comment_node_ids.append(node_id)
        
        # Sequential connections
        if i > 0:
            G.add_edge(node_id - 1, node_id)
        
        # Skip connections
        if i > 1:
            G.add_edge(node_id - 2, node_id, weight=0.5)
        
        node_id += 1
    
    # Enhanced bridge connections
    if code_node_ids and comment_node_ids:
        # Multiple bridge connections for better information flow
        
        # 1. Connect start and end points
        G.add_edge(code_node_ids[0], comment_node_ids[0], weight=0.8)  # Start-start
        G.add_edge(code_node_ids[-1], comment_node_ids[-1], weight=0.8)  # End-end
        
        # 2. Connect middle points if sequences are long enough
        if len(code_node_ids) >= 3 and len(comment_node_ids) >= 3:
            mid_code = len(code_node_ids) // 2
            mid_comment = len(comment_node_ids) // 2
            G.add_edge(code_node_ids[mid_code], comment_node_ids[mid_comment], weight=0.7)
        
        # 3. Cross connections for semantic similarity (simplified)
        # Connect a few strategic pairs
        n_connections = min(3, len(code_node_ids), len(comment_node_ids))
        for i in range(n_connections):
            code_idx = i * len(code_node_ids) // n_connections
            comment_idx = i * len(comment_node_ids) // n_connections
            if code_idx < len(code_node_ids) and comment_idx < len(comment_node_ids):
                G.add_edge(code_node_ids[code_idx], comment_node_ids[comment_idx], weight=0.6)
    
    # Add self-loops for important tokens (like attention mechanism)
    all_node_ids = code_node_ids + comment_node_ids
    for node_id in all_node_ids:
        G.add_edge(node_id, node_id, weight=1.0)  # Self-loop
    
    # Convert to PyTorch Geometric format
    if G.number_of_nodes() == 0:
        return None
    
    if not node_features:
        return None
    
    # Stack features
    try:
        x = torch.tensor(np.vstack(node_features), dtype=torch.float)
    except Exception as e:
        print(f"Error stacking features: {e}")
        return None
    
    # Create edge index and edge weights
    edges = list(G.edges(data=True))
    if len(edges) == 0:
        # Create self-loops if no edges
        edge_index = torch.tensor([[i, i] for i in range(G.number_of_nodes())]).t().contiguous()
        edge_weight = torch.ones(G.number_of_nodes())
    else:
        edge_list = [(u, v) for u, v, _ in edges]
        edge_weights = [data.get('weight', 1.0) for _, _, data in edges]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create data object with edge weights
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(-1),  # Add edge features
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=G.number_of_nodes()
    )
    
    return data

# ------------------------------
# Main function with enhanced pipeline
# ------------------------------
def main():
    input_csv = "data/raw/data.csv"
    output_dir = "data/graphs"
    embeddings_path = "data/simplified_embeddings.pkl"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Rename columns
    df = df.rename(columns={
        "Comments": "comment",
        "Surrounding Code Context": "code", 
        "Class": "label"
    })
    
    print(f"📊 Total rows: {len(df)}")
    
    # Load or create advanced embeddings
    if os.path.exists(embeddings_path):
        print(f"📂 Loading existing embeddings from {embeddings_path}")
        embedder = AdvancedTokenEmbeddings()
        embedder.load(embeddings_path)
    else:
        print("🚀 Creating new advanced embeddings...")
        from embeddings import create_advanced_embeddings
        embedder = create_advanced_embeddings(input_csv, embeddings_path)
    
    print(f"🎯 Feature dimension: {embedder.get_feature_dim()}")
    
    # Process data
    valid_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        label = safe_label(row["label"])
        if label is None:
            continue
            
        code_tokens = advanced_tokenize_code(row["code"])
        comment_tokens = advanced_tokenize_comment(row["comment"])
        
        if len(code_tokens) == 0 and len(comment_tokens) == 0:
            continue
            
        valid_rows.append((i, row, label))
    
    print(f"📈 Processing {len(valid_rows)} valid samples...")
    
    # Build enhanced graphs
    saved = 0
    skipped = 0
    
    for i, row, label in tqdm(valid_rows, desc="Building enhanced graphs"):
        graph = build_enhanced_graph(row["code"], row["comment"], label, embedder)
        
        if graph is None:
            skipped += 1
            continue
        
        # Save graph
        torch.save(graph, os.path.join(output_dir, f"graph_{i}.pt"))
        saved += 1
    
    print(f"\n✅ Enhanced graph building complete!")
    print(f"📊 Saved: {saved} graphs")
    print(f"⚠️  Skipped: {skipped} graphs") 
    print(f"🎯 Feature dimensions: {embedder.get_feature_dim()}")
    print(f"💾 Embeddings saved at: {embeddings_path}")

if __name__ == "__main__":
    main()