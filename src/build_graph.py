import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
        return int(label)  # fallback if numeric
    except Exception:
        return None

# ------------------------------
# Tokenization functions
# ------------------------------
def tokenize_code(code_text):
    """Tokenize code into meaningful tokens."""
    if not isinstance(code_text, str):
        return []
    
    # Remove extra whitespace and split by common delimiters
    # This regex captures words, operators, punctuation as separate tokens
    tokens = re.findall(r'\w+|[^\w\s]', code_text)
    
    # Filter out empty tokens and normalize
    tokens = [token.strip().lower() for token in tokens if token.strip()]
    return tokens

def tokenize_comment(comment_text):
    """Tokenize comment into words."""
    if not isinstance(comment_text, str):
        return []
    
    # Simple word tokenization for comments
    tokens = re.findall(r'\w+', comment_text.lower())
    return tokens

# ------------------------------
# Feature extraction
# ------------------------------
class TokenFeatureExtractor:
    def __init__(self):
        self.code_vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
        self.comment_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.is_fitted = False
    
    def fit(self, all_code_tokens, all_comment_tokens):
        """Fit the vectorizers on all data."""
        # Prepare text data for vectorizers
        code_texts = [' '.join(tokens) for tokens in all_code_tokens]
        comment_texts = [' '.join(tokens) for tokens in all_comment_tokens]
        
        # Fit vectorizers
        self.code_vectorizer.fit(code_texts)
        self.comment_vectorizer.fit(comment_texts)
        self.is_fitted = True
    
    def get_token_features(self, token, token_type, code_tokens, comment_tokens):
        """Get features for a single token."""
        features = []
        
        # 1. Token type (0=code, 1=comment)
        features.append(float(token_type))
        
        # 2. Token length
        features.append(len(token))
        
        # 3. Is numeric
        features.append(float(token.isdigit()))
        
        # 4. Is alphabetic
        features.append(float(token.isalpha()))
        
        # 5. Has special characters
        features.append(float(not token.isalnum()))
        
        # 6. TF-IDF features (simplified - just check if token appears in vocabulary)
        if token_type == 0:  # code token
            vocab = self.code_vectorizer.vocabulary_
            features.append(float(token in vocab))
        else:  # comment token
            vocab = self.comment_vectorizer.vocabulary_
            features.append(float(token in vocab))
        
        return features

# ------------------------------
# Improved Graph builder
# ------------------------------
def build_improved_graph(code, comment, label, feature_extractor):
    """
    Build a more meaningful graph from code and comment.
    Structure:
    - Code tokens connected in sequence
    - Comment tokens connected in sequence  
    - Bridge connections between code and comment sections
    """
    if not isinstance(code, str) or not isinstance(comment, str):
        return None
    
    # Tokenize
    code_tokens = tokenize_code(code)
    comment_tokens = tokenize_comment(comment)
    
    if len(code_tokens) == 0 and len(comment_tokens) == 0:
        return None
    
    # Create graph
    G = nx.Graph()
    node_features = []
    node_id = 0
    
    # Add code nodes and edges
    code_node_ids = []
    for i, token in enumerate(code_tokens):
        features = feature_extractor.get_token_features(token, 0, code_tokens, comment_tokens)
        node_features.append(features)
        G.add_node(node_id, token=token, type='code')
        code_node_ids.append(node_id)
        
        # Connect to previous code node
        if i > 0:
            G.add_edge(node_id - 1, node_id)
        
        node_id += 1
    
    # Add comment nodes and edges
    comment_node_ids = []
    for i, token in enumerate(comment_tokens):
        features = feature_extractor.get_token_features(token, 1, code_tokens, comment_tokens)
        node_features.append(features)
        G.add_node(node_id, token=token, type='comment')
        comment_node_ids.append(node_id)
        
        # Connect to previous comment node
        if i > 0:
            G.add_edge(node_id - 1, node_id)
        
        node_id += 1
    
    # Add bridge connections between code and comment sections
    if code_node_ids and comment_node_ids:
        # Connect last code node to first comment node
        G.add_edge(code_node_ids[-1], comment_node_ids[0])
        
        # Add a few more strategic connections if both sections are long enough
        if len(code_node_ids) > 2 and len(comment_node_ids) > 2:
            # Connect middle nodes
            mid_code = code_node_ids[len(code_node_ids)//2]
            mid_comment = comment_node_ids[len(comment_node_ids)//2]
            G.add_edge(mid_code, mid_comment)
    
    # Convert to PyTorch Geometric format
    if G.number_of_nodes() == 0:
        return None
    
    # Node features
    if not node_features:
        return None
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index
    if G.number_of_edges() == 0:
        # If no edges, create a self-loop for the single node
        if G.number_of_nodes() == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            return None
    else:
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=G.number_of_nodes()
    )
    
    return data

# ------------------------------
# Main function with two-pass approach
# ------------------------------
def main():
    input_csv = "data/raw/data.csv"
    output_dir = "data/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Rename columns consistently
    df = df.rename(columns={
        "Comments": "comment",
        "Surrounding Code Context": "code",
        "Class": "label"
    })
    
    print(f"📊 Total rows: {len(df)}")
    
    # First pass: collect all tokens to fit feature extractors
    print("🔍 First pass: analyzing tokens for feature extraction...")
    all_code_tokens = []
    all_comment_tokens = []
    valid_rows = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        label = safe_label(row["label"])
        if label is None:
            continue
            
        code_tokens = tokenize_code(row["code"])
        comment_tokens = tokenize_comment(row["comment"])
        
        if len(code_tokens) == 0 and len(comment_tokens) == 0:
            continue
            
        all_code_tokens.append(code_tokens)
        all_comment_tokens.append(comment_tokens)
        valid_rows.append((i, row, label, code_tokens, comment_tokens))
    
    print(f"📈 Found {len(valid_rows)} valid samples")
    
    # Fit feature extractors
    print("🧠 Training feature extractors...")
    feature_extractor = TokenFeatureExtractor()
    feature_extractor.fit(all_code_tokens, all_comment_tokens)
    
    # Second pass: build graphs
    print("🏗️  Second pass: building graphs...")
    saved = 0
    skipped = 0
    
    for i, row, label, code_tokens, comment_tokens in tqdm(valid_rows, desc="Building graphs"):
        graph = build_improved_graph(row["code"], row["comment"], label, feature_extractor)
        
        if graph is None:
            skipped += 1
            continue
        
        # Save graph
        torch.save(graph, os.path.join(output_dir, f"graph_{i}.pt"))
        saved += 1
    
    print(f"\n✅ Finished! Saved {saved} graphs to {output_dir}")
    print(f"⚠️ Skipped: {skipped} rows out of {len(df)}")
    
    # Print some statistics
    if saved > 0:
        print(f"\n📊 Statistics:")
        print(f"  - Average code vocabulary size: {len(feature_extractor.code_vectorizer.vocabulary_)}")
        print(f"  - Average comment vocabulary size: {len(feature_extractor.comment_vectorizer.vocabulary_)}")
        print(f"  - Node feature dimensions: {len(all_code_tokens[0]) if all_code_tokens and all_code_tokens[0] else 'N/A'}")

if __name__ == "__main__":
    main()