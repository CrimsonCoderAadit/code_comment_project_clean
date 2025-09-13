"""
Embedding system for code and comment tokens.
Provides rich semantic features to replace simple statistical features.
"""
import os
import pickle
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re

class TokenEmbeddings:
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.code_word2vec = None
        self.comment_word2vec = None
        self.code_tfidf = None
        self.comment_tfidf = None
        self.is_fitted = False
        
    def tokenize_code(self, code_text):
        """Tokenize code into meaningful tokens."""
        if not isinstance(code_text, str):
            return []
        
        # Split by common code delimiters and operators
        tokens = re.findall(r'\w+|[^\w\s]', code_text)
        tokens = [token.strip().lower() for token in tokens if token.strip()]
        return tokens
    
    def tokenize_comment(self, comment_text):
        """Tokenize comment into words."""
        if not isinstance(comment_text, str):
            return []
        
        # Simple word tokenization for natural language
        tokens = re.findall(r'\w+', comment_text.lower())
        return tokens
    
    def fit(self, code_texts, comment_texts):
        """
        Train embeddings on the corpus of code and comments.
        
        Args:
            code_texts: List of code strings
            comment_texts: List of comment strings
        """
        print("🚀 Training embeddings...")
        
        # Tokenize all texts
        print("📝 Tokenizing code...")
        code_tokens_list = [self.tokenize_code(text) for text in tqdm(code_texts)]
        code_tokens_list = [tokens for tokens in code_tokens_list if len(tokens) > 0]
        
        print("💬 Tokenizing comments...")
        comment_tokens_list = [self.tokenize_comment(text) for text in tqdm(comment_texts)]  
        comment_tokens_list = [tokens for tokens in comment_tokens_list if len(tokens) > 0]
        
        # Train Word2Vec models
        print("🧠 Training Word2Vec for code...")
        if len(code_tokens_list) > 0:
            self.code_word2vec = Word2Vec(
                sentences=code_tokens_list,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
        
        print("🧠 Training Word2Vec for comments...")
        if len(comment_tokens_list) > 0:
            self.comment_word2vec = Word2Vec(
                sentences=comment_tokens_list,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
        
        # Train TF-IDF for fallback features
        print("📊 Training TF-IDF...")
        code_texts_clean = [' '.join(tokens) for tokens in code_tokens_list]
        comment_texts_clean = [' '.join(tokens) for tokens in comment_tokens_list]
        
        if len(code_texts_clean) > 0:
            self.code_tfidf = TfidfVectorizer(max_features=50, stop_words=None)
            self.code_tfidf.fit(code_texts_clean)
        
        if len(comment_texts_clean) > 0:
            self.comment_tfidf = TfidfVectorizer(max_features=30, stop_words='english')
            self.comment_tfidf.fit(comment_texts_clean)
        
        self.is_fitted = True
        print("✅ Embedding training complete!")
        
        # Print vocabulary stats
        if self.code_word2vec:
            print(f"📈 Code vocabulary: {len(self.code_word2vec.wv)} tokens")
        if self.comment_word2vec:
            print(f"💭 Comment vocabulary: {len(self.comment_word2vec.wv)} tokens")
    
    def get_token_embedding(self, token, token_type):
        """
        Get embedding for a single token.
        
        Args:
            token: String token
            token_type: 0 for code, 1 for comment
            
        Returns:
            numpy array of embedding features
        """
        if not self.is_fitted:
            raise ValueError("Embeddings not fitted! Call fit() first.")
        
        features = []
        
        # Word2Vec embedding
        if token_type == 0 and self.code_word2vec:  # Code token
            if token in self.code_word2vec.wv:
                embedding = self.code_word2vec.wv[token]
            else:
                # Unknown token - use zero vector
                embedding = np.zeros(self.embedding_dim)
            features.extend(embedding.tolist())
            
        elif token_type == 1 and self.comment_word2vec:  # Comment token
            if token in self.comment_word2vec.wv:
                embedding = self.comment_word2vec.wv[token]
            else:
                # Unknown token - use zero vector  
                embedding = np.zeros(self.embedding_dim)
            features.extend(embedding.tolist())
            
        else:
            # Fallback - zero embedding
            features.extend([0.0] * self.embedding_dim)
        
        # Add basic statistical features (keep some from original)
        features.extend([
            float(token_type),  # Token type
            len(token),         # Length
            float(token.isdigit()),    # Is numeric
            float(token.isalpha()),    # Is alphabetic
            float(not token.isalnum()) # Has special chars
        ])
        
        return np.array(features)
    
    def get_feature_dim(self):
        """Return the total feature dimension."""
        return self.embedding_dim + 5  # embedding + 5 statistical features
    
    def save(self, filepath):
        """Save the fitted embeddings."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted embeddings!")
        
        data = {
            'embedding_dim': self.embedding_dim,
            'code_word2vec': self.code_word2vec,
            'comment_word2vec': self.comment_word2vec,
            'code_tfidf': self.code_tfidf,
            'comment_tfidf': self.comment_tfidf,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Embeddings saved to {filepath}")
    
    def load(self, filepath):
        """Load pre-fitted embeddings."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embedding_dim = data['embedding_dim']
        self.code_word2vec = data['code_word2vec']
        self.comment_word2vec = data['comment_word2vec']
        self.code_tfidf = data['code_tfidf']
        self.comment_tfidf = data['comment_tfidf']
        self.is_fitted = data['is_fitted']
        
        print(f"📂 Embeddings loaded from {filepath}")

# Convenience function for integration with build_graph.py
def create_embeddings_from_csv(csv_path, save_path="embeddings.pkl"):
    """
    Create and save embeddings from a CSV file.
    
    Args:
        csv_path: Path to CSV with 'code' and 'comment' columns
        save_path: Where to save the fitted embeddings
    """
    import pandas as pd
    
    print(f"📊 Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Handle different column names
    if 'Surrounding Code Context' in df.columns:
        df = df.rename(columns={'Surrounding Code Context': 'code'})
    if 'Comments' in df.columns:
        df = df.rename(columns={'Comments': 'comment'})
    
    code_texts = df['code'].fillna('').astype(str).tolist()
    comment_texts = df['comment'].fillna('').astype(str).tolist()
    
    # Create and fit embeddings
    embedder = TokenEmbeddings(embedding_dim=64)
    embedder.fit(code_texts, comment_texts)
    
    # Save for later use
    embedder.save(save_path)
    
    return embedder

if __name__ == "__main__":
    # Example usage - create embeddings from your dataset
    print("🚀 Creating embeddings from dataset...")
    
    csv_path = "data/raw/data.csv"  # Your original dataset
    embedder = create_embeddings_from_csv(csv_path, "data/embeddings.pkl")
    
    print("✅ Done! Embeddings saved and ready to use.")