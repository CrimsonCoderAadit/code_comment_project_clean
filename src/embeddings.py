"""
Selective embeddings system - only the highest quality features.
Uses proven statistical features + targeted TF-IDF without noise.
"""
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import re
import string
from collections import Counter

class SelectiveTokenEmbeddings:
    def __init__(self):
        # Fixed dimensions for consistency
        self.tfidf_dim = 32
        
        # TF-IDF models (only the most effective ones)
        self.code_tfidf = None
        self.comment_tfidf = None
        self.code_pca = None
        self.comment_pca = None
        
        # Vocabularies for features
        self.code_vocab = set()
        self.comment_vocab = set()
        self.code_token_freq = Counter()
        self.comment_token_freq = Counter()
        
        # High-value programming keywords
        self.programming_keywords = {
            'if', 'else', 'for', 'while', 'return', 'function', 'class', 'def',
            'int', 'float', 'string', 'bool', 'void', 'null', 'true', 'false',
            'public', 'private', 'static', 'const', 'var', 'let', 'import', 'include',
            'main', 'printf', 'scanf', 'malloc', 'free', 'sizeof', 'struct', 'enum',
            'break', 'continue', 'switch', 'case', 'default', 'typedef', 'extern'
        }
        
        self.is_fitted = False
        
    def clean_tokenize_code(self, code_text):
        """Clean, focused code tokenization."""
        if not isinstance(code_text, str):
            return []
        
        code_text = re.sub(r'\s+', ' ', code_text.strip())
        tokens = re.findall(r'\w+|[^\w\s]', code_text)
        tokens = [token.strip().lower() for token in tokens if token.strip()]
        
        # Keep only meaningful tokens
        meaningful_punct = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '++', '--'}
        filtered_tokens = []
        for token in tokens:
            if token.isalnum() or token in meaningful_punct:
                filtered_tokens.append(token)
                
        return filtered_tokens
    
    def clean_tokenize_comment(self, comment_text):
        """Clean, focused comment tokenization."""
        if not isinstance(comment_text, str):
            return []
        
        # Remove comment markers and normalize
        comment_text = re.sub(r'[/*#]+', ' ', comment_text)
        comment_text = re.sub(r'\s+', ' ', comment_text.strip())
        
        # Extract meaningful words only
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', comment_text.lower())
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def extract_statistical_features(self, token, token_type, context_tokens):
        """Extract exactly 21 high-quality statistical features."""
        features = []
        
        # Core features (7)
        features.extend([
            float(token_type),  # 0=code, 1=comment
            len(token),
            float(token.isdigit()),
            float(token.isalpha()),
            float(token.isupper()),
            float(token.islower()),
            float(any(c in string.punctuation for c in token)),
        ])
        
        # Programming context features (5)
        features.extend([
            float(token in self.programming_keywords),
            float(token.startswith('_')),
            float(token.endswith('_')),
            float('_' in token),
            float(any(c.isdigit() for c in token)),
        ])
        
        # Frequency-based features (3)
        if token_type == 0:
            freq = self.code_token_freq.get(token, 0)
            max_freq = max(self.code_token_freq.values()) if self.code_token_freq else 1
        else:
            freq = self.comment_token_freq.get(token, 0)
            max_freq = max(self.comment_token_freq.values()) if self.comment_token_freq else 1
        
        features.extend([
            freq / max_freq,  # Normalized frequency
            float(freq > 1),   # Is frequent
            float(freq > 5),   # Is very frequent
        ])
        
        # Context position features (3)
        if context_tokens and len(context_tokens) > 0:
            features.extend([
                float(token in context_tokens[:3]),  # In first 3
                float(token in context_tokens[-3:]), # In last 3
                context_tokens.count(token) / len(context_tokens),  # Relative frequency
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Vocabulary membership (2)
        features.extend([
            float(token in self.code_vocab),
            float(token in self.comment_vocab),
        ])
        
        # Length category (1)
        if len(token) <= 2:
            length_category = 0.0
        elif len(token) <= 5:
            length_category = 0.5
        else:
            length_category = 1.0
        features.append(length_category)
        
        assert len(features) == 21, f"Expected 21 features, got {len(features)}"
        return features
    
    def fit(self, code_texts, comment_texts):
        """Train selective embeddings with only high-quality features."""
        print("Training selective embeddings (53D features)...")
        
        # Clean tokenization
        print("Tokenizing...")
        code_tokens_list = [self.clean_tokenize_code(text) for text in tqdm(code_texts, desc="Code")]
        code_tokens_list = [tokens for tokens in code_tokens_list if len(tokens) > 0]
        
        comment_tokens_list = [self.clean_tokenize_comment(text) for text in tqdm(comment_texts, desc="Comments")]
        comment_tokens_list = [tokens for tokens in comment_tokens_list if len(tokens) > 0]
        
        # Build vocabularies
        print("Building vocabularies...")
        for tokens in code_tokens_list:
            self.code_vocab.update(tokens)
            self.code_token_freq.update(tokens)
            
        for tokens in comment_tokens_list:
            self.comment_vocab.update(tokens)
            self.comment_token_freq.update(tokens)
        
        # Filter to high-frequency tokens only
        min_freq = 3  # Higher threshold for quality
        self.code_vocab = {token for token in self.code_vocab if self.code_token_freq[token] >= min_freq}
        self.comment_vocab = {token for token in self.comment_vocab if self.comment_token_freq[token] >= min_freq}
        
        print(f"  Code vocabulary: {len(self.code_vocab)} high-quality tokens")
        print(f"  Comment vocabulary: {len(self.comment_vocab)} high-quality tokens")
        
        # Focused TF-IDF models
        print("Training focused TF-IDF models...")
        
        if code_tokens_list:
            code_texts_clean = [' '.join(tokens) for tokens in code_tokens_list]
            self.code_tfidf = TfidfVectorizer(
                max_features=80,  # Focused feature set
                ngram_range=(1, 2),
                min_df=3,  # Higher threshold
                max_df=0.7  # Remove very common terms
            )
            tfidf_matrix = self.code_tfidf.fit_transform(code_texts_clean)
            
            # Consistent PCA dimensions
            self.code_pca = PCA(n_components=self.tfidf_dim)
            self.code_pca.fit(tfidf_matrix.toarray())
        
        if comment_tokens_list:
            comment_texts_clean = [' '.join(tokens) for tokens in comment_tokens_list]
            self.comment_tfidf = TfidfVectorizer(
                max_features=60,  # Focused feature set
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.7
            )
            tfidf_matrix = self.comment_tfidf.fit_transform(comment_texts_clean)
            
            # Consistent PCA dimensions 
            self.comment_pca = PCA(n_components=self.tfidf_dim)
            self.comment_pca.fit(tfidf_matrix.toarray())
        
        self.is_fitted = True
        print(f"Selective embeddings ready! Total dimensions: {self.get_feature_dim()}")
        
        # Show feature breakdown
        print(f"Feature breakdown:")
        print(f"  - Statistical features: 21D")
        print(f"  - TF-IDF features: {self.tfidf_dim}D")
        print(f"  - Total: {self.get_feature_dim()}D")
    
    def get_token_embedding(self, token, token_type, context_tokens=None):
        """Get selective high-quality embedding."""
        if not self.is_fitted:
            raise ValueError("Embeddings not fitted!")
        
        features = []
        
        # 1. Statistical features (21 dimensions) - proven to work
        statistical_features = self.extract_statistical_features(token, token_type, context_tokens or [])
        features.extend(statistical_features)
        
        # 2. Targeted TF-IDF features (32 dimensions) - domain-specific patterns
        if token_type == 0 and self.code_tfidf and self.code_pca:  # Code token
            try:
                tfidf_vec = self.code_tfidf.transform([token]).toarray()
                if tfidf_vec.shape[1] > 0:
                    pca_vec = self.code_pca.transform(tfidf_vec)[0]
                    features.extend(pca_vec.tolist())
                else:
                    features.extend([0.0] * self.tfidf_dim)
            except:
                features.extend([0.0] * self.tfidf_dim)
        elif token_type == 1 and self.comment_tfidf and self.comment_pca:  # Comment token
            try:
                tfidf_vec = self.comment_tfidf.transform([token]).toarray()
                if tfidf_vec.shape[1] > 0:
                    pca_vec = self.comment_pca.transform(tfidf_vec)[0]
                    features.extend(pca_vec.tolist())
                else:
                    features.extend([0.0] * self.tfidf_dim)
            except:
                features.extend([0.0] * self.tfidf_dim)
        else:
            features.extend([0.0] * self.tfidf_dim)
        
        # Verify exact dimensions
        expected_dim = 21 + self.tfidf_dim  # 53 total
        if len(features) != expected_dim:
            print(f"Warning: Expected {expected_dim} features, got {len(features)}")
            if len(features) < expected_dim:
                features.extend([0.0] * (expected_dim - len(features)))
            else:
                features = features[:expected_dim]
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self):
        """Return exact feature dimension."""
        return 21 + self.tfidf_dim  # 53 dimensions total
    
    def save(self, filepath):
        """Save embeddings."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'tfidf_dim': self.tfidf_dim,
            'code_tfidf': self.code_tfidf,
            'comment_tfidf': self.comment_tfidf,
            'code_pca': self.code_pca,
            'comment_pca': self.comment_pca,
            'code_vocab': self.code_vocab,
            'comment_vocab': self.comment_vocab,
            'code_token_freq': self.code_token_freq,
            'comment_token_freq': self.comment_token_freq,
            'programming_keywords': self.programming_keywords,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Selective embeddings saved to {filepath}")
    
    def load(self, filepath):
        """Load embeddings."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for key, value in data.items():
            setattr(self, key, value)
        
        print(f"Selective embeddings loaded from {filepath}")

# Compatibility alias
AdvancedTokenEmbeddings = SelectiveTokenEmbeddings

def create_advanced_embeddings(csv_path, save_path="data/selective_embeddings.pkl"):
    """Create selective high-quality embeddings."""
    import pandas as pd
    
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'Surrounding Code Context' in df.columns:
        df = df.rename(columns={'Surrounding Code Context': 'code'})
    if 'Comments' in df.columns:
        df = df.rename(columns={'Comments': 'comment'})
    
    code_texts = df['code'].fillna('').astype(str).tolist()
    comment_texts = df['comment'].fillna('').astype(str).tolist()
    
    embedder = SelectiveTokenEmbeddings()
    embedder.fit(code_texts, comment_texts)
    embedder.save(save_path)
    
    return embedder

if __name__ == "__main__":
    print("Creating selective high-quality embeddings...")
    embedder = create_advanced_embeddings("data/raw/data.csv")
    print("Selective embeddings ready!")