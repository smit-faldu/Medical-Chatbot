"""
Medical Text Embedder using Sentence Transformers
Handles text embedding for medical questions and content
"""
import os
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

class MedicalEmbedder:
    """
    Medical text embedder using all-MiniLM-L6-v2 model
    Optimized for medical question-answer embeddings
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the medical embedder
        
        Args:
            model_name: Name of the sentence transformer model
        """
        # Use the model from your .env or default to all-MiniLM-L6-v2
        self.model_name = model_name or os.getenv("PINECONE_MODEL", "all-MiniLM-L6-v2")
        
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded successfully")
        print(f"📏 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            numpy array embedding for the query
        """
        embedding = self.embed([query])
        return embedding[0]  # Return single embedding
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple documents
        
        Args:
            documents: List of document texts
            
        Returns:
            numpy array of document embeddings
        """
        return self.embed(documents)
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }

# For backward compatibility
class PineconeConnectionManager:
    """
    Legacy class name for backward compatibility
    """
    def __init__(self):
        print("⚠️  PineconeConnectionManager is deprecated. Use MedicalEmbedder instead.")
        self.embedder = MedicalEmbedder()
    
    def __getattr__(self, name):
        return getattr(self.embedder, name) 