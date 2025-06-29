"""
Pinecone Retriever for MedMCQA
Updated for Pinecone SDK v3+ and matches your notebook implementation
"""
from pinecone import Pinecone, ServerlessSpec
import os
from .embedder import MedicalEmbedder
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any

# Load environment variables
load_dotenv()

class MedMCQAPineconeRetriever:
    """
    Pinecone retriever for MedMCQA dataset
    Handles vector search and context retrieval
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the Pinecone retriever
        
        Args:
            data_dir: Data directory (kept for compatibility)
        """
        print("🔧 Initializing Pinecone retriever...")
        
        # Initialize embedder
        self.embedder = MedicalEmbedder()
        
        # Get configuration from environment
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medmcqa-embeddings")
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        print(f"📍 Connecting to Pinecone index: {self.index_name}")
        
        try:
            # Initialize Pinecone with new SDK
            self.pc = Pinecone(api_key=self.api_key)
            
            # Connect to existing index
            self.index = self.pc.Index(self.index_name)
            
            # Test the connection and get stats
            stats = self.index.describe_index_stats()
            print(f"✅ Connected to Pinecone successfully!")
            print(f"📊 Index stats: {stats.get('total_vector_count', 0)} vectors, {stats.get('dimension', 0)} dimensions")
            
        except Exception as e:
            print(f"❌ Failed to connect to Pinecone index '{self.index_name}': {e}")
            print("💡 Please check:")
            print("   - Your PINECONE_API_KEY is correct")
            print("   - Your index name matches the one in Pinecone")
            print("   - Your index exists and is active")
            raise

    def get_relevant_context(self, question: str, k: int = 5, min_score: float = 0.3) -> Tuple[List[Dict], float]:
        """
        Get relevant context for a medical question
        
        Args:
            question: Medical question to search for
            k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            Tuple of (relevant documents, best score)
        """
        try:
            print(f"🔍 Searching for: {question[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_query(question)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True,
                include_values=False
            )
            
            # Process results
            docs = []
            best_score = 0.0
            
            print(f"📋 Found {len(results.matches)} matches")
            
            for i, match in enumerate(results.matches):
                score = match.score
                metadata = match.metadata or {}
                
                print(f"   {i+1}. Score: {score:.3f}")
                
                if score >= min_score:
                    docs.append({
                        'id': match.id,
                        'score': score,
                        'question': metadata.get('question', ''),
                        'options': {
                            'A': metadata.get('opa', ''),
                            'B': metadata.get('opb', ''),
                            'C': metadata.get('opc', ''),
                            'D': metadata.get('opd', '')
                        },
                        'correct_answer': metadata.get('cop', ''),
                        'explanation': metadata.get('exp', ''),
                        'subject': metadata.get('subject_name', ''),
                        'topic': metadata.get('topic_name', ''),
                        'text': metadata.get('text', ''),  # Full concatenated text
                        'context': self._format_single_context(metadata)
                    })
                    
                    if score > best_score:
                        best_score = score
                else:
                    print(f"   ⚠️  Skipping match {i+1} (score {score:.3f} < {min_score})")
            
            print(f"✅ Retrieved {len(docs)} relevant documents (best score: {best_score:.3f})")
            return docs, best_score
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return [], 0.0

    def _format_single_context(self, metadata: Dict) -> str:
        """
        Format a single document's metadata into context text
        
        Args:
            metadata: Document metadata from Pinecone
            
        Returns:
            Formatted context string
        """
        question = metadata.get('question', '')
        options = [
            metadata.get('opa', ''),
            metadata.get('opb', ''),
            metadata.get('opc', ''),
            metadata.get('opd', '')
        ]
        explanation = metadata.get('exp', '')
        subject = metadata.get('subject_name', '')
        
        # Format similar to your notebook structure
        context_parts = []
        
        if question:
            context_parts.append(f"Question: {question}")
        
        if any(options):
            options_text = " | ".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options) if opt])
            context_parts.append(f"Options: {options_text}")
        
        if explanation:
            context_parts.append(f"Explanation: {explanation}")
            
        if subject:
            context_parts.append(f"Subject: {subject}")
        
        return " | ".join(context_parts)

    def format_context_for_llm(self, docs: List[Dict]) -> str:
        """
        Format multiple documents into context for LLM
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            Formatted context string for LLM
        """
        if not docs:
            return "No relevant medical information found."
        
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Reference {i} (Score: {doc.get('score', 0):.3f}):")
            context_parts.append(doc.get('context', ''))
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)

    def search_similar_questions(self, question: str, k: int = 3) -> List[Dict]:
        """
        Search for similar medical questions
        
        Args:
            question: Input medical question
            k: Number of similar questions to return
            
        Returns:
            List of similar questions with metadata
        """
        docs, _ = self.get_relevant_context(question, k=k, min_score=0.2)
        
        similar_questions = []
        for doc in docs:
            similar_questions.append({
                'question': doc.get('question', ''),
                'subject': doc.get('subject', ''),
                'topic': doc.get('topic', ''),
                'score': doc.get('score', 0),
                'explanation': doc.get('explanation', '')[:200] + "..." if len(doc.get('explanation', '')) > 200 else doc.get('explanation', '')
            })
        
        return similar_questions

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Pinecone index statistics
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": list(stats.get('namespaces', {}).keys()) if stats.get('namespaces') else ['default'],
                "environment": self.environment,
                "model": self.embedder.model_name,
                "status": "connected"
            }
            
        except Exception as e:
            return {
                "index_name": self.index_name,
                "error": str(e),
                "total_vectors": "Unknown",
                "status": "error"
            }

    def test_connection(self) -> bool:
        """
        Test the Pinecone connection
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            stats = self.index.describe_index_stats()
            print(f"✅ Connection test successful: {stats.get('total_vector_count', 0)} vectors available")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test the retriever
    print("🧪 Testing Pinecone Retriever")
    
    try:
        retriever = MedMCQAPineconeRetriever()
        
        # Test connection
        if retriever.test_connection():
            # Test search
            test_question = "What is hypertension?"
            docs, score = retriever.get_relevant_context(test_question)
            
            print(f"\n📋 Search Results for: {test_question}")
            print(f"Found {len(docs)} relevant documents")
            print(f"Best score: {score:.3f}")
            
            if docs:
                print(f"\nTop result: {docs[0].get('question', 'N/A')}")
                print(f"Subject: {docs[0].get('subject', 'N/A')}")
            
            # Get stats
            stats = retriever.get_stats()
            print(f"\n📊 Index Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("\n✅ Retriever test completed successfully!")
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}") 