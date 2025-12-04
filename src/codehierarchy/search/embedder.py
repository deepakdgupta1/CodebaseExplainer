from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import pickle
import logging

class HighQualityEmbedder:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', dimension: int = 768):
        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate normalized embeddings for a batch of texts.
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )

    def build_index(self, summaries: Dict[str, str]) -> Tuple[faiss.Index, Dict[int, str]]:
        """
        Build FAISS index from summaries.
        Returns (index, id_mapping).
        """
        node_ids = list(summaries.keys())
        texts = list(summaries.values())
        
        if not texts:
            # Return empty index
            index = faiss.IndexFlatIP(self.dimension)
            return index, {}
            
        logging.info(f"Generating embeddings for {len(texts)} items...")
        embeddings = self.encode_batch(texts)
        
        # Create IVF index
        # Heuristic for n_list (number of clusters): 4 * sqrt(N)
        n_list = min(int(4 * np.sqrt(len(texts))), len(texts) // 39)
        if n_list < 1: n_list = 1
        
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(
            quantizer, 
            self.dimension, 
            n_list, 
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Train
        logging.info("Training FAISS index...")
        index.train(embeddings.astype('float32'))
        
        # Add
        logging.info("Adding vectors to index...")
        index.add(embeddings.astype('float32'))
        
        # Create mapping int_id -> node_id
        mapping = {i: nid for i, nid in enumerate(node_ids)}
        
        return index, mapping

    def save_index(self, index: faiss.Index, mapping: Dict[int, str], path: Path):
        """
        Save index and mapping to disk.
        """
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path / "vector.index"))
        with open(path / "mapping.pkl", 'wb') as f:
            pickle.dump(mapping, f)
            
    def load_index(self, path: Path) -> Tuple[faiss.Index, Dict[int, str]]:
        """
        Load index and mapping from disk.
        """
        index = faiss.read_index(str(path / "vector.index"))
        with open(path / "mapping.pkl", 'rb') as f:
            mapping = pickle.load(f)
        return index, mapping
