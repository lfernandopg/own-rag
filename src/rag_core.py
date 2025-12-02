import json
import pickle
import numpy as np
import chromadb
from chromadb.config import Settings
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, field
from tokenizers import Tokenizer
from rank_bm25 import BM25Okapi

# --- Entidad de Documento ---
@dataclass
class Document:
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Adaptador ONNX (Se mantiene IGUAL) ---
class OnnxEmbeddingAdapter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._load_resources()

    def _load_resources(self):
        self.tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)

        onnx_file = self.model_path / "onnx" / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = self.model_path / "onnx" / "model.onnx"
        print(f"Loading ONNX model from {onnx_file}")
        # Inferencia en CPU
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norm, a_min=1e-9, a_max=None)

    def embed(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        input_names = [i.name for i in self.session.get_inputs()]
        if 'token_type_ids' in input_names:
            inputs['token_type_ids'] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self.session.run(None, inputs)
        embeddings = self._mean_pooling(outputs[0], attention_mask)
        return self._normalize(embeddings).tolist()

class ChromaDBManager:
    def __init__(self, db_path: str, collection_name: str = "docs"):
        # Configuramos persistencia local y desactivamos telemetría
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, docs: List[Document]):
        if not docs: return
        
        # Chroma requiere listas separadas
        ids = [d.id for d in docs]
        embeddings = [d.vector for d in docs]
        contents = [d.content for d in docs]
        metadatas = [d.metadata for d in docs]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        # Normalizar la respuesta de Chroma (que es un diccionario de listas)
        # a una lista plana de diccionarios como esperaba nuestra lógica anterior.
        normalized_results = []
        if results['ids']:
            # Chroma devuelve listas de listas (una por cada query)
            ids = results['ids'][0]
            metas = results['metadatas'][0]
            
            for i, doc_id in enumerate(ids):
                item = {
                    'id': doc_id,
                    'metadata': json.dumps(metas[i]) if metas[i] else "{}", # Stringificamos para mantener compatibilidad
                    # Chroma usa distancia coseno/euclidiana.
                    # RRF espera ranking, no score absoluto, así que el orden importa más que el valor.
                }
                normalized_results.append(item)
                
        return normalized_results

# --- Motor RAG (Hybrid + Parent-Child) ---
class RagEngine:
    def __init__(self, model_path: str, db_path: str):
        self.embedder = OnnxEmbeddingAdapter(model_path)
        # Aquí inyectamos el nuevo manager
        self.db = ChromaDBManager(db_path) 
        
        self.bm25 = None
        self.bm25_corpus_map = [] 
        self.parents_storage = {} 
        
        self.aux_path = Path(db_path) / "aux_indices"
        self.aux_path.mkdir(parents=True, exist_ok=True)
        self._load_indices()

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    def ingest(self, text: str, project_id: str = "default"):
        # 1. Parent-Child Chunking
        parent_chunks = self._chunk_text(text, chunk_size=800, overlap=0)
        child_docs = []
        new_tokenized_corpus = []

        for parent_text in parent_chunks:
            parent_id = str(uuid4())
            self.parents_storage[parent_id] = parent_text 
            
            child_chunks = self._chunk_text(parent_text, chunk_size=150, overlap=30)
            if not child_chunks: continue
            
            vectors = self.embedder.embed(child_chunks)
            
            for i, child_text in enumerate(child_chunks):
                child_id = str(uuid4())
                doc = Document(
                    id=child_id,
                    content=child_text,
                    vector=vectors[i],
                    # Chroma maneja metadatos planos mejor, pero mantenemos estructura
                    metadata={"parent_id": parent_id, "project_id": project_id}
                )
                child_docs.append(doc)
                
                new_tokenized_corpus.append(child_text.lower().split())
                self.bm25_corpus_map.append(child_id)

        # 2. Guardar Vectores
        self.db.add(child_docs)
        
        # 3. Actualizar BM25
        self.bm25 = BM25Okapi(new_tokenized_corpus)
        self._save_indices()
        
        return len(child_docs)

    def search(self, query: str, top_k: int = 3):
        # A. Búsqueda Vectorial
        query_vector = self.embedder.embed([query])[0]
        # Nota: Chroma devuelve resultados ordenados por proximidad
        vec_results = self.db.search(query_vector, limit=top_k * 3)
        
        # B. Búsqueda BM25
        kw_results = []
        if self.bm25:
            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n = np.argsort(doc_scores)[::-1][:top_k * 3]
            for idx in top_n:
                if doc_scores[idx] > 0:
                    kw_results.append({"id": self.bm25_corpus_map[idx], "score": doc_scores[idx]})

        # C. Reciprocal Rank Fusion
        fused_ids = self._rrf(vec_results, kw_results)
        
        # D. Resolver Padres
        final_context = []
        seen_parents = set()
        
        for doc_id in fused_ids:
            if len(final_context) >= top_k: break
            
            # Buscar metadata
            meta = None
            found_vec = next((r for r in vec_results if r['id'] == doc_id), None)
            
            if found_vec:
                # En Chroma hemos guardado JSON string en metadata para compatibilidad
                # pero Chroma devuelve dict real si se guarda como dict.
                # Como en add() pasamos dict directo a Chroma, aquí recuperamos dict directo.
                # Ajuste defensivo:
                raw_meta = found_vec['metadata']
                if isinstance(raw_meta, str):
                    meta = json.loads(raw_meta)
                else:
                    meta = raw_meta
            
            if meta and meta.get('parent_id'):
                p_id = meta['parent_id']
                if p_id not in seen_parents:
                    final_context.append(self.parents_storage.get(p_id, ""))
                    seen_parents.add(p_id)
                    
        return "\n\n".join(final_context)

    def _rrf(self, list_a, list_b, k=60):
        scores = {}
        for rank, item in enumerate(list_a):
            scores[item['id']] = scores.get(item['id'], 0) + (1 / (k + rank + 1))
        for rank, item in enumerate(list_b):
            scores[item['id']] = scores.get(item['id'], 0) + (1 / (k + rank + 1))
        return sorted(scores, key=scores.get, reverse=True)

    def _save_indices(self):
        # Guarda índices auxiliares (BM25 y textos padres)
        with open(self.aux_path / "bm25.pkl", 'wb') as f:
            pickle.dump({'model': self.bm25, 'map': self.bm25_corpus_map}, f)
        with open(self.aux_path / "parents.json", 'w') as f:
            json.dump(self.parents_storage, f)

    def _load_indices(self):
        try:
            with open(self.aux_path / "bm25.pkl", 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['model']
                self.bm25_corpus_map = data['map']
            with open(self.aux_path / "parents.json", 'r') as f:
                self.parents_storage = json.load(f)
        except FileNotFoundError:
            pass