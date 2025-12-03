import json
import pickle
import sqlite3
import numpy as np
import chromadb
from chromadb.config import Settings
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
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

# --- 1. Almacenamiento Eficiente (SQLite) ---
class DocStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT
            )
        ''')
        self.conn.commit()

    def add_document(self, doc_id: str, content: str):
        self.cursor.execute('INSERT OR REPLACE INTO documents VALUES (?, ?)', (doc_id, content))
        self.conn.commit()

    def get_document(self, doc_id: str) -> str:
        self.cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
        result = self.cursor.fetchone()
        return result[0] if result else ""

# --- 2. Adaptador de Embeddings ONNX ---
class OnnxEmbeddingAdapter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        
        onnx_file = self.model_path / "onnx" / "model_quantized.onnx"
        if not onnx_file.exists(): onnx_file = self.model_path / "onnx" / "model.onnx"
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
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        # Limpieza de inputs si el modelo no usa token_type_ids
        model_inputs = [i.name for i in self.session.get_inputs()]
        if 'token_type_ids' not in model_inputs: del inputs['token_type_ids']

        outputs = self.session.run(None, inputs)
        embeddings = self._mean_pooling(outputs[0], attention_mask)
        return self._normalize(embeddings).tolist()

# --- 3. Adaptador de Reranking ONNX (Cross-Encoder) ---
class OnnxCrossEncoder:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        
        onnx_file = self.model_path / "onnx" / "model_quantized.onnx"
        if not onnx_file.exists(): onnx_file = self.model_path / "onnx" / "model.onnx"
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    def predict(self, query: str, documents: List[str]) -> List[float]:
        if not documents: return []
        pairs = [(query, doc) for doc in documents]
        encoded = self.tokenizer.encode_batch(pairs)
        
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        
        # Ejecutar inferencia (logits)
        logits = self.session.run(None, inputs)[0]
        return logits.flatten().tolist()

# --- 4. ChromaDB Manager ---
class ChromaDBManager:
    def __init__(self, db_path: str, collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, docs: List[Document]):
        if not docs: return
        self.collection.add(
            ids=[d.id for d in docs],
            embeddings=[d.vector for d in docs],
            documents=[d.content for d in docs], # Opcional: puedes guardar string vacío para ahorrar espacio
            metadatas=[d.metadata for d in docs]
        )

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        results = self.collection.query(query_embeddings=[query_vector], n_results=limit)
        normalized = []
        if results['ids']:
            ids, metas = results['ids'][0], results['metadatas'][0]
            for i, doc_id in enumerate(ids):
                normalized.append({'id': doc_id, 'metadata': metas[i]})
        return normalized

# --- 5. Motor Principal ---
class RagEngine:
    def __init__(self, embed_model_path: str, rerank_model_path: str, db_path: str):
        self.embedder = OnnxEmbeddingAdapter(embed_model_path)
        self.reranker = OnnxCrossEncoder(rerank_model_path)
        self.db = ChromaDBManager(db_path)
        self.doc_store = DocStore(str(Path(db_path) / "docstore.db"))
        
        self.bm25 = None
        self.bm25_corpus_map = []
        self.aux_path = Path(db_path) / "aux_indices"
        self.aux_path.mkdir(parents=True, exist_ok=True)
        self._load_indices()

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    def ingest(self, text: str, project_id: str = "default"):
        parent_chunks = self._chunk_text(text, chunk_size=900, overlap=90)
        child_docs = []
        new_tokenized_corpus = []

        for parent_text in parent_chunks:
            parent_id = str(uuid4())
            self.doc_store.add_document(parent_id, parent_text) # Guardar en SQLite
            
            child_chunks = self._chunk_text(parent_text, chunk_size=250, overlap=25)
            if not child_chunks: continue
            
            vectors = self.embedder.embed(child_chunks)
            for i, child_text in enumerate(child_chunks):
                child_id = str(uuid4())
                child_docs.append(Document(
                    id=child_id,
                    content=child_text,
                    vector=vectors[i],
                    metadata={"parent_id": parent_id, "project_id": project_id}
                ))
                new_tokenized_corpus.append(child_text.lower().split())
                self.bm25_corpus_map.append(child_id)

        self.db.add(child_docs)
        self.bm25 = BM25Okapi(new_tokenized_corpus)
        self._save_indices()
        return len(child_docs)

    def search(self, query: str, top_k: int = 3):
        # A. Búsqueda Vectorial (Top 50)
        q_vec = self.embedder.embed([query])[0]
        vec_results = self.db.search(q_vec, limit=50)
        
        # B. Búsqueda BM25 (Top 50)
        kw_results = []
        if self.bm25:
            scores = self.bm25.get_scores(query.lower().split())
            top_n = np.argsort(scores)[::-1][:50]
            kw_results = [{"id": self.bm25_corpus_map[i]} for i in top_n if scores[i] > 0]

        # C. RRF Fusion para obtener IDs únicos ordenados por relevancia preliminar
        fused_ids = self._rrf(vec_results, kw_results, k=60)
        
        # --- CAMBIO PRINCIPAL: Reranking sobre HIJOS, retorno de PADRES ---
        
        # 1. Recuperar Textos de los HIJOS para el Reranker
        # Tomamos los top 20 candidatos de la fusión para refinar
        candidates_to_rerank = fused_ids[:20]
        if not candidates_to_rerank: return []

        # Hacemos una consulta directa a Chroma para obtener texto y metadata de estos IDs
        # Esto nos asegura tener el texto del hijo (300 palabras) para el reranker
        data = self.db.collection.get(
            ids=candidates_to_rerank,
            include=['documents', 'metadatas']
        )
        
        # Chroma devuelve listas no ordenadas, creamos un mapa para procesar
        # Estructura: id -> (texto_hijo, metadata)
        doc_map = {
            did: (doc, meta) 
            for did, doc, meta in zip(data['ids'], data['documents'], data['metadatas'])
        }

        # Preparamos listas alineadas para el Reranker
        valid_child_texts = []
        valid_metadatas = []
        
        for cid in candidates_to_rerank:
            if cid in doc_map and doc_map[cid][0]: # Verificar que exista y tenga texto
                text, meta = doc_map[cid]
                valid_child_texts.append(text)
                valid_metadatas.append(meta)

        if not valid_child_texts: return []

        # 2. Reranking (Usando los textos HIJOS)
        # El modelo ahora lee fragmentos de ~300 palabras, que caben en su ventana de 512 tokens.
        print(f"Reranking {valid_child_texts} child texts...")
        scores = self.reranker.predict(query, valid_child_texts)
        print(f"Reranker scores: {scores}")
        # Emparejamos (Metadata, Score) y ordenamos por Score descendente
        # Notar que ya no necesitamos el texto del hijo en el resultado final
        scored_candidates = sorted(zip(valid_metadatas, scores), key=lambda x: x[1], reverse=True)
        
        # 3. Recuperar PADRES para el resultado final
        final_texts = []
        seen_parents = set()
        
        for meta, score in scored_candidates:
            if len(final_texts) >= top_k:
                break
            
            parent_id = meta.get('parent_id')
            if parent_id and parent_id not in seen_parents:
                # AQUÍ ocurre la magia: Usamos el ID del padre para sacar el texto completo de SQLite
                parent_text = self.doc_store.get_document(parent_id)
                if parent_text:
                    final_texts.append(parent_text)
                    seen_parents.add(parent_id)
        
        return final_texts

    def _rrf(self, list_a, list_b, k=60):
        scores = {}
        for rank, item in enumerate(list_a):
            scores[item['id']] = scores.get(item['id'], 0) + (1 / (k + rank + 1))
        for rank, item in enumerate(list_b):
            scores[item['id']] = scores.get(item['id'], 0) + (1 / (k + rank + 1))
        return sorted(scores, key=scores.get, reverse=True)

    def _save_indices(self):
        with open(self.aux_path / "bm25.pkl", 'wb') as f:
            pickle.dump({'model': self.bm25, 'map': self.bm25_corpus_map}, f)

    def _load_indices(self):
        try:
            with open(self.aux_path / "bm25.pkl", 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['model']
                self.bm25_corpus_map = data['map']
        except FileNotFoundError: pass