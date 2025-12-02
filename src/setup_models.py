import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_models():
    # Definir rutas base
    base_path = Path(__file__).parent.parent / "data" / "models"
    
    # 1. Modelo de Embeddings
    embed_model_id = "Xenova/all-MiniLM-L6-v2"
    embed_dir = base_path / "all-MiniLM-L6-v2"
    
    # 2. Modelo de Reranking
    reranker_model_id = "Xenova/ms-marco-TinyBERT-L-2-v2"
    reranker_dir = base_path / "ms-marco-TinyBERT-L-2-v2"

    print(f"⬇️  Descargando modelos en {base_path}...")

    # Descarga Embeddings
    print(f"   - Embeddings: {embed_model_id}")
    snapshot_download(
        repo_id=embed_model_id,
        local_dir=embed_dir,
        allow_patterns=["*.onnx", "tokenizer.json", "config.json", "special_tokens_map.json", "tokenizer_config.json"],
        local_dir_use_symlinks=False
    )

    # Descarga Reranker
    print(f"   - Reranker: {reranker_model_id}")
    snapshot_download(
        repo_id=reranker_model_id,
        local_dir=reranker_dir,
        allow_patterns=["*.onnx", "tokenizer.json", "config.json", "special_tokens_map.json", "tokenizer_config.json"],
        local_dir_use_symlinks=False
    )

    print("✅ Todos los modelos descargados correctamente.")

if __name__ == "__main__":
    download_models()