import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model():
    # Modelo ligero y eficiente (aprox 40MB cuantizado)
    MODEL_ID = "Xenova/all-MiniLM-L6-v2"

    # Directorio local relativo
    base_dir = Path(__file__).parent.parent / "data" / "models" / "all-MiniLM-L6-v2"

    print(f"⬇️  Descargando modelo {MODEL_ID} en {base_dir}...")

    # Descargamos solo lo necesario para ONNX
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=base_dir,
        allow_patterns=["*.onnx", "tokenizer.json", "config.json", "special_tokens_map.json", "tokenizer_config.json"],
        local_dir_use_symlinks=False
    )

    print("✅ Modelo descargado correctamente.")

if __name__ == "__main__":
    download_model()
