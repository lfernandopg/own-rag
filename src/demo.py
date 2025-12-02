import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ajustar el path para importar los módulos hermanos si es necesario
sys.path.append(str(Path(__file__).parent))

from async_pdf_processor import PDFProcessor
from rag_core import RagEngine

# --- Configuración de Rutas (Igual que en benchmark.py) ---
BASE_DIR = Path(__file__).parent.parent 
MODEL_PATH = BASE_DIR / "data" / "models" / "all-MiniLM-L6-v2"
RERANK_PATH = BASE_DIR / "data" / "models" / "ms-marco-TinyBERT-L-2-v2"
# Usamos una DB específica para la demo para no mezclar con pruebas
DB_PATH = BASE_DIR / "data" / "databases" / "chroma_demo"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("🤖  RAG PROTOTYPE DEMO - CONFIDENTIAL AI")
    print("    Stack: ONNX (CPU) + ChromaDB + Hybrid Search")
    print("=" * 60 + "\n")

async def process_documents(folder_path: str) -> Dict[str, str]:
    """
    Usa el procesador asíncrono para leer los PDFs.
    Retorna un diccionario {nombre_archivo: texto_completo}.
    """
    processor = PDFProcessor(max_chunk_size=1000, chunk_overlap=100, logs_level=40) # 40 = ERROR para reducir ruido
    file_contents: Dict[str, List[str]] = {}
    
    print(f"📂  Leyendo archivos desde: {folder_path}")
    print("⏳  Procesando PDFs (Extracción de texto y tablas)...")
    
    start_time = time.time()
    count = 0
    
    # Procesamiento asíncrono
    try:
        async for filepath, page_num, chunk in processor.process_pdfs(folder_path, detect_tables=True):
            filename = Path(filepath).name
            if filename not in file_contents:
                file_contents[filename] = []
                print(f"   📄 Detectado: {filename}")
            
            file_contents[filename].append(chunk)
            count += 1
            # Simple indicador de progreso
            print(f"      ↳ Extrayendo: {filename} (Pág {page_num})", end="\r")
            
    except Exception as e:
        print(f"\n❌  Error procesando PDFs: {e}")
        return {}

    print(f"\n✅  Procesamiento finalizado: {count} fragmentos extraídos en {time.time() - start_time:.2f}s.\n")
    
    # Unir fragmentos para entregar documentos "enteros" al RagEngine
    # Esto permite que el RagEngine aplique su propia lógica de Parent-Child splitting
    documents = {name: "\n".join(chunks) for name, chunks in file_contents.items()}
    return documents

def run_rag_demo():
    clear_screen()
    print_header()

    # 1. Verificación de Modelos
    if not MODEL_PATH.exists() or not RERANK_PATH.exists():
        print("❌  Error: Modelos no encontrados.")
        print("    Por favor ejecuta primero: python src/setup_models.py")
        return

    # 2. Carga del Motor (Cold Start)
    print("⚙️   Cargando Modelos ONNX y Base de Datos Vectorial...")
    start_load = time.time()
    engine = RagEngine(str(MODEL_PATH), str(RERANK_PATH), str(DB_PATH))
    print(f"✅  Motor listo en {time.time() - start_load:.2f}s.\n")

    # 3. Selección de Carpeta
    while True:
        folder_input = input("ptr  Introduce la ruta de la carpeta con PDFs (o 'enter' para usar ./docs): ").strip()
        if not folder_input:
            target_folder = BASE_DIR / "docs"
        else:
            target_folder = Path(folder_input)
        
        if target_folder.exists() and target_folder.is_dir():
            break
        print("❌  Ruta inválida. Intenta de nuevo.")

    # 4. Procesamiento e Ingesta
    # Ejecutamos la parte asíncrona
    raw_documents = asyncio.run(process_documents(str(target_folder)))
    
    if not raw_documents:
        print("⚠️  No se encontraron documentos válidos o texto extraíble.")
    else:
        print("🧠  Ingestando en RAG (Embedding + Indexado + BM25)...")
        total_chunks = 0
        for filename, content in raw_documents.items():
            print(f"   ↳ Indexando: {filename}...", end="")
            # Ingesta en el motor
            n_chunks = engine.ingest(content, project_id="demo_v1")
            total_chunks += n_chunks
            print(f" Hecho ({n_chunks} sub-chunks creados)")
        print(f"✅  Base de conocimiento actualizada. Total vectores: {total_chunks}\n")

    # 5. Bucle de Preguntas
    print("=" * 60)
    print("💬  SISTEMA LISTO. Escribe 'salir' para terminar.")
    print("=" * 60)

    while True:
        query = input("\nPregunta ➤ ")
        if query.lower() in ['salir', 'exit', 'quit']:
            break
        
        if not query.strip():
            continue

        print("🔍  Buscando y Re-rankeando información relevante...")
        start_q = time.time()
        
        # Búsqueda
        try:
            # engine.search devuelve un string con los contextos unidos
            context_chunks = engine.search(query, top_k=3)
            
            elapsed = time.time() - start_q
            
            print(f"\n--- 📄 Contexto Recuperado (Tiempo: {elapsed:.3f}s) ---")
            if not context_chunks:
                print("⚠️  No se encontró información relevante en los documentos.")
            else:
                # Mostramos el contexto recuperado
                print("Fragmentos relevantes:  \n")
                for chunk in context_chunks:
                    print("-" * 60)
                    print(chunk)
            print("-" * 60)
            
        except Exception as e:
            print(f"❌  Error durante la búsqueda: {e}")

if __name__ == "__main__":
    try:
        run_rag_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo finalizada por el usuario.")