import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Importar librería de OpenAI ---
try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: La librería 'openai' no está instalada.")
    print("   Ejecuta: pip install openai")
    sys.exit(1)

# Ajustar el path para importar los módulos hermanos (rag_core, etc.)
sys.path.append(str(Path(__file__).parent))

from async_pdf_processor import PDFProcessor
from rag_core import RagEngine

# --- Configuración de Rutas ---
BASE_DIR = Path(__file__).parent.parent 
MODEL_PATH = BASE_DIR / "data" / "models" / "all-MiniLM-L6-v2"
RERANK_PATH = BASE_DIR / "data" / "models" / "ms-marco-TinyBERT-L-2-v2"
DB_PATH = BASE_DIR / "data" / "databases" / "chroma_demo"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("🤖  RAG DEMO + OPENAI DIRECT")
    print("    RAG: Local (ONNX + Chroma) | LLM: gpt-4o-mini (OpenAI)")
    print("=" * 60 + "\n")

def get_openai_client() -> Optional[OpenAI]:
    """Inicializa el cliente de OpenAI buscando la API Key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No se encontró la variable de entorno OPENAI_API_KEY.")
        api_key = input("🔑  Por favor, introduce tu API Key de OpenAI: ").strip()
    
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)

def stream_openai_response(client: OpenAI, context: str, query: str):
    """Envía el prompt a OpenAI y muestra la respuesta en streaming."""
    
    # Construcción del Prompt (System + User con Contexto)
    system_prompt = (
        "Eres un asistente experto y preciso. "
        "Usa EXCLUSIVAMENTE el contexto proporcionado para responder a la pregunta del usuario. "
        "Cita las fuentes si es posible. "
        "Si la respuesta no está en el contexto, di 'No tengo información suficiente en los documentos'."
    )
    
    user_content = f"""
    --- CONTEXTO RECUPERADO ---
    {context}
    ---------------------------
    
    Pregunta: {query}
    """

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",  # Puedes cambiar a "gpt-4o" o "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=True,
            temperature=0  # Temperatura baja para mayor fidelidad al contexto
        )

        print("\n🤖  Respuesta OpenAI:\n" + "-" * 40)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n" + "-" * 40)

    except Exception as e:
        print(f"\n❌  Error de OpenAI: {e}")

# --- Funciones de Procesamiento (Igual que antes) ---

async def process_documents(folder_path: str) -> Dict[str, str]:
    # Reutilizamos tu lógica existente de async_pdf_processor
    processor = PDFProcessor(max_chunk_size=2560, chunk_overlap=250, logs_level=40)
    file_contents: Dict[str, List[str]] = {}

    print(f"📂  Leyendo archivos desde: {folder_path}")
    count = 0
    try:
        async for filepath, page_num, total_pages, chunk in processor.process_pdfs(folder_path, detect_tables=True):
            filename = Path(filepath).name
            if filename not in file_contents: file_contents[filename] = []
            file_contents[filename].append(chunk)
            count += 1
            # print(f"      ↳ Extrayendo: {filename} (Pág {page_num}/{total_pages}) || Chunk Size: {len(chunk)}", end="\r")
            print(f"      ↳ Extrayendo: {filename} (Pág {page_num}/{total_pages})", end="\r")

    except Exception as e:
        print(f"\n❌  Error procesando PDFs: {e}")
        return {}

    print(f"\n✅  Procesamiento finalizado: {count} fragmentos extraídos.\n")
    documents = {name: "\n".join(chunks) for name, chunks in file_contents.items()}
    return documents

def run_rag_demo():
    clear_screen()
    print_header()

    # 1. Configurar OpenAI
    client = get_openai_client()
    if not client:
        print("❌  No se puede continuar sin API Key. Saliendo.")
        return

    # 2. Verificación de Modelos Locales
    if not MODEL_PATH.exists() or not RERANK_PATH.exists():
        print("❌  Error: Modelos RAG locales no encontrados.")
        print("    Ejecuta primero: python src/setup_models.py")
        return

    # 3. Carga del Motor RAG Local
    print("⚙️   Cargando Motor RAG Local (Embeddings + VectorDB)...")
    start_load = time.time()
    engine = RagEngine(str(MODEL_PATH), str(RERANK_PATH), str(DB_PATH))
    print(f"✅  Motor listo en {time.time() - start_load:.2f}s.\n")

    # 4. Ingesta de Documentos
    target_folder = BASE_DIR / "docs"
    folder_input = input(f"📂  Carpeta de PDFs (Enter para '{target_folder}'): ").strip()
    if folder_input: 
        target_folder = Path(folder_input)
    
    if target_folder.exists() and target_folder.is_dir():
        # Ejecutamos el procesador asíncrono
        raw_documents = asyncio.run(process_documents(str(target_folder)))
        if raw_documents:
            print("🧠  Actualizando base vectorial...")
            for filename, content in raw_documents.items():
                engine.ingest(content, project_id="demo_openai")
                print(f"   ✓ Indexado: {filename}")
            print("✅  Ingesta completada.\n")
    else:
        print("⚠️  Carpeta no válida o vacía. Usando base de datos existente.")

    # 5. Bucle de Chat
    print("=" * 60)
    print("💬  SISTEMA LISTO. Escribe 'salir' para terminar.")
    print("=" * 60)

    while True:
        query = input("\nPregunta ➤ ")
        if query.lower() in ['salir', 'exit', 'quit']:
            break
        if not query.strip():
            continue

        print("🔍  Buscando información relevante en local...")
        start_q = time.time()
        
        # Búsqueda Híbrida + Rerank
        context_chunks = engine.search(query, top_k=3)
        elapsed = time.time() - start_q
        
        print(f"   ✓ {len(context_chunks)} fragmentos recuperados en {elapsed:.2f}s.")

        if not context_chunks:
            print("⚠️  No se encontró información relevante en los documentos.")
        else:
            # Mostramos el contexto recuperado
            print("Fragmentos relevantes:  \n")
            for chunk in context_chunks:
                print("-" * 60)
                print(chunk)
        print("-" * 60)

        if not context_chunks:
            print("⚠️  No se encontró contexto relevante. El modelo responderá sin contexto.")
            context_text = "No hay información disponible en los documentos."
        else:
            context_text = "\n\n".join(context_chunks)

        # Llamada directa a OpenAI
        print("📡  Enviando a OpenAI (gpt-4o-mini)...")
        stream_openai_response(client, context_text, query)

if __name__ == "__main__":
    try:
        run_rag_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo finalizada.")
