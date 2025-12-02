import time
import shutil
from pathlib import Path
from rag_core import RagEngine

# Configuración de rutas
# Ajustamos para que apunte correctamente a donde Docker monta los volúmenes
BASE_DIR = Path(__file__).parent.parent 
MODEL_PATH = BASE_DIR / "data" / "models" / "all-MiniLM-L6-v2"
RERANK_PATH = BASE_DIR / "data" / "models" / "ms-marco-TinyBERT-L-2-v2" # Nueva ruta
DB_PATH = BASE_DIR / "data" / "databases" / "chroma_benchmark"

# Texto de prueba (Lorem Ipsum largo o texto real)
SAMPLE_TEXT = """
La arquitectura hexagonal, o patrón de puertos y adaptadores, es un patrón arquitectónico 
utilizado en el diseño de software. Su objetivo es crear componentes de aplicación 
débilmente acoplados que puedan conectarse fácilmente a su entorno de software 
mediante puertos y adaptadores. Esto hace que los componentes sean intercambiables 
en cualquier nivel y facilita la automatización de las pruebas.

ConfidentialAI es una aplicación diseñada para ejecutarse localmente sin depender de la nube.
Utiliza tecnologías como ONNX para inferencia ligera y ChromaDB para almacenamiento vectorial.
El objetivo es mantener la privacidad de los datos del usuario.

PyInstaller es una herramienta que congela aplicaciones Python en ejecutables independientes.
Para optimizar el tamaño, es crucial evitar librerías pesadas como PyTorch o TensorFlow 
si solo se necesita inferencia.
"""  # Multiplicamos para tener volumen

def run_benchmark():
    # 1. Limpieza inicial (para que el test sea justo)
    if DB_PATH.exists():
        print(f"🧹 Limpiando base de datos previa en {DB_PATH}...")
        shutil.rmtree(DB_PATH)
    
    print("="*60)
    print("🚀 INICIANDO BENCHMARK DE RAG HÍBRIDO (ONNX + CHROMA + BM25)")
    print("="*60)

    # 2. Medir tiempo de carga (Cold Start)
    start_time = time.perf_counter()
    # Inicializamos el motor híbrido
    engine = RagEngine(str(MODEL_PATH), str(RERANK_PATH), str(DB_PATH))
    load_time = time.perf_counter() - start_time
    print(f"⏱️  Carga de Modelos (Cold Start): {load_time:.4f} segundos")

    # 3. Medir tiempo de Ingesta (Embedding + Indexado + BM25)
    print("\n📥 Ingestando documentos...")
    start_time = time.perf_counter()
    # La nueva firma de ingest acepta project_id, usamos el default
    num_chunks = engine.ingest(SAMPLE_TEXT)
    ingest_time = time.perf_counter() - start_time
    print(f"⏱️  Ingesta ({len(SAMPLE_TEXT)} caracteres, {num_chunks} chunks hijos): {ingest_time:.4f} segundos")
    print(f"📊 Velocidad de Ingesta: {len(SAMPLE_TEXT)/ingest_time:.2f} chars/seg")

    # 4. Medir tiempo de Consulta (Retrieval)
    query = "¿Qué es ConfidentialAI y qué tecnologías usa?"
    print(f"\n🔍 Query: '{query}'")
    
    start_time = time.perf_counter()
    
    # CORRECCIÓN AQUÍ: 
    # 1. Usamos .search() en lugar de .query()
    # 2. El resultado ahora es un string (contexto completo), no un DataFrame
    context_result = engine.search(query, top_k=3)
    
    query_time = time.perf_counter() - start_time
    
    print(f"⏱️  Tiempo de Búsqueda Híbrida: {query_time:.4f} segundos")
    
    print("\n--- Contexto Recuperado (Documentos Padre) ---")
    print(context_result)
    print("-" * 60)

if __name__ == "__main__":
    print("="*60)
    if not MODEL_PATH.exists():
        print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
        print("   Asegúrate de que los volúmenes de Docker estén bien montados o ejecuta setup_models.py")
    else:
        print(f"✅ Modelo encontrado en {MODEL_PATH}")
        run_benchmark()