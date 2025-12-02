import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from pathlib import Path


def convert_table_to_markdown(table_data: List[List[str]]) -> str:
    """Convierte una lista de listas (datos de tabla) a una tabla en formato Markdown."""
    markdown_table = ""
    if not table_data:
        return ""

    # Limpiar celdas None o vacías para un formato limpio
    # pdfplumber puede devolver celdas con saltos de línea, los reemplazamos
    cleaned_header = [(str(h).replace("\n", " ") if h is not None else "") for h in table_data[0]]

    header = "| " + " | ".join(cleaned_header) + " |"
    markdown_table += header + "\n"

    separator = "| " + " | ".join(["---"] * len(cleaned_header)) + " |"
    markdown_table += separator + "\n"

    for row in table_data[1:]:
        cleaned_row = [(str(c).replace("\n", " ") if c is not None else "") for c in row]
        # Asegurarse de que la fila tenga el mismo número de columnas que el encabezado
        if len(cleaned_row) == len(cleaned_header):
            row_str = "| " + " | ".join(cleaned_row) + " |"
            markdown_table += row_str + "\n"

    return markdown_table


def process_pdf_to_parent_chunks(filepath: str, max_chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """
    Extrae texto y tablas de un PDF usando pdfplumber, convirtiendo las tablas a Markdown.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"El archivo no fue encontrado en la ruta: {filepath}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    parent_chunks = []

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            # 1. Extraer y procesar tablas primero
            tables = page.extract_tables()
            for table_data in tables:
                markdown_table = convert_table_to_markdown(table_data)
                if markdown_table.strip():
                    parent_chunks.append(markdown_table)

            # 2. Extraer texto que NO está en las tablas
            # pdfplumber tiene una forma ingeniosa de hacer esto:
            # Filtramos los objetos de texto que no están dentro de los bounding_box de las tablas.
            text_outside_tables = page.filter(
                lambda obj: all(
                    obj.get("y0", 0) < table.bbox[1] or obj.get("y1", 0) > table.bbox[3] or obj.get("x0", 0) < table.bbox[0] or obj.get("x1", 0) > table.bbox[2] for table in page.find_tables()
                )
            ).extract_text()

            if text_outside_tables:
                text = text_outside_tables.strip()
                if len(text) > max_chunk_size:
                    sub_chunks = text_splitter.split_text(text)
                    parent_chunks.extend(sub_chunks)
                else:
                    parent_chunks.append(text)

    if not parent_chunks:
        print(f"Advertencia: No se pudo extraer texto o tablas del PDF '{filepath}'.")
        return []

    print(f"✅ PDF procesado con pdfplumber: '{filepath}'. Texto y tablas divididos en {len(parent_chunks)} trozos padre.")
    return parent_chunks


# --- El bloque de prueba individual se mantiene igual ---
if __name__ == "__main__":
    print("🚀 Ejecutando prueba individual del módulo 'pdf_processor'...")

    test_pdf_path = Path(__file__).parent.parent / "constitucion.pdf"

    if not test_pdf_path.exists():
        print(f"❌ Error de prueba: No se encontró el archivo '{test_pdf_path}'.")
    else:
        try:
            parent_texts = process_pdf_to_parent_chunks(str(test_pdf_path))
            if parent_texts:
                print("\n--- Ejemplo de Trozos Padre Generados ---")
                for i, chunk in enumerate(parent_texts[:100]):
                    print(f"--- Trozo {i+1} (Longitud: {len(chunk)}) ---")
                    print(chunk)
                    print("-" * 40)
                print(f"\nNúmero total de trozos padre generados: {len(parent_texts)}")
        except Exception as e:
            print(f"Ha ocurrido un error durante la prueba: {e}")
