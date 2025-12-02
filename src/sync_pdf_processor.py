import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple
from pathlib import Path
import logging
import time

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SyncPDFProcessor:
    """
    Una clase para procesar archivos PDF de forma síncrona, extrayendo texto y tablas.
    Procesa archivos individuales o directorios y devuelve una lista completa de trozos.
    """

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 150, logs_level: int = logging.INFO):
        """
        Inicializa el procesador de PDF síncrono.

        Args:
            max_chunk_size (int): El tamaño máximo de cada trozo de texto.
            chunk_overlap (int): El solapamiento de caracteres entre trozos consecutivos.
            logs_level (int): El nivel de logging a utilizar.
        """
        if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
            raise ValueError("max_chunk_size debe ser un entero positivo.")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("chunk_overlap debe ser un entero no negativo.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.max_chunk_size = max_chunk_size
        logging.getLogger().setLevel(logs_level)

    def _convert_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convierte datos de tabla extraídos a formato Markdown."""
        if not table_data or not table_data[0]:
            return ""
        header = table_data[0]
        cleaned_header = [(str(h).replace("\n", " ") if h is not None else "") for h in header]
        header_line = "| " + " | ".join(cleaned_header) + " |"
        separator_line = "| " + " | ".join(["---"] * len(cleaned_header)) + " |"
        rows_lines = ["| " + " | ".join([(str(c).replace("\n", " ") if c is not None else "") for c in row]) + " |" for row in table_data[1:] if len(row) == len(header)]
        return "\n".join([header_line, separator_line] + rows_lines)

    def _process_single_pdf(self, filepath: Path, detect_tables: bool) -> List[Tuple[int, str]]:
        """
        Procesa un único archivo PDF y devuelve una lista de tuplas (numero_de_pagina, trozo).
        """
        all_chunks = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logging.info(f"Procesando página {page_num}/{len(pdf.pages)} del archivo '{filepath}'")
                    page_chunks = []
                    page_text_content = ""

                    if detect_tables:
                        tables = page.extract_tables()
                        for table_data in tables:
                            markdown_table = self._convert_table_to_markdown(table_data)
                            if markdown_table:
                                page_chunks.append(markdown_table)

                        page_text_content = page.filter(
                            lambda obj: not any(table.bbox[1] <= obj.get("y0", 0) <= table.bbox[3] and table.bbox[0] <= obj.get("x0", 0) <= table.bbox[2] for table in page.find_tables())
                        ).extract_text()
                    else:
                        page_text_content = page.extract_text()

                    if page_text_content:
                        text = page_text_content.strip()
                        if len(text) > self.max_chunk_size:
                            page_chunks.extend(self.text_splitter.split_text(text))
                        elif text:
                            page_chunks.append(text)

                    for chunk in page_chunks:
                        all_chunks.append((page_num, chunk))
        except Exception as e:
            logging.error(f"Error procesando el archivo {filepath}: {e}")

        return all_chunks

    def process_pdfs(self, source_path: str, detect_tables: bool = True) -> List[Tuple[str, int, str]]:
        """
        Procesa archivos PDF desde una ruta (archivo o directorio) de forma síncrona.

        Args:
            source_path (str): Ruta a un archivo PDF o un directorio que contiene archivos PDF.
            detect_tables (bool): Si es True, detecta y convierte tablas a Markdown.

        Returns:
            List[Tuple[str, int, str]]: Una lista de tuplas que contienen
            (ruta_del_archivo, numero_de_pagina, contenido_del_trozo).
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"La ruta especificada no existe: {source_path}")

        if path.is_dir():
            pdf_files = list(path.glob("*.pdf"))
            logging.info(f"Se encontraron {len(pdf_files)} archivos PDF en el directorio '{source_path}'.")
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files = [path]
        else:
            raise ValueError("La ruta debe ser un archivo .pdf o un directorio.")

        if not pdf_files:
            logging.warning(f"No se encontraron archivos PDF en la ruta '{source_path}'.")
            return []

        init_time = time.time()
        logging.info(f"Iniciando el procesamiento de {len(pdf_files)} archivos PDF...")

        all_chunks_with_metadata = []
        for pdf_file in pdf_files:
            logging.info(f"Procesando archivo: {pdf_file}")
            file_chunks = self._process_single_pdf(pdf_file, detect_tables)
            for page_num, chunk in file_chunks:
                all_chunks_with_metadata.append((str(pdf_file), page_num, chunk))

        end_time = time.time()
        logging.info(f"Procesamiento completado en {end_time - init_time:.2f} segundos.")

        return all_chunks_with_metadata


# --- Bloque de prueba individual ---
def main():
    """Función principal síncrona para probar la clase SyncPDFProcessor."""
    print("🚀 Ejecutando prueba individual del módulo 'sync_pdf_processor'...")

    # Ruta de prueba. Asume que está en el directorio padre del script.
    test_path = Path(__file__).parent.parent / "docs"

    if not test_path.exists():
        print(f"❌ Error de prueba: No se encontró el archivo en '{test_path}'.")
        print("Asegúrate de que el archivo o directorio de prueba exista.")
        return

    # --- Inicializar el procesador ---
    pdf_processor = SyncPDFProcessor(max_chunk_size=2560, chunk_overlap=250)

    try:
        print(f"\nProcesando: '{test_path}' con detección de tablas activada...")

        # Llama al método síncrono para obtener todos los trozos
        all_chunks = pdf_processor.process_pdfs(str(test_path), detect_tables=True)

        print(f"\n✅ Prueba completada. Número total de trozos generados: {len(all_chunks)}")

        # Imprime los primeros 5 trozos para verificar
        print("\n--- Mostrando los primeros 5 trozos ---")
        for i, (source_file, page_num, chunk) in enumerate(all_chunks[:5]):
            print("\n" + "=" * 50)
            print(f"Fuente: {source_file}")
            print(f"Página: {page_num}")
            print(f"--- Trozo {i + 1} (Longitud: {len(chunk)}) ---")
            print("=" * 50)
            print(chunk[:300] + "[...]" if len(chunk) > 300 else chunk)
            print("\n")

    except Exception as e:
        logging.error(f"Ha ocurrido un error durante la prueba: {e}")


if __name__ == "__main__":
    main()
