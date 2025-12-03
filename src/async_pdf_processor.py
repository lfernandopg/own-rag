import pdfplumber
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, AsyncGenerator, Tuple, Generator
from pathlib import Path
import logging

# --- Definir un objeto centinela para señalar el fin de la iteración ---
_SENTINEL = object()


def _get_next_chunk(gen: Generator[Tuple[int, str], None, None]) -> object:
    """
    Función auxiliar para obtener el siguiente elemento de un generador.
    Devuelve un objeto centinela cuando el generador se agota.
    """
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PDFProcessor:
    """
    Una clase para procesar archivos PDF, extrayendo texto y tablas de manera eficiente.
    Permite el procesamiento de archivos individuales o directorios completos de forma asíncrona.
    """

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 150, logs_level: int = logging.INFO):
        """
        Inicializa el procesador de PDF.

        Args:
            max_chunk_size (int): El tamaño máximo de cada trozo de texto.
            chunk_overlap (int): El solapamiento de caracteres entre trozos consecutivos.
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

        rows_lines = []
        for row in table_data[1:]:
            if len(row) == len(header):
                cleaned_row = [(str(c).replace("\n", " ") if c is not None else "") for c in row]
                rows_lines.append("| " + " | ".join(cleaned_row) + " |")

        return "\n".join([header_line, separator_line] + rows_lines)

    # def _process_single_pdf(self, filepath: Path, detect_tables: bool) -> List[str]:
    #     """
    #     Procesa un único archivo PDF para extraer texto y, opcionalmente, tablas.
    #     Esta es una función síncrona y bloqueante, diseñada para ser ejecutada en un hilo separado.
    #     """
    #     chunks = []
    #     try:
    #         with pdfplumber.open(filepath) as pdf:
    #             for page_num, page in enumerate(pdf.pages, 1):
    #                 logging.info(f"Procesando página {page_num} del archivo '{filepath}'...")
    #                 page_text = ""
    #                 if detect_tables:
    #                     tables = page.extract_tables()
    #                     for table_data in tables:
    #                         markdown_table = self._convert_table_to_markdown(table_data)
    #                         if markdown_table:
    #                             chunks.append(markdown_table)

    #                     # Extraer texto que no está en las tablas
    #                     page_text = page.filter(
    #                         lambda obj: all(
    #                             obj.get("y0", 0) < table.bbox[1] or obj.get("y1", 0) > table.bbox[3] or obj.get("x0", 0) < table.bbox[0] or obj.get("x1", 0) > table.bbox[2]
    #                             for table in page.find_tables()
    #                         )
    #                     ).extract_text()
    #                 else:
    #                     # Extraer todo el texto de la página si no se detectan tablas
    #                     page_text = page.extract_text()

    #                 if page_text:
    #                     text = page_text.strip()
    #                     if len(text) > self.max_chunk_size:
    #                         chunks.extend(self.text_splitter.split_text(text))
    #                     elif text:
    #                         chunks.append(text)

    #         if not chunks:
    #             logging.warning(f"No se pudo extraer contenido del PDF '{filepath}'.")
    #         else:
    #             logging.info(f"PDF '{filepath}' procesado. Se generaron {len(chunks)} trozos.")

    #     except Exception as e:
    #         logging.error(f"Error procesando el archivo {filepath}: {e}")

    #     return chunks

    def _generate_chunks_from_pdf_sync(self, filepath: Path, detect_tables: bool) -> Generator[Tuple[int, int, str], None, None]:
        """
        Procesa un único archivo PDF y genera trozos de texto de forma síncrona.
        Esta función es un generador, produciendo trozos a medida que se procesan las páginas.
        """
        try:
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    logging.debug(f"Procesando página {page_num}/{total_pages} del archivo '{filepath}'...")

                    page_chunks = []
                    page_text_content = ""

                    if detect_tables:
                        tables = page.extract_tables()
                        for table_data in tables:
                            markdown_table = self._convert_table_to_markdown(table_data)
                            if markdown_table:
                                page_chunks.append(markdown_table)

                        page_text_content = page.filter(
                            lambda obj: all(
                                obj.get("y0", 0) < table.bbox[1] or obj.get("y1", 0) > table.bbox[3] or obj.get("x0", 0) < table.bbox[0] or obj.get("x1", 0) > table.bbox[2]
                                for table in page.find_tables()
                            )
                        ).extract_text()
                    else:
                        page_text_content = page.extract_text()

                    if page_text_content:
                        text = page_text_content.strip()
                        if len(text) > self.max_chunk_size:
                            page_chunks.extend(self.text_splitter.split_text(text))
                        elif text:
                            page_chunks.append(text)

                    # Cede cada trozo de la página actual antes de pasar a la siguiente
                    for chunk in page_chunks:
                        yield (page_num, total_pages, chunk)

        except Exception as e:
            logging.error(f"Error procesando el archivo {filepath}: {e}")

    async def process_pdfs(
        self, source_path: str, detect_tables: bool = True
    ) -> AsyncGenerator[Tuple[str, int, int, str], None]:
        """
        Procesa archivos PDF desde una ruta (archivo o directorio) de forma asíncrona.

        Args:
            source_path (str): Ruta a un archivo PDF o un directorio que contiene archivos PDF.
            detect_tables (bool): Si es True, detecta y convierte tablas a Markdown.

        Yields:
            AsyncGenerator[Tuple[str, int, int, str], None]: Un generador asíncrono que produce
            tuplas de (ruta_del_archivo, numero_de_pagina, total_paginas, contenido_del_trozo).
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
            return

        init_time = asyncio.get_event_loop().time()
        logging.info(f"Iniciando el procesamiento de {len(pdf_files)} archivos PDF...")

        loop = asyncio.get_running_loop()
        for pdf_file in pdf_files:
            logging.info(f"Iniciando streaming de trozos desde: {pdf_file}")
            # Crea un generador síncrono para el archivo actual
            chunk_generator = self._generate_chunks_from_pdf_sync(pdf_file, detect_tables)

            # Itera sobre el generador en un hilo separado para no bloquear el bucle de eventos
            while True:
                # Pide el siguiente trozo usando la función auxiliar segura
                result = await loop.run_in_executor(None, _get_next_chunk, chunk_generator)

                if result is _SENTINEL:
                    # El generador ha terminado para este archivo
                    logging.info(f"Streaming completado para: {pdf_file}")
                    break

                page_num, total_pages, chunk = result
                logging.debug(f"Generando trozo del archivo: {pdf_file}, página: {page_num}/{total_pages}")
                yield (str(pdf_file), page_num, total_pages, chunk)

        end_time = asyncio.get_event_loop().time()
        logging.info(f"Procesamiento completado en {end_time - init_time:.2f} segundos.")


# --- Bloque de prueba individual ---
async def main():
    """Función principal asíncrona para probar la clase PDFProcessor."""
    print("🚀 Ejecutando prueba individual del módulo 'pdf_processor'...")

    # Ruta de prueba. Asume que está en el directorio padre del script.
    test_path = Path(__file__).parent.parent / "docs"

    if not test_path.exists():
        print(f"❌ Error de prueba: No se encontró el archivo en '{test_path}'.")
        print("Asegúrate de que el archivo o directorio de prueba exista antes de ejecutar la prueba.")
        return

    # --- Inicializar el procesador ---
    # Puedes ajustar max_chunk_size y chunk_overlap aquí
    pdf_processor = PDFProcessor(max_chunk_size=2560, chunk_overlap=250)

    total_chunks = 0

    try:
        print(f"\nProcesando: '{test_path}' con detección de tablas activada...")
        # Itera asíncronamente sobre los trozos a medida que se generan
        async for source_file, page_num, total_pages, chunk in pdf_processor.process_pdfs(
            str(test_path), detect_tables=True
        ):
            if total_chunks < 10000:  # Imprime solo los primeros 10000 trozos para brevedad
                print("\n" + "=" * 50)
                print(f"Fuente: {source_file}")
                print(f"Página: {page_num}/{total_pages}")
                print(f"--- Trozo {total_chunks + 1} (Longitud: {len(chunk)}) ---")
                print("=" * 50)
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                print("\n")
            total_chunks += 1

        print(f"\n✅ Prueba completada. Número total de trozos generados: {total_chunks}")

    except Exception as e:
        logging.error(f"Ha ocurrido un error durante la prueba: {e}")


if __name__ == "__main__":
    # Ejecuta la función principal asíncrona
    asyncio.run(main())
