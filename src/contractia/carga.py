"""Carga de documentos PDF/DOCX desde una carpeta (celda 2 del notebook)."""
from __future__ import annotations
import os
from typing import List, Optional, Tuple

from .logging_setup import get_logger

log = get_logger("carga")


def procesar_documentos_carpeta(folder_path: str) -> Tuple[Optional[List], Optional[str]]:
    documentos_combinados: List = []
    texto_completo = ""
    full_folder_path = os.path.join(os.getcwd(), folder_path)

    if not os.path.exists(full_folder_path):
        full_folder_path = folder_path
        if not os.path.exists(full_folder_path):
            log.warning("La carpeta '%s' no existe.", folder_path)
            return None, None

    archivos_en_carpeta = sorted(os.listdir(full_folder_path))
    if not archivos_en_carpeta: return None, None

    log.info("Procesando carpeta: %s", full_folder_path)
    for file_name in archivos_en_carpeta:
        file_path = os.path.join(full_folder_path, file_name)
        if not os.path.isfile(file_path): continue
        log.info("  - Cargando: %s", file_name)
        try:
            from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
            if file_name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_name.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            docs = loader.load()
            documentos_combinados.extend(docs)
            # P2 fix: separador entre archivos para no fusionar finales con inicios
            if texto_completo:
                texto_completo += f"\n\n=== ARCHIVO: {file_name} ===\n\n"
            texto_completo += "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            log.warning("  No se pudo cargar %s: %s", file_name, e)

    return documentos_combinados, texto_completo
