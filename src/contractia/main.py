"""Entry point: _build_llm, _save_report, main()."""
from __future__ import annotations
import time
from typing import Any, List, Optional

from .config import CFG
from .logging_setup import get_logger, setup_logger
from .carga import procesar_documentos_carpeta
from .metricas import TokenCounterCallback
from .auditoria import ejecutar_auditoria_contrato
from .grafo import construir_indice_nodos_por_cid
from .informe import render_auditoria_markdown
from .chat import iniciar_chat_interactivo

log = get_logger("main")


def _build_llm():
    from langchain_google_vertexai import ChatVertexAI

    if not CFG.enable_llm:
        raise RuntimeError("CFG.enable_llm=False. Actívalo para usar el LLM.")
    try:
        log.info("Inicializando LLM: %s", CFG.modelo_principal)
        return ChatVertexAI(
            model_name=CFG.modelo_principal,
            temperature=CFG.temperature,
            timeout=CFG.timeout_segundos,
            max_output_tokens=CFG.max_output_tokens,
        )
    except Exception as e:
        log.warning("No se pudo iniciar '%s': %s. Usando fallback.", CFG.modelo_principal, e)
        return ChatVertexAI(
            model_name=CFG.modelo_fallback,
            temperature=CFG.temperature,
            timeout=CFG.timeout_segundos,
            max_output_tokens=CFG.max_output_tokens,
        )


def _save_report(md_text: str, filename: Optional[str] = None) -> None:
    filename = filename or CFG.reporte_md
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_text or "")
        log.info("Informe guardado en '%s'.", filename)
    except Exception as e:
        log.warning("No se pudo guardar el informe: %s", e)


def main() -> None:
    import vertexai
    from IPython.display import display, Markdown

    setup_logger()

    try:
        vertexai.init(project=CFG.project_id, location=CFG.location)
        log.info("Vertex AI inicializado. Proyecto: %s", CFG.project_id)
    except Exception as e:
        log.warning("configurar_entorno_vertexai() avisó: %s", e)

    try:
        llm = _build_llm()
        log.info("LLM de Vertex AI inicializado.")
    except Exception as e:
        log.error("No se pudo inicializar el LLM: %s", e)
        return

    log.info("PASO 1: Procesando el contrato desde '%s'...", CFG.ruta_contrato)
    try:
        _, texto_contrato = procesar_documentos_carpeta(CFG.ruta_contrato)
    except Exception as e:
        log.error("Error leyendo el contrato: %s", e)
        return

    if not texto_contrato:
        log.error("No se encontraron documentos en '%s'. Abortando.", CFG.ruta_contrato)
        return

    token_counter = TokenCounterCallback()
    callbacks: List[Any] = [token_counter]

    start_time = time.time()
    try:
        resultado = ejecutar_auditoria_contrato(
            texto_contrato=texto_contrato,
            llm=llm,
            callbacks=callbacks,
        )
    except Exception as e:
        log.error("Error ejecutando el pipeline de auditoría: %s", e)
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - start_time
    log.info("AUDITORÍA COMPLETADA en %.2fs", elapsed)
    log.info("USO DE TOKENS:\n%s", token_counter.resumen())

    try:
        md = render_auditoria_markdown(resultado)
        md += f"\n\n---\n*Tiempo de ejecución: {elapsed:.2f}s*\n\n"
        md += f"## Uso de Tokens\n\n{token_counter.resumen()}\n"
        try:
            display(Markdown(md))
        except Exception:
            print(md)
    except Exception as e:
        log.warning("Error renderizando informe: %s", e)
        md = "Sin datos."

    _save_report(md)

    if resultado.get("abortado_por_seguridad"):
        return

    if "grafo" in resultado and "secciones" in resultado:
        indice_nodos = construir_indice_nodos_por_cid(resultado["grafo"])
        iniciar_chat_interactivo(
            resultado["grafo"],
            resultado["secciones"],
            resultado["indice_secciones"],
            indice_nodos,
            llm,
            callbacks=callbacks,
        )


if __name__ == "__main__":
    main()
