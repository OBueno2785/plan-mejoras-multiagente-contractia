"""Chatbot Q&A con GraphRAG real y multi-turn (celda 7 del notebook)."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .logging_setup import get_logger
from .segmentacion import _NUM_PATTERN
from .utils import extraer_cid_de_string

log = get_logger("chat")

_PATRON_CAP_PREG = re.compile(r"cap[ií]tulo[\s]+([IVXLCDM\d]+)", re.IGNORECASE)
_PATRON_ANEXO_PREG = re.compile(r"anexo[\s]+([IVXLCDM\d]+|[A-Z])\b", re.IGNORECASE)


def _seleccionar_nodos_seed(
    pregunta: str,
    G: nx.MultiDiGraph,
    indice_nodos: Dict[str, List[str]],
    indice_secciones: List[Dict],
) -> Set[str]:
    seeds: Set[str] = set()

    for m in re.finditer(_NUM_PATTERN, pregunta):
        cid = m.group(0)
        for n in indice_nodos.get(cid, []):
            if n in G:
                seeds.add(n)

    capitulos_preguntados = {m.group(1).upper() for m in _PATRON_CAP_PREG.finditer(pregunta)}
    anexos_preguntados = {m.group(1).upper() for m in _PATRON_ANEXO_PREG.finditer(pregunta)}
    for s in indice_secciones:
        n_norm = (s.get("n", "") or "").upper()
        if s["tipo"] == "CAPITULO" and n_norm in capitulos_preguntados:
            if s["titulo"] in G:
                seeds.add(s["titulo"])
        elif s["tipo"] == "ANEXO" and n_norm in anexos_preguntados:
            if s["titulo"] in G:
                seeds.add(s["titulo"])

    if not seeds:
        tokens = {t.lower() for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{5,}", pregunta)}
        for n in G.nodes():
            n_low = str(n).lower()
            if any(tok in n_low for tok in tokens):
                seeds.add(n)
    return seeds


def _ego_subgrafo(G: nx.MultiDiGraph, seeds: Set[str], radius: int = 2) -> Set[str]:
    G_und = G.to_undirected(as_view=True)
    nodos: Set[str] = set()
    for seed in seeds:
        try:
            nodos.update(nx.ego_graph(G_und, seed, radius=radius, undirected=True).nodes())
        except nx.NodeNotFound:
            continue
    return nodos


def _resumen_subgrafo(G: nx.MultiDiGraph, nodos: Set[str]) -> str:
    if not nodos:
        return "(sin contexto del grafo)"
    lineas: List[str] = []
    for u in nodos:
        if u not in G:
            continue
        for v in G.successors(u):
            if v not in nodos:
                continue
            for _key, data in (G.get_edge_data(u, v) or {}).items():
                rel = data.get("relacion", "CONECTA_CON")
                ctx = data.get("contexto", "")
                lineas.append(f"[{u}] --({rel})--> [{v}] (Contexto: {ctx})")
    return "\n".join(lineas) if lineas else "(grafo sin aristas relevantes)"


def _recolectar_textos_relevantes(
    nodos_relevantes: Set[str],
    secciones: List[Dict],
    indice_secciones: List[Dict],
) -> str:
    titulos_sec = {s["titulo"] for s in secciones}
    titulos_relevantes = nodos_relevantes & titulos_sec

    for n in nodos_relevantes:
        cid = extraer_cid_de_string(str(n))
        if not cid:
            continue
        for s in secciones:
            if cid in (s.get("contenido", "") or ""):
                titulos_relevantes.add(s["titulo"])
                break

    if not titulos_relevantes:
        return "(sin texto relevante)"
    bloques: List[str] = []
    for s in secciones:
        if s["titulo"] in titulos_relevantes:
            bloques.append(f"\n=== {s['titulo']} ===\n{s['contenido']}")
    return "\n".join(bloques)


def consultar_contrato_graphrag(
    pregunta_usuario: str,
    G: nx.MultiDiGraph,
    secciones: List[Dict],
    indice_secciones: List[Dict],
    indice_nodos_grafo: Dict[str, List[str]],
    llm,
    historial: Optional[List[Tuple[str, str]]] = None,
    callbacks: Optional[List[Any]] = None,
) -> str:
    seeds = _seleccionar_nodos_seed(pregunta_usuario, G, indice_nodos_grafo, indice_secciones)
    nodos_relevantes = _ego_subgrafo(G, seeds, radius=2) if seeds else set()
    contexto_grafo = _resumen_subgrafo(G, nodos_relevantes)
    textos_relevantes = _recolectar_textos_relevantes(nodos_relevantes, secciones, indice_secciones)

    str_indice = "\n".join([f"- {s['tipo']} {s['n']}: {s['titulo']}" for s in indice_secciones])
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    historial_str = ""
    if historial:
        historial_str = "\n".join([f"USER: {p}\nASSISTANT: {r}" for p, r in historial[-3:]])

    prompt_qa = PromptTemplate(
        template=(
            "# SISTEMA\n"
            "Motor de consulta y análisis de contratos.\n\n"
            "# TAREA\n"
            "Responde a la pregunta del usuario basándote estrictamente en los datos provistos (índice, sub-grafo y textos relevantes).\n\n"
            "# REGLAS DE ORO\n"
            "- **CERO EXTERNALIDADES:** Basa tu respuesta ÚNICA Y EXCLUSIVAMENTE en lo provisto. No uses conocimiento legal externo.\n"
            "- **CONCIENCIA TEMPORAL:** Hoy es {fecha_actual}. Si la pregunta es sobre plazos/vigencia, calcula con esta fecha.\n"
            "- Si los textos relevantes no contienen la respuesta, dilo explícitamente.\n\n"
            "# FORMATO DE SALIDA ESPERADO\n"
            "### 🔍 SECCIONES CONSULTADAS\n"
            "- [Enumera los Capítulos o Anexos que revisaste]\n\n"
            "### ⚖️ RESPUESTA\n"
            "[Tu respuesta detallada citando cláusulas concretas.]\n\n"
            "---\n"
            "# DATOS DE ENTRADA\n\n"
            "<historial_reciente>\n{historial}\n</historial_reciente>\n\n"
            "<indice_contrato>\n{indice}\n</indice_contrato>\n\n"
            "<sub_grafo_relevante>\n{grafo}\n</sub_grafo_relevante>\n\n"
            "<textos_relevantes>\n{textos}\n</textos_relevantes>\n\n"
            "<pregunta_usuario>\n{pregunta}\n</pregunta_usuario>\n"
        ),
        input_variables=["indice", "grafo", "textos", "pregunta", "fecha_actual", "historial"],
    )
    cadena = (prompt_qa | llm | StrOutputParser()).with_config(
        {"callbacks": callbacks or [], "tags": ["chat"]}
    )
    return cadena.invoke({
        "indice": str_indice,
        "grafo": contexto_grafo,
        "textos": textos_relevantes,
        "pregunta": pregunta_usuario,
        "fecha_actual": fecha_hoy,
        "historial": historial_str or "(sin historial)",
    })


def iniciar_chat_interactivo(
    G: nx.MultiDiGraph,
    secciones: List[Dict],
    indice_secciones: List[Dict],
    indice_nodos_grafo: Dict[str, List[str]],
    llm,
    callbacks: Optional[List[Any]] = None,
) -> None:
    print("\n" + "=" * 50)
    print("ASISTENTE LEGAL ACTIVADO (GraphRAG real, multi-turn)")
    print("Escribe 'salir' para terminar.")
    print("=" * 50 + "\n")

    historial: List[Tuple[str, str]] = []
    while True:
        pregunta = input("\nTú: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Asistente: ¡Hasta luego!")
            break
        if not pregunta.strip():
            continue

        print("Asistente pensando (GraphRAG selectivo)...")
        try:
            respuesta = consultar_contrato_graphrag(
                pregunta, G, secciones, indice_secciones,
                indice_nodos_grafo, llm,
                historial=historial, callbacks=callbacks,
            )
            print(f"\n{respuesta}")
            historial.append((pregunta, respuesta))
        except Exception as e:
            log.warning("Error al consultar: %s", e)
