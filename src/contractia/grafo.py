"""Construcción del grafo de conocimiento (P0 + P1)."""
from __future__ import annotations
import asyncio
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from .config import CFG, RELACIONES_VALIDAS
from .logging_setup import get_logger
from .models import RespuestaExtraccion, TripletaGrafo
from .segmentacion import _extraer_num_anexo, _extraer_num_cap
from .utils import canonicalizar_nodo, extraer_cid_de_string, path_cache_grafo

log = get_logger("grafo")


# --- Cadena de extracción con structured output ---
def _construir_cadena_extraccion(llm):
    prompt_extraccion = PromptTemplate(
        template=(
            "# SISTEMA\n"
            "Motor automatizado de extracción de tripletas para un grafo de conocimiento contractual.\n\n"
            "# TAREA\n"
            "Analiza el <texto_seccion> y devuelve una lista de tripletas (origen, relacion, destino).\n\n"
            "# REGLAS DE EXTRACCIÓN\n"
            "- **ENTIDADES VÁLIDAS:** Cláusula, Plazo, Rol, Entregable, Penalidad. Indica el tipo en `tipo_origen` y `tipo_destino`.\n"
            "- **REGLA DE DESAMBIGUACIÓN (CRÍTICO):** Para cláusulas usa EXACTAMENTE el formato 'Cláusula X.Y ({seccion_contenedora})'. La sección contenedora ya viene dada — NO la inventes ni la infieras del texto.\n"
            "- **PROHIBICIÓN DE EXTERNALIDADES (CRÍTICO):** No extraigas leyes, decretos, códigos civiles ni documentos externos al contrato.\n"
            "- **RELACIONES VÁLIDAS:** REFERENCIA_A, ESTABLECE_PLAZO, MODIFICA_A, DEPENDE_DE, OBLIGA_A. Devuelve EXACTAMENTE una de estas en `relacion`.\n\n"
            "# DATOS DE ENTRADA\n"
            "<seccion_contenedora>\n{seccion_contenedora}\n</seccion_contenedora>\n\n"
            "<texto_seccion>\n{texto}\n</texto_seccion>\n"
        ),
        input_variables=["texto", "seccion_contenedora"],
    )
    return prompt_extraccion | llm.with_structured_output(RespuestaExtraccion)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
async def _ainvocar_estructurado_con_retry(cadena, payload: Dict[str, Any]) -> Any:
    return await cadena.ainvoke(payload)


async def _procesar_seccion_grafo(
    sec: Dict,
    cadena,
    semaforo: asyncio.Semaphore,
    callbacks: Optional[List[Any]] = None,
) -> Tuple[str, str, List[TripletaGrafo]]:
    titulo_seccion = sec.get("titulo", "Sección Desconocida")
    tipo_seccion = sec.get("tipo", "DESCONOCIDO")
    if tipo_seccion == "CAPITULO":
        seccion_contenedora = f"Capitulo {_extraer_num_cap(titulo_seccion)}"
    elif tipo_seccion == "ANEXO":
        seccion_contenedora = f"Anexo {_extraer_num_anexo(titulo_seccion)}"
    else:
        seccion_contenedora = titulo_seccion

    payload = {"texto": sec["contenido"], "seccion_contenedora": seccion_contenedora}
    config = {"callbacks": callbacks or [], "tags": ["grafo"]}

    async with semaforo:
        try:
            cadena_con_config = cadena.with_config(config)
            resp = await _ainvocar_estructurado_con_retry(cadena_con_config, payload)
            tripletas = resp.tripletas if isinstance(resp, RespuestaExtraccion) else []
        except Exception as e:
            log.warning("Error extrayendo grafo en sección %s: %s", titulo_seccion, e)
            tripletas = []
    return titulo_seccion, tipo_seccion, tripletas


async def _construir_grafo_async(
    secciones: List[Dict],
    llm,
    mapa_clausula_a_seccion: Dict[str, Dict],
    callbacks: Optional[List[Any]] = None,
) -> nx.MultiDiGraph:
    cadena = _construir_cadena_extraccion(llm)
    semaforo = asyncio.Semaphore(CFG.max_concurrencia_grafo)

    tareas = [_procesar_seccion_grafo(s, cadena, semaforo, callbacks) for s in secciones]

    G = nx.MultiDiGraph()
    descartadas_relacion = 0
    pbar = tqdm(total=len(tareas), desc="Extrayendo Nodos y Aristas")
    for fut in asyncio.as_completed(tareas):
        titulo_seccion, tipo_seccion, tripletas = await fut
        G.add_node(titulo_seccion, tipo=tipo_seccion)
        for t in tripletas:
            relacion = (t.relacion or "").upper().strip()
            if relacion not in RELACIONES_VALIDAS:
                descartadas_relacion += 1
                continue
            origen = canonicalizar_nodo(t.origen, mapa_clausula_a_seccion)
            destino = canonicalizar_nodo(t.destino, mapa_clausula_a_seccion)
            G.add_edge(origen, destino, relacion=relacion, contexto=t.contexto or "")
            G.add_edge(titulo_seccion, origen, relacion="CONTIENE", contexto="Estructura del documento")
            if t.tipo_origen and "tipo" not in G.nodes[origen]:
                G.nodes[origen]["tipo"] = t.tipo_origen
            if t.tipo_destino and "tipo" not in G.nodes[destino]:
                G.nodes[destino]["tipo"] = t.tipo_destino
        pbar.update(1)
    pbar.close()

    if descartadas_relacion:
        log.warning("%d aristas descartadas por relación fuera de schema.", descartadas_relacion)
    return G


def construir_grafo_conocimiento(
    secciones: List[Dict],
    llm,
    mapa_clausula_a_seccion: Dict[str, Dict],
    callbacks: Optional[List[Any]] = None,
    cache_dir: Optional[str] = None,
    usar_cache: Optional[bool] = None,
) -> nx.MultiDiGraph:
    cache_dir = cache_dir or CFG.cache_dir
    if usar_cache is None: usar_cache = CFG.usar_cache

    log.info("FASE 1.5: Construyendo Grafo de Conocimiento (GraphRAG Jerárquico)...")
    cache_path = path_cache_grafo(secciones, cache_dir)
    if usar_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                G = pickle.load(f)
            log.info("Grafo cargado desde cache: %s (%d nodos / %d aristas)", cache_path, G.number_of_nodes(), G.number_of_edges())
            return G
        except Exception as e:
            log.warning("Cache de grafo corrupto, reconstruyendo: %s", e)

    G = asyncio.get_event_loop().run_until_complete(
        _construir_grafo_async(secciones, llm, mapa_clausula_a_seccion, callbacks)
    )

    log.info("Grafo construido: %d nodos y %d relaciones.", G.number_of_nodes(), G.number_of_edges())

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        log.info("Grafo persistido en %s", cache_path)
    except Exception as e:
        log.warning("No se pudo persistir el grafo: %s", e)

    return G


def construir_indice_nodos_por_cid(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    indice: Dict[str, List[str]] = {}
    for n in G.nodes():
        cid = extraer_cid_de_string(str(n))
        if cid:
            indice.setdefault(cid, []).append(n)
    return indice


def visualizar_grafo(G: nx.MultiDiGraph) -> None:
    log.info("Generando visualización del grafo en 2D...")
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue", alpha=0.9, edgecolors="black")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", font_weight="bold")
    plt.title("Grafo de Conocimiento del Contrato (GraphRAG Jerárquico)", fontsize=18, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def obtener_contexto_grafo(
    clausulas_locales: List[str],
    G: nx.MultiDiGraph,
    mapa_textos: Dict[str, Dict],
    indice_nodos: Dict[str, List[str]],
    profundidad: int = 2,
) -> str:
    contexto: List[str] = []
    aristas_emitidas: Set[Tuple[str, str, str]] = set()
    textos_emitidos: Set[str] = set()

    G_und = G.to_undirected(as_view=True)

    seeds: Set[str] = set()
    for cid in clausulas_locales:
        for nodo in indice_nodos.get(cid, []):
            if nodo in G:
                seeds.add(nodo)

    nodos_relevantes: Set[str] = set()
    for seed in seeds:
        try:
            ego = nx.ego_graph(G_und, seed, radius=profundidad, undirected=True)
            nodos_relevantes.update(ego.nodes())
        except nx.NodeNotFound:
            continue

    for u in nodos_relevantes:
        if u not in G: continue
        for v in G.successors(u):
            if v not in nodos_relevantes: continue
            edges_dict = G.get_edge_data(u, v) or {}
            for _key, datos_arista in edges_dict.items():
                rel = datos_arista.get("relacion", "CONECTA_CON")
                ctx = datos_arista.get("contexto", "")
                clave = (u, v, rel)
                if clave in aristas_emitidas: continue
                aristas_emitidas.add(clave)
                contexto.append(f"- {u} --[{rel}]--> {v} (Contexto: {ctx})")

                id_ref = extraer_cid_de_string(str(v))
                if id_ref and id_ref in mapa_textos and id_ref not in textos_emitidos:
                    textos_emitidos.add(id_ref)
                    contexto.append(f"  [TEXTO RECUPERADO DE {v}]:\n{mapa_textos[id_ref]['texto']}\n")

    return "\n".join(contexto) if contexto else "No hay relaciones en el grafo para esta sección."
