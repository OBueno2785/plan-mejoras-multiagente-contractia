"""Métricas: tokens consumidos (P1 6.2) y calidad del grafo (P2 2.6.1)."""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain_core.callbacks import BaseCallbackHandler

from .utils import extraer_cid_de_string


# --- P1 6.2: TokenCounterCallback ---
class TokenCounterCallback(BaseCallbackHandler):
    """Acumula tokens (input/output/total) por etiqueta (agente, fase)."""

    def __init__(self):
        self.por_etiqueta: Dict[str, Dict[str, int]] = {}

    def _bucket(self, etiqueta: str) -> Dict[str, int]:
        return self.por_etiqueta.setdefault(
            etiqueta, {"input": 0, "output": 0, "total": 0, "calls": 0}
        )

    def on_llm_end(self, response, *, tags=None, **kwargs):
        etiqueta = (tags[0] if tags else "general")
        bucket = self._bucket(etiqueta)
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None: continue
                um = getattr(msg, "usage_metadata", None) or {}
                bucket["input"] += um.get("input_tokens", 0)
                bucket["output"] += um.get("output_tokens", 0)
                bucket["total"] += um.get("total_tokens", 0)
                bucket["calls"] += 1

    def resumen(self) -> str:
        if not self.por_etiqueta: return "(sin uso registrado)"
        lineas = ["| Etiqueta | Calls | Input | Output | Total |", "|---|---:|---:|---:|---:|"]
        gran_total = {"input": 0, "output": 0, "total": 0, "calls": 0}
        for et in sorted(self.por_etiqueta):
            b = self.por_etiqueta[et]
            lineas.append(f"| {et} | {b['calls']} | {b['input']:,} | {b['output']:,} | {b['total']:,} |")
            for k in gran_total: gran_total[k] += b[k]
        lineas.append(
            f"| **TOTAL** | **{gran_total['calls']}** | **{gran_total['input']:,}** | "
            f"**{gran_total['output']:,}** | **{gran_total['total']:,}** |"
        )
        return "\n".join(lineas)


# --- P2 2.6.1: métricas de calidad del grafo ---
def calcular_metricas_grafo(
    G: nx.MultiDiGraph,
    mapa_clausula_a_seccion: Dict[str, Dict],
) -> Dict[str, Any]:
    """Devuelve un dict con cobertura, fragmentación, distribución y top-degree.

    - cobertura: % de cláusulas en `mapa_clausula_a_seccion` que aparecen como nodo.
    - inventadas: nodos cláusula con cid que NO está en el mapa.
    - fragmentacion: para cada cid, cuántas variantes (nodos) lo contienen.
    - distribucion_relaciones: count por tipo de relación.
    - distribucion_tipos_nodo: count por atributo `tipo` de nodo.
    - componentes_debiles: nº de componentes débilmente conexos.
    - top_degree: top 10 nodos por degree total.
    """
    cids_en_mapa = set(mapa_clausula_a_seccion.keys())

    # Mapa cid -> [nodos]
    cid_a_nodos: Dict[str, List[str]] = {}
    nodos_clausula: List[str] = []
    for n in G.nodes():
        cid = extraer_cid_de_string(str(n))
        if cid:
            cid_a_nodos.setdefault(cid, []).append(n)
            nodos_clausula.append(n)

    cids_en_grafo = set(cid_a_nodos.keys())
    cobertura_pct = (len(cids_en_grafo & cids_en_mapa) / max(1, len(cids_en_mapa))) * 100
    inventadas = sorted(cids_en_grafo - cids_en_mapa)

    fragmentacion = {cid: len(nodos) for cid, nodos in cid_a_nodos.items() if len(nodos) > 1}
    fragmentacion_top = sorted(fragmentacion.items(), key=lambda x: -x[1])[:10]

    distribucion_relaciones = Counter(
        data.get("relacion", "?") for _, _, data in G.edges(data=True)
    )

    distribucion_tipos_nodo = Counter(
        G.nodes[n].get("tipo", "(sin tipo)") for n in G.nodes()
    )

    componentes_debiles = nx.number_weakly_connected_components(G)

    grados = sorted(G.degree(), key=lambda x: -x[1])[:10]

    return {
        "n_nodos": G.number_of_nodes(),
        "n_aristas": G.number_of_edges(),
        "cobertura_pct": round(cobertura_pct, 1),
        "n_clausulas_mapa": len(cids_en_mapa),
        "n_clausulas_grafo": len(cids_en_grafo),
        "n_inventadas": len(inventadas),
        "inventadas": inventadas[:20],
        "n_fragmentadas": len(fragmentacion),
        "fragmentacion_top": fragmentacion_top,
        "distribucion_relaciones": dict(distribucion_relaciones),
        "distribucion_tipos_nodo": dict(distribucion_tipos_nodo),
        "componentes_debiles": componentes_debiles,
        "top_degree": grados,
    }


def render_metricas_grafo_md(m: Dict[str, Any]) -> str:
    """Genera un fragmento Markdown con la salud del grafo."""
    lineas = ["## Calidad del Grafo de Conocimiento", ""]
    lineas.append(f"- **Nodos / Aristas**: {m['n_nodos']:,} / {m['n_aristas']:,}")
    lineas.append(
        f"- **Cobertura**: {m['cobertura_pct']}% "
        f"({m['n_clausulas_grafo']} de {m['n_clausulas_mapa']} cláusulas en grafo)"
    )
    lineas.append(f"- **Cláusulas inventadas por el LLM**: {m['n_inventadas']}")
    if m["inventadas"]:
        lineas.append("  - Ejemplos: " + ", ".join(m["inventadas"][:10]))
    lineas.append(f"- **Cláusulas fragmentadas (varios nodos por cid)**: {m['n_fragmentadas']}")
    if m["fragmentacion_top"]:
        ejemplos = ", ".join(f"{cid}×{n}" for cid, n in m["fragmentacion_top"][:5])
        lineas.append(f"  - Top: {ejemplos}")
    lineas.append(f"- **Componentes débilmente conexos**: {m['componentes_debiles']}")

    if m["distribucion_relaciones"]:
        lineas.append("\n### Distribución de relaciones")
        lineas.append("| Relación | Aristas |")
        lineas.append("|---|---:|")
        for rel, n in sorted(m["distribucion_relaciones"].items(), key=lambda x: -x[1]):
            lineas.append(f"| {rel} | {n:,} |")

    if m["distribucion_tipos_nodo"]:
        lineas.append("\n### Distribución de tipos de nodo")
        lineas.append("| Tipo | Nodos |")
        lineas.append("|---|---:|")
        for tipo, n in sorted(m["distribucion_tipos_nodo"].items(), key=lambda x: -x[1]):
            lineas.append(f"| {tipo} | {n:,} |")

    if m["top_degree"]:
        lineas.append("\n### Top nodos por degree")
        for n, d in m["top_degree"]:
            lineas.append(f"- `{n}` (degree {d})")

    return "\n".join(lineas)
