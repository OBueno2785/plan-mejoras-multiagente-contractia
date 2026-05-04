"""Renderizado del informe de auditoría (P2 8.1/8.2: dashboard + deduplicación)."""
from __future__ import annotations
from typing import Any, Dict, List

from .segmentacion import _key_sort_clauses


# --- P2 8.2: deduplicación de hallazgos ---
def deduplicar_hallazgos(hallazgos: List[Dict]) -> List[Dict]:
    """Elimina hallazgos duplicados por (tipo, clausula_afectada, primeros 60 chars de explicacion)."""
    vistos = set()
    out = []
    for h in hallazgos:
        clave = (
            h.get("tipo", ""),
            h.get("clausula_afectada", ""),
            h.get("explicacion", "")[:60],
        )
        if clave not in vistos:
            vistos.add(clave)
            out.append(h)
    return out


# --- P2 8.1: dashboard de severidades ---
def dashboard_severidades(resultados: List[Dict]) -> str:
    """Genera tabla Markdown con conteo de hallazgos por nivel de severidad."""
    conteos: Dict[str, int] = {"CRITICA": 0, "ALTA": 0, "MEDIA": 0, "BAJA": 0}
    for r in resultados:
        for h in r.get("hallazgos", []):
            sev = (
                h.get("severidad") if isinstance(h, dict) else getattr(h, "severidad", "MEDIA")
            ) or "MEDIA"
            sev = sev.upper().replace("CRÍTICA", "CRITICA").replace("CRÍTICO", "CRITICA")
            conteos[sev] = conteos.get(sev, 0) + 1
            if sev not in ("CRITICA", "ALTA", "MEDIA", "BAJA"):
                conteos["MEDIA"] += 1

    total = sum(conteos[k] for k in ("CRITICA", "ALTA", "MEDIA", "BAJA"))
    if not total:
        return "_(sin hallazgos)_"

    lineas = ["| Severidad | Cantidad | % |", "|:---|---:|---:|"]
    iconos = {"CRITICA": "🔴", "ALTA": "🟠", "MEDIA": "🟡", "BAJA": "🟢"}
    for sev in ("CRITICA", "ALTA", "MEDIA", "BAJA"):
        n = conteos[sev]
        pct = 100 * n / total if total else 0
        lineas.append(f"| {iconos[sev]} {sev} | {n} | {pct:.1f}% |")
    lineas.append(f"| **TOTAL** | **{total}** | 100% |")
    return "\n".join(lineas)


def render_auditoria_markdown(resultado: Dict[str, Any]) -> str:
    if resultado.get("abortado_por_seguridad"):
        return (
            "# Informe de Auditoría Contractual\n\n"
            "⛔ **AUDITORÍA ABORTADA POR ALERTA DE SEGURIDAD**\n\n"
            f"**Evidencia:** {resultado.get('evidencia', 'No disponible')}\n"
        )

    secciones_idx = resultado.get("indice_secciones", [])
    claus_idx = resultado.get("indice_global_clausulas", [])
    resultados = resultado.get("resultados_auditoria", [])

    # Deduplicar hallazgos en cada sección
    resultados_dedup = []
    for r in resultados:
        hallazgos_dedup = deduplicar_hallazgos(r.get("hallazgos", []))
        resultados_dedup.append({**r, "hallazgos": hallazgos_dedup})

    total_errores = sum(len(r["hallazgos"]) for r in resultados_dedup)

    md = ["# Informe de Auditoría Contractual (Aumentado con GraphRAG)"]

    md.append("## Resumen Estructural")
    md.append(f"- **Secciones Analizadas**: {len(secciones_idx)}")
    md.append(f"- **Cláusulas Definidas**: {len(claus_idx)}")
    md.append(f"- **Total de Inconsistencias Detectadas**: {total_errores}")

    md.append("\n## Dashboard de Severidades")
    md.append(dashboard_severidades(resultados_dedup))

    md.append("\n## Índice Global de Cláusulas (Definiciones)")
    md.append(", ".join(claus_idx) if claus_idx else "_No se detectaron cláusulas._")

    md.append("\n## Hallazgos Detallados")
    if not resultados_dedup:
        md.append("_No se detectaron inconsistencias en el contrato._")
        return "\n\n".join(md)

    for res_sec in resultados_dedup:
        titulo_sec = res_sec["seccion"]
        hallazgos = res_sec["hallazgos"]
        md.append(f"\n### {titulo_sec}")

        mapa_clausulas: Dict[str, list] = {}
        for h in hallazgos:
            c_id = h.get("clausula_afectada", "General") if isinstance(h, dict) else getattr(h, "clausula_afectada", "General")
            mapa_clausulas.setdefault(c_id, []).append(h)

        claves_ordenadas = sorted(
            mapa_clausulas.keys(),
            key=lambda x: _key_sort_clauses(x) if x != "General" else [0],
        )

        for c_id in claves_ordenadas:
            icono = "⚠️" if c_id == "General" else "📌"
            md.append(f"\n#### {icono} Cláusula {c_id}")

            for item in mapa_clausulas[c_id]:
                if isinstance(item, dict):
                    tipo = item.get("tipo", "ERROR")
                    sev = item.get("severidad", "MEDIA")
                    expl = item.get("explicacion", "")
                    cita = item.get("cita", "")
                else:
                    tipo = item.tipo
                    sev = item.severidad
                    expl = item.explicacion
                    cita = item.cita

                md.append(f"- **[{tipo}]** ({sev})")
                md.append(f"  - *Problema:* {expl}")
                if cita:
                    md.append(f"  - *Cita:* \"{cita}\"")

    return "\n\n".join(md)
