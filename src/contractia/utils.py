"""Helpers transversales (parseo de JSON, hashing, canonicalización)."""
from __future__ import annotations
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from .segmentacion import _NUM_PATTERN, _extraer_num_cap, _extraer_num_anexo


def parse_json_seguro(texto_llm: str) -> Any:
    """Fallback: parsea JSON cuando structured output no está disponible."""
    if not texto_llm: return {}
    texto = texto_llm.strip()
    if "```" in texto:
        match = re.search(r"```(?:json)?(.*?)```", texto, re.DOTALL | re.IGNORECASE)
        if match: texto = match.group(1).strip()
    if "sin inconsistencias" in texto.lower() or "no se encontraron errores" in texto.lower():
        return {}
    texto = re.sub(r"//.*", "", texto)
    texto = re.sub(r",\s*([\]}])", r"\1", texto)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        try:
            match = re.search(r"(\{.*\}|\[.*\])", texto, re.DOTALL)
            if match: return json.loads(match.group(0))
        except Exception:
            pass
        return {}


def extraer_cid_de_string(s: str) -> Optional[str]:
    """Extrae el primer ID con formato N.N(.N)... (reusa _NUM_PATTERN)."""
    m = re.search(_NUM_PATTERN, s)
    return m.group(0) if m else None


def canonicalizar_nodo(raw: str, mapa_clausula_a_seccion: Dict[str, Dict]) -> str:
    """String libre del LLM → identificador canónico estable usando el mapa."""
    if not isinstance(raw, str): return str(raw)
    raw_clean = " ".join(raw.split()).strip()
    if not raw_clean: return raw_clean

    cid = extraer_cid_de_string(raw_clean)
    if cid and cid in mapa_clausula_a_seccion:
        info = mapa_clausula_a_seccion[cid]
        tipo_sec = info.get("tipo", "?")
        titulo_sec = info.get("seccion", "")
        if tipo_sec == "CAPITULO":
            num_sec = _extraer_num_cap(titulo_sec)
            return f"Cláusula {cid} (Capitulo {num_sec})"
        else:
            num_sec = _extraer_num_anexo(titulo_sec)
            return f"Cláusula {cid} (Anexo {num_sec})"
    return raw_clean


def hash_secciones(secciones: List[Dict]) -> str:
    payload = json.dumps(
        [(s.get("titulo", ""), s.get("contenido", "")) for s in secciones],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def path_cache_grafo(secciones: List[Dict], cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"grafo_{hash_secciones(secciones)}.pkl")


def path_checkpoint_hallazgos(secciones: List[Dict], cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"hallazgos_{hash_secciones(secciones)}.jsonl")
