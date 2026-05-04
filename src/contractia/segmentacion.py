"""Segmentación e índices (celda 3 del notebook original).

ESTE MÓDULO NO DEBE MODIFICARSE: contiene los regex de segmentación
que dependen del formato exacto de los contratos en español. Cualquier
cambio aquí puede romper la detección de capítulos, anexos y cláusulas.
"""
from __future__ import annotations
import re
from typing import List, Dict, Tuple, Set

import logging
_log = logging.getLogger("contractia.segmentacion")


# --- helpers de normalización ---
def _norm_text(s: str) -> str:
    s = s.replace("﻿", "").replace("\r", "")
    s = s.replace(" ", " ")
    s = s.replace("­", "")
    s = re.sub(r"\f", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_toc_trailers(header: str) -> str:
    h = re.sub(r"[ \.\·•]{2,}\s*\d+\s*$", "", header)
    h = re.sub(r"\s{2,}", " ", h).strip()
    return h


def _clean_header(header: str) -> str:
    return _strip_toc_trailers(" ".join(header.split()).strip())


def _is_toc_like(raw_line: str) -> bool:
    line = raw_line.rstrip()
    if re.search(r"[ \.\·•]{3,}\s*\d+\s*$", line): return True
    if line.count(".") >= 6: return True
    if re.search(r"\s\d{1,4}\s*$", line) and not re.search(r"[a-záéíóúñ]", line): return True
    return False


_UPPER_TOKEN = r"[A-ZÁÉÍÓÚÜÑ0-9]"
_UPPER_SPAN  = rf"{_UPPER_TOKEN}[_A-ZÁÉÍÓÚÜÑ0-9 ,\-/()º°\.]*"


def _truncate_upper_block(title: str) -> str:
    t = _strip_toc_trailers(title)
    m = re.search(r"(Cap[ií]tulo\s+(?:[IVXLCDM]+|\d+))\s+(" + _UPPER_SPAN + r")", t, flags=re.IGNORECASE)
    if m:
        base = m.group(1)
        up   = m.group(2)
        return f"{base} {up}".strip()
    return _clean_header(t)


_UPPER_WORD = re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9][A-ZÁÉÍÓÚÜÑ0-9/()º°\-.,]+$")


def _is_proper_caps_title(title: str, min_words: int = 2) -> bool:
    t = title.strip()
    if re.search(r"[a-záéíóúñ]", t): return False
    words = [w for w in re.split(r"[ \t,;/\-]+", t) if w]
    cap_words = [w for w in words if _UPPER_WORD.match(w)]
    if len(cap_words) >= min_words: return True
    if len(cap_words) == 1 and len(cap_words[0]) >= 5: return True
    return False


_CAP_RX = re.compile(
    rf"^[ \t]*Cap[ií]tulo[ \t]+(?P<num>(?:[IVXLCDM]+|\d+))[ \t]+(?P<title>{_UPPER_SPAN})(?=\s+(?:[a-záéíóúñ]|del\b|de\b|la\b)|\s*$)",
    re.IGNORECASE | re.MULTILINE
)

_ANEXO_RX = re.compile(
    r"^[ \t]*Anexo(?:s)?[ \t]+(?P<num>([IVXLCDM]+|\d+|[A-Z]))[ \t]+(?P<title>[A-ZÁÉÍÓÚÜÑ0-9][A-ZÁÉÍÓÚÜÑ0-9 ,\-\./()º°]+)[ \t]*$",
    re.IGNORECASE | re.MULTILINE
)

_NUM_PATTERN = r"\d+(?:\.\d+)+(?:\.[a-zA-Z])?"

_CLAUSE_PATTERNS = [
    rf"\b(?:CL[AÁ]USULA|ART[IÍ]CULO|SECCI[ÓO]N)\s+(?:N[°º]\s*)?({_NUM_PATTERN})\b",
    rf"(?<!S/\.)(?<!US\$\.)(?<!\$)\b({_NUM_PATTERN})\s*[.)\-]?\s+(?=[A-ZÁÉÍÓÚÜÑ])"
]

_CLAUSE_RX = re.compile("|".join(f"(?:{p})" for p in _CLAUSE_PATTERNS), re.IGNORECASE | re.UNICODE | re.MULTILINE)

_CLAUSE_LIST_RX = re.compile(
    rf"\bCL[AÁ]USULAS?\b[ \t]+(?:N[°º]\s*)?({_NUM_PATTERN}"
    rf"(?:[ \t]*(?:,|;|/|y|e)[ \t]*{_NUM_PATTERN})+)",
    re.IGNORECASE | re.UNICODE
)

_RANGE_RX = re.compile(r"(\d+(?:\.\d+)+)\s*(?:a|-|–|—)\s*(\d+(?:\.\d+)+)")


def _extraer_num_cap(titulo: str) -> str:
    m = re.search(r"Cap[ií]tulo[ \t]+([IVXLCDM]+|\d+)\b", titulo, re.IGNORECASE)
    return m.group(1) if m else "?"


def _extraer_num_anexo(titulo: str) -> str:
    m = re.search(r"\bAnexo[ \t]+([IVXLCDM]+|\d+|[A-Z])\b", titulo, re.IGNORECASE)
    return m.group(1) if m else "?"


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    if not s: return 0
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    if not all(c in rom_val for c in s if c.isalpha()): return 0
    if s.isdigit(): return 0
    int_val = 0
    try:
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i-1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i-1]]
            else:
                int_val += rom_val[s[i]]
    except Exception: return 0
    return int_val


def _extraer_numeros_clausula(texto_capitulo: str, prefijo_capitulo: str) -> List[str]:
    ids_encontrados = set()
    for m in _CLAUSE_RX.finditer(texto_capitulo):
        gid = m.group(1) or m.group(2)
        if gid and gid.startswith(prefijo_capitulo + "."):
             ids_encontrados.add(gid.strip().rstrip('.'))
    return list(ids_encontrados)


def _validar_secuencia_clausulas(numeros_encontrados: List[str], prefijo: str) -> Tuple[bool, List[str], int]:
    indices_primer_nivel = set()
    try:
        for num in numeros_encontrados:
            partes = num.split('.')
            if len(partes) >= 2 and partes[0] == prefijo:
                if partes[1].isdigit():
                    indices_primer_nivel.add(int(partes[1]))
    except ValueError: return False, ["Formato numérico inválido"], 0

    if not indices_primer_nivel: return True, [], 0
    max_indice = max(indices_primer_nivel)
    n_clausulas_primer_nivel = len(indices_primer_nivel)
    conjunto_esperado = set(range(1, max_indice + 1))
    if indices_primer_nivel == conjunto_esperado:
        return True, [], n_clausulas_primer_nivel
    else:
        faltantes_num = sorted(list(conjunto_esperado - indices_primer_nivel))
        faltantes_str = [f'{prefijo}.{i}' for i in faltantes_num]
        return False, faltantes_str, n_clausulas_primer_nivel


def _find_sections(text: str) -> List[Tuple[int,int,str,str]]:
    t = _norm_text(text)
    L = len(t)
    raw_hits: List[Tuple[int,str,str]] = []
    for m in _CAP_RX.finditer(t):
        header = _truncate_upper_block(m.group(0).strip())
        raw_hits.append((m.start(), header, "CAPITULO"))
    for m in _ANEXO_RX.finditer(t):
        header = m.group(0).strip()
        raw_hits.append((m.start(), header, "ANEXO"))
    if not raw_hits: return [(0, L, "DOCUMENTO COMPLETO", "CAPITULO")]
    kept = []
    for pos, raw, kind in raw_hits:
        if _is_toc_like(raw): continue
        clean = _clean_header(raw)
        if kind == "CAPITULO":
            m = re.search(r"Cap[ií]tulo[ \t]+(?:[IVXLCDM]+|\d+)[ \t]+(.+)$", clean, re.IGNORECASE)
            title_part = (m.group(1).strip() if m else "")
        else:
            m = re.search(r"Anexo(?:s)?[ \t]+(?:[IVXLCDM]+|\d+|[A-Z])[ \t]+(.+)$", clean, re.IGNORECASE)
            title_part = (m.group(1).strip() if m else "")
        if not _is_proper_caps_title(title_part, min_words=2): continue
        kept.append((pos, clean, kind))
    if not kept:
        kept = [(p, _clean_header(h), k) for p, h, k in raw_hits if not _is_toc_like(h)]
    kept.sort(key=lambda x: x[0])
    spans_tmp = []
    for i, (s, h, k) in enumerate(kept):
        e = kept[i+1][0] if i+1 < len(kept) else L
        spans_tmp.append((s, e, h, k))
    def _title_quality(kind: str, header: str) -> int:
        if kind == "CAPITULO":
            m = re.search(r"Cap[ií]tulo[ \t]+(?:[IVXLCDM]+|\d+)[ \t]+(.+)$", header, re.IGNORECASE)
        else:
            m = re.search(r"Anexo(?:s)?[ \t]+(?:[IVXLCDM]+|\d+|[A-Z])[ \t]+(.+)$", header, re.IGNORECASE)
        title_part = (m.group(1).strip() if m else "")
        if _is_proper_caps_title(title_part, min_words=2): return 1000 + min(len(title_part), 120)
        if re.search(r"[a-záéíóúñ]", title_part): return -500
        return 0
    best_by_key = {}
    for (s, e, h, k) in spans_tmp:
        ident = _extraer_num_cap(h) if k == "CAPITULO" else _extraer_num_anexo(h)
        key = (k, ident)
        span_len = e - s
        body_bonus = int(0.15 * L) if s > 0.05 * L else 0
        qual = span_len + body_bonus + _title_quality(k, h)
        cur = best_by_key.get(key)
        if cur is None or qual > cur["quality"]:
            best_by_key[key] = {"start": s, "end": e, "header": h, "kind": k, "quality": qual}
    chosen = sorted([(v["start"], v["end"], v["header"], v["kind"]) for v in best_by_key.values()], key=lambda x: x[0])
    spans: List[Tuple[int,int,str,str]] = []
    for i, (s, _, h, k) in enumerate(chosen):
        e = chosen[i+1][0] if i+1 < len(chosen) else L
        spans.append((s, e, h, k))
    return spans


def _post_secciones(text: str, spans: List[Tuple[int,int,str,str]]) -> List[Dict]:
    out: List[Dict] = []
    for (s, e, h, k) in spans:
        content = text[s:e].strip()
        h = _clean_header(h)
        if not content.upper().startswith(h.upper()[:50]):
            content = f"{h}\n{content}"
        out.append({"tipo": k, "titulo": h, "contenido": content})
    return out


def separar_en_secciones(texto_contrato: str) -> List[Dict]:
    t = _norm_text(texto_contrato)
    spans = _find_sections(t)
    secciones = _post_secciones(t, spans)
    caps = [s["titulo"] for s in secciones if s["tipo"] == "CAPITULO"]
    anxs = [s["titulo"] for s in secciones if s["tipo"] == "ANEXO"]

    _log.info("FASE 0: Análisis Estructural — %d capítulos, %d anexos (%d secciones).", len(caps), len(anxs), len(secciones))
    _log.info("FASE 0.5: Auditoría de Secuencia de Cláusulas...")
    for s in secciones:
        tipo_seccion = s.get("tipo", "?")
        if tipo_seccion != "CAPITULO": continue
        titulo_seccion = s.get("titulo", "N/A")
        contenido_seccion = s.get("contenido", "")
        num_raw = _extraer_num_cap(titulo_seccion)
        num_int = 0
        prefijo_seccion = ""
        if num_raw.isdigit():
            try:
                num_int = int(num_raw)
                prefijo_seccion = num_raw
            except ValueError: pass
        elif num_raw != '?':
            num_int = _roman_to_int(num_raw)
            if num_int > 0: prefijo_seccion = str(num_int)
        if num_int == 0 or not prefijo_seccion: continue
        clausulas_encontradas_total = _extraer_numeros_clausula(contenido_seccion, prefijo_seccion)
        if not clausulas_encontradas_total: continue
        es_valido, faltantes, n_primer_nivel = _validar_secuencia_clausulas(clausulas_encontradas_total, prefijo_seccion)
        if es_valido:
            _log.info("  OK: %s (%d cláusulas de primer nivel, secuencia válida).", titulo_seccion, n_primer_nivel)
        else:
            _log.warning("  ERROR SECUENCIA: %s. Faltan: %s", titulo_seccion, faltantes)

    return secciones


def crear_indice_capitulos_anexos(secciones: List[Dict]) -> List[Dict]:
    out = []
    for s in secciones:
        if s["tipo"] == "CAPITULO":
            n = _extraer_num_cap(s["titulo"])
            out.append({"tipo":"CAPITULO","n":n, "titulo": _clean_header(s["titulo"])})
        elif s["tipo"] == "ANEXO":
            n = _extraer_num_anexo(s["titulo"])
            out.append({"tipo":"ANEXO","n":n, "titulo": _clean_header(s["titulo"])})
    return out


def _expand_clause_ranges(text: str) -> Set[str]:
    found: Set[str] = set()
    for a, b in _RANGE_RX.findall(text):
        a_parts = a.split("."); b_parts = b.split(".")
        if len(a_parts) == len(b_parts) and a_parts[:-1] == b_parts[:-1]:
            try:
                start = int(a_parts[-1]); end = int(b_parts[-1])
                if start <= end:
                    base = ".".join(a_parts[:-1])
                    for k in range(start, end+1): found.add(f"{base}.{k}" if base else str(k))
            except Exception: pass
    return found


def _key_sort_clauses(v: str) -> List[int]:
    parts = v.split(".")
    out = []
    for p in parts:
        if p.isdigit(): out.append(int(p))
        else:
            val = ord(p.lower()) if len(p) == 1 and p.isalpha() else 999999
            out.append(val)
    return out


def _clause_ids_in_text(texto: str) -> Set[str]:
    ids: Set[str] = set()
    t = _norm_text(texto)
    ids |= _expand_clause_ranges(t)
    for m in _CLAUSE_LIST_RX.finditer(t):
        bloque = m.group(0)
        found_in_list = re.findall(_NUM_PATTERN, bloque)
        for cid in found_in_list: ids.add(cid.strip())
    for m in _CLAUSE_RX.finditer(t):
        gid = m.group(1) or m.group(2)
        if gid:
            clean_id = gid.strip().rstrip('.')
            ids.add(clean_id)
    return ids


def _get_all_section_numbers_as_str(secciones: List[Dict]) -> Set[str]:
    indices = crear_indice_capitulos_anexos(secciones)
    all_ids_raw = {s['n'] for s in indices if s['n'] != '?'}
    all_nums_int_str = set()
    for r in all_ids_raw:
        if r.isdigit(): all_nums_int_str.add(r)
        else:
            num_int = _roman_to_int(r)
            if num_int > 0: all_nums_int_str.add(str(num_int))
    return all_nums_int_str


def crear_indice_de_clausulas_por_seccion(texto_seccion: str) -> List[str]:
    ids = list(_clause_ids_in_text(texto_seccion))
    ids.sort(key=_key_sort_clauses)
    return ids


def construir_mapa_clausula_a_seccion(secciones: List[Dict]) -> Dict[str, Dict]:
    mapa: Dict[str, Dict] = {}
    for s in secciones:
        tipo_seccion = s.get("tipo", "?")
        titulo_seccion = s.get("titulo", "N/A")
        contenido_seccion = s.get("contenido", "")

        num_raw = ""
        if tipo_seccion == "CAPITULO": num_raw = _extraer_num_cap(titulo_seccion)
        elif tipo_seccion == "ANEXO": num_raw = _extraer_num_anexo(titulo_seccion)
        else: continue

        prefijo_seccion = ""
        if num_raw.isdigit():
            try:
                num_int = int(num_raw)
                prefijo_seccion = num_raw
            except ValueError: pass
        elif num_raw != '?' and tipo_seccion == "CAPITULO":
            num_int = _roman_to_int(num_raw)
            if num_int > 0: prefijo_seccion = str(num_int)

        if not prefijo_seccion: continue

        ids = _extraer_numeros_clausula(contenido_seccion, prefijo_seccion)
        ids.sort(key=_key_sort_clauses)

        # Segmentación por Diccionario Exacto
        posiciones = []
        for cid in ids:
            pattern = rf'(?:^|\n)\s*(?:CL[AÁ]USULA\s+|ART[IÍ]CULO\s+|SECCI[ÓO]N\s+)?(?:N[°º]\s*)?{re.escape(cid)}\b'
            match = re.search(pattern, contenido_seccion, re.IGNORECASE)
            if match:
                posiciones.append((cid, match.start()))
            else:
                match_fallback = re.search(rf'\b{re.escape(cid)}\b', contenido_seccion)
                if match_fallback:
                    posiciones.append((cid, match_fallback.start()))

        posiciones.sort(key=lambda x: x[1])

        for i, (cid, start_pos) in enumerate(posiciones):
            if i + 1 < len(posiciones):
                end_pos = posiciones[i+1][1]
                texto_exacto = contenido_seccion[start_pos:end_pos].strip()
            else:
                texto_exacto = contenido_seccion[start_pos:].strip()

            # Prioridad absoluta al CAPITULO sobre el ANEXO
            if cid not in mapa:
                mapa[cid] = {"tipo": tipo_seccion, "seccion": titulo_seccion, "texto": texto_exacto}
            elif tipo_seccion == "CAPITULO" and mapa[cid]["tipo"] == "ANEXO":
                mapa[cid] = {"tipo": tipo_seccion, "seccion": titulo_seccion, "texto": texto_exacto}

    return mapa


def crear_indice_global_clausulas(secciones: List[Dict]) -> List[str]:
    mapa_definiciones = construir_mapa_clausula_a_seccion(secciones)
    defined_clauses_set = set(mapa_definiciones.keys())
    section_nums_str = _get_all_section_numbers_as_str(secciones)
    filtered_seen = {cid for cid in defined_clauses_set if cid not in section_nums_str}
    return sorted(filtered_seen, key=_key_sort_clauses)
