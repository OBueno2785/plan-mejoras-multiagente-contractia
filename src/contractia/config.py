"""Configuración centralizada (P2 5.1).

Lee env vars con fallbacks razonables. Los valores hardcoded del notebook
original viven aquí.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # Vertex AI
    project_id: str = field(default_factory=lambda: os.getenv("CONTRACTIA_PROJECT_ID", "agenteia-471917"))
    location: str = field(default_factory=lambda: os.getenv("CONTRACTIA_LOCATION", "us-central1"))
    google_credentials: str | None = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    # Modelos
    modelo_principal: str = field(default_factory=lambda: os.getenv("CONTRACTIA_MODEL", "gemini-2.5-pro"))
    modelo_fallback: str = field(default_factory=lambda: os.getenv("CONTRACTIA_MODEL_FALLBACK", "gemini-2.5-flash"))
    temperature: float = field(default_factory=lambda: float(os.getenv("CONTRACTIA_TEMPERATURE", "0.0")))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("CONTRACTIA_MAX_OUTPUT_TOKENS", "8192")))
    timeout_segundos: int = field(default_factory=lambda: int(os.getenv("CONTRACTIA_TIMEOUT", "600")))

    # Concurrencia
    max_concurrencia_grafo: int = field(default_factory=lambda: int(os.getenv("CONTRACTIA_CONC_GRAFO", "5")))
    max_concurrencia_auditoria: int = field(default_factory=lambda: int(os.getenv("CONTRACTIA_CONC_AUDITORIA", "3")))

    # Paths
    ruta_contrato: str = field(default_factory=lambda: os.getenv("CONTRACTIA_RUTA_CONTRATO", "contrato_nuevo"))
    reporte_md: str = field(default_factory=lambda: os.getenv("CONTRACTIA_REPORTE", "informe_auditoria_contrato.md"))
    cache_dir: str = field(default_factory=lambda: os.getenv("CONTRACTIA_CACHE_DIR", "cache"))
    log_file: str = field(default_factory=lambda: os.getenv("CONTRACTIA_LOG_FILE", "contractia.log"))

    # Feature flags
    enable_llm: bool = field(default_factory=lambda: os.getenv("CONTRACTIA_ENABLE_LLM", "1") == "1")
    usar_cache: bool = field(default_factory=lambda: os.getenv("CONTRACTIA_USAR_CACHE", "1") == "1")


CFG = Config()


# Constantes de schema del grafo (no son configurables — son parte del contrato del modelo)
RELACIONES_VALIDAS: frozenset[str] = frozenset({
    "REFERENCIA_A", "ESTABLECE_PLAZO", "MODIFICA_A", "DEPENDE_DE", "OBLIGA_A"
})
TIPOS_NODO_VALIDOS: frozenset[str] = frozenset({
    "Cláusula", "Plazo", "Rol", "Entregable", "Penalidad"
})
