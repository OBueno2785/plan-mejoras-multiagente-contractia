"""Schemas Pydantic para structured output (P1 4.2)."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


# ----- Hallazgos de auditoría -----
class Hallazgo(BaseModel):
    clausula_afectada: str = Field(default="General")
    tipo: str = Field(default="ERROR")
    cita: str = Field(default="")
    explicacion: str = Field(default="")
    severidad: str = Field(default="MEDIA")


class RespuestaJurista(BaseModel):
    hay_inconsistencias: bool = False
    hallazgos: List[Hallazgo] = Field(default_factory=list)


class RespuestaAuditor(BaseModel):
    hay_inconsistencias: bool = False
    hallazgos: List[Hallazgo] = Field(default_factory=list)


class RespuestaCronista(BaseModel):
    hay_procedimientos: bool = False
    hay_errores_logicos: bool = False
    hay_inconsistencia_plazos: bool = False
    hallazgos_procesos: List[Hallazgo] = Field(default_factory=list)


# ----- Seguridad -----
class RespuestaSeguridad(BaseModel):
    es_seguro: bool = True
    evidencia: str = "Ninguna"


# ----- Extracción del grafo -----
class TripletaGrafo(BaseModel):
    origen: str = Field(description="Entidad origen (ej. 'Cláusula 5.1')")
    relacion: str = Field(description="Una de: REFERENCIA_A, ESTABLECE_PLAZO, MODIFICA_A, DEPENDE_DE, OBLIGA_A")
    destino: str = Field(description="Entidad destino")
    contexto: str = Field(default="", description="Una frase corta sobre el motivo")
    tipo_origen: str = Field(default="", description="Una de: Cláusula, Plazo, Rol, Entregable, Penalidad")
    tipo_destino: str = Field(default="", description="Una de: Cláusula, Plazo, Rol, Entregable, Penalidad")


class RespuestaExtraccion(BaseModel):
    tripletas: List[TripletaGrafo] = Field(default_factory=list)
