"""Tests para informe.py: deduplicar_hallazgos, dashboard_severidades, render_auditoria_markdown."""
import pytest
from contractia.informe import (
    deduplicar_hallazgos,
    dashboard_severidades,
    render_auditoria_markdown,
)


class TestDeduplicarHallazgos:
    def test_elimina_duplicados_exactos(self):
        h = {"tipo": "ERROR", "clausula_afectada": "5.1", "explicacion": "problema X", "cita": "", "severidad": "ALTA"}
        resultado = deduplicar_hallazgos([h, h, h])
        assert len(resultado) == 1

    def test_mantiene_distintos(self):
        hallazgos = [
            {"tipo": "ERROR", "clausula_afectada": "5.1", "explicacion": "problema X", "cita": "", "severidad": "ALTA"},
            {"tipo": "ERROR", "clausula_afectada": "5.2", "explicacion": "problema X", "cita": "", "severidad": "ALTA"},
        ]
        assert len(deduplicar_hallazgos(hallazgos)) == 2

    def test_diferente_tipo_no_deduplica(self):
        hallazgos = [
            {"tipo": "ERROR", "clausula_afectada": "5.1", "explicacion": "problema X", "cita": "", "severidad": "ALTA"},
            {"tipo": "ADVERTENCIA", "clausula_afectada": "5.1", "explicacion": "problema X", "cita": "", "severidad": "MEDIA"},
        ]
        assert len(deduplicar_hallazgos(hallazgos)) == 2

    def test_lista_vacia(self):
        assert deduplicar_hallazgos([]) == []

    def test_preserva_primer_hallazgo(self):
        h1 = {"tipo": "ERROR", "clausula_afectada": "5.1", "explicacion": "X", "cita": "cita1", "severidad": "ALTA"}
        h2 = {"tipo": "ERROR", "clausula_afectada": "5.1", "explicacion": "X", "cita": "cita2", "severidad": "MEDIA"}
        result = deduplicar_hallazgos([h1, h2])
        assert result[0]["cita"] == "cita1"


class TestDashboardSeveridades:
    def test_conteo_correcto(self):
        resultados = [{"hallazgos": [
            {"severidad": "ALTA"},
            {"severidad": "ALTA"},
            {"severidad": "MEDIA"},
            {"severidad": "BAJA"},
        ]}]
        md = dashboard_severidades(resultados)
        assert "ALTA" in md
        assert "2" in md  # dos ALTA

    def test_normaliza_acento(self):
        resultados = [{"hallazgos": [{"severidad": "CRÍTICA"}]}]
        md = dashboard_severidades(resultados)
        assert "CRITICA" in md

    def test_sin_hallazgos(self):
        md = dashboard_severidades([])
        assert "sin hallazgos" in md.lower()

    def test_contiene_total(self):
        resultados = [{"hallazgos": [{"severidad": "ALTA"}, {"severidad": "BAJA"}]}]
        md = dashboard_severidades(resultados)
        assert "TOTAL" in md
        assert "2" in md

    def test_porcentajes_suman_100(self):
        resultados = [{"hallazgos": [
            {"severidad": "ALTA"},
            {"severidad": "MEDIA"},
            {"severidad": "BAJA"},
            {"severidad": "BAJA"},
        ]}]
        md = dashboard_severidades(resultados)
        assert "100%" in md


class TestRenderAuditoriaMarkdown:
    def _resultado_base(self):
        return {
            "secciones": [{"titulo": "Capítulo 1 OBLIGACIONES", "contenido": "texto", "tipo": "CAPITULO"}],
            "indice_secciones": [{"tipo": "CAPITULO", "n": "1", "titulo": "Cap 1"}],
            "indice_global_clausulas": ["1.1", "1.2"],
            "mapa_clausula_a_seccion": {},
            "resultados_auditoria": [],
        }

    def test_sin_hallazgos(self):
        md = render_auditoria_markdown(self._resultado_base())
        assert "No se detectaron inconsistencias" in md

    def test_abortado_seguridad(self):
        resultado = {"abortado_por_seguridad": True, "evidencia": "inyeccion detectada"}
        md = render_auditoria_markdown(resultado)
        assert "ABORTADA" in md
        assert "inyeccion detectada" in md

    def test_incluye_dashboard(self):
        resultado = self._resultado_base()
        resultado["resultados_auditoria"] = [{
            "seccion": "Capítulo 1 OBLIGACIONES",
            "tipo": "CAPITULO",
            "hallazgos": [
                {"clausula_afectada": "1.1", "tipo": "ERROR", "cita": "", "explicacion": "test", "severidad": "ALTA"},
            ],
        }]
        md = render_auditoria_markdown(resultado)
        assert "Dashboard" in md
        assert "ALTA" in md

    def test_deduplica_en_render(self):
        h = {"clausula_afectada": "1.1", "tipo": "ERROR", "cita": "", "explicacion": "duplicado exacto", "severidad": "ALTA"}
        resultado = self._resultado_base()
        resultado["resultados_auditoria"] = [{
            "seccion": "Capítulo 1 OBLIGACIONES",
            "tipo": "CAPITULO",
            "hallazgos": [h, h, h],
        }]
        md = render_auditoria_markdown(resultado)
        # El informe solo debe mostrar una vez "duplicado exacto"
        assert md.count("duplicado exacto") == 1

    def test_indice_clausulas_en_informe(self):
        md = render_auditoria_markdown(self._resultado_base())
        assert "1.1" in md
        assert "1.2" in md
