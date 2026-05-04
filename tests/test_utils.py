"""Tests para utils.py: parse_json_seguro, hash_secciones, canonicalizar_nodo, extraer_cid."""
import pytest
from contractia.utils import (
    parse_json_seguro,
    extraer_cid_de_string,
    canonicalizar_nodo,
    hash_secciones,
    path_cache_grafo,
    path_checkpoint_hallazgos,
)


class TestParseJsonSeguro:
    def test_json_valido(self):
        assert parse_json_seguro('{"key": "value"}') == {"key": "value"}

    def test_lista_valida(self):
        assert parse_json_seguro('[1, 2, 3]') == [1, 2, 3]

    def test_bloque_code_json(self):
        texto = '```json\n{"key": "value"}\n```'
        assert parse_json_seguro(texto) == {"key": "value"}

    def test_bloque_code_sin_lang(self):
        texto = '```\n{"key": "value"}\n```'
        assert parse_json_seguro(texto) == {"key": "value"}

    def test_sin_inconsistencias(self):
        assert parse_json_seguro("sin inconsistencias encontradas") == {}

    def test_vacio(self):
        assert parse_json_seguro("") == {}

    def test_json_invalido(self):
        assert parse_json_seguro("esto no es json @#$%") == {}

    def test_json_con_comentarios(self):
        texto = '{"key": "value" // comentario\n}'
        result = parse_json_seguro(texto)
        assert result == {"key": "value"}

    def test_json_con_trailing_comma(self):
        texto = '{"key": "value",}'
        result = parse_json_seguro(texto)
        assert result == {"key": "value"}


class TestExtraerCidDeString:
    def test_patron_simple(self):
        assert extraer_cid_de_string("Cláusula 5.1 del contrato") == "5.1"

    def test_patron_anidado(self):
        assert extraer_cid_de_string("ver 3.2.1 para detalles") == "3.2.1"

    def test_sin_patron(self):
        assert extraer_cid_de_string("texto sin cid") is None

    def test_string_vacio(self):
        assert extraer_cid_de_string("") is None

    def test_solo_numero(self):
        # Un número aislado (sin punto) NO es un CID válido
        assert extraer_cid_de_string("ver artículo 5") is None


class TestCanonicalizarNodo:
    def test_clausula_en_mapa_capitulo(self):
        mapa = {"5.1": {"tipo": "CAPITULO", "seccion": "Capítulo 5 TITULO", "texto": "..."}}
        result = canonicalizar_nodo("Cláusula 5.1", mapa)
        assert "5.1" in result
        assert "Capitulo" in result

    def test_clausula_en_mapa_anexo(self):
        # CIDs con letra (A.1) no encajan con _NUM_PATTERN (requiere dígito inicial),
        # por lo que canonicalizar_nodo devuelve el string limpio tal cual.
        mapa = {"1.1": {"tipo": "ANEXO", "seccion": "Anexo 1 ESPECIFICACIONES", "texto": "..."}}
        result = canonicalizar_nodo("Cláusula 1.1", mapa)
        assert "1.1" in result
        assert "Anexo" in result

    def test_sin_mapa(self):
        result = canonicalizar_nodo("texto libre sin clausula", {})
        assert result == "texto libre sin clausula"

    def test_cid_no_en_mapa(self):
        mapa = {"5.1": {"tipo": "CAPITULO", "seccion": "Cap 5", "texto": ""}}
        result = canonicalizar_nodo("Cláusula 9.9", mapa)
        assert result == "Cláusula 9.9"

    def test_normaliza_espacios(self):
        result = canonicalizar_nodo("  texto   con   espacios  ", {})
        assert result == "texto con espacios"

    def test_no_string(self):
        result = canonicalizar_nodo(42, {})
        assert result == "42"


class TestHashSecciones:
    def test_determinista(self):
        secciones = [{"titulo": "Cap 1", "contenido": "texto uno"}]
        assert hash_secciones(secciones) == hash_secciones(secciones)

    def test_longitud_16(self):
        secciones = [{"titulo": "Cap 1", "contenido": "texto"}]
        assert len(hash_secciones(secciones)) == 16

    def test_distinto_con_distinto_contenido(self):
        s1 = [{"titulo": "Cap 1", "contenido": "texto uno"}]
        s2 = [{"titulo": "Cap 1", "contenido": "texto dos"}]
        assert hash_secciones(s1) != hash_secciones(s2)

    def test_distinto_con_distinto_titulo(self):
        s1 = [{"titulo": "Cap 1", "contenido": "texto"}]
        s2 = [{"titulo": "Cap 2", "contenido": "texto"}]
        assert hash_secciones(s1) != hash_secciones(s2)

    def test_orden_importa(self):
        s1 = [{"titulo": "A", "contenido": "x"}, {"titulo": "B", "contenido": "y"}]
        s2 = [{"titulo": "B", "contenido": "y"}, {"titulo": "A", "contenido": "x"}]
        # El orden de secciones SÍ afecta el hash (json.dumps es sensible al orden de la lista)
        assert hash_secciones(s1) != hash_secciones(s2)
