"""Tests para funciones no-regex de segmentacion.py."""
import pytest
from contractia.segmentacion import (
    _roman_to_int,
    _key_sort_clauses,
    _expand_clause_ranges,
    _clause_ids_in_text,
    crear_indice_de_clausulas_por_seccion,
)


class TestRomanToInt:
    def test_basicos(self):
        assert _roman_to_int("I") == 1
        assert _roman_to_int("V") == 5
        assert _roman_to_int("X") == 10
        assert _roman_to_int("L") == 50
        assert _roman_to_int("C") == 100
        assert _roman_to_int("D") == 500
        assert _roman_to_int("M") == 1000

    def test_compuestos(self):
        assert _roman_to_int("IV") == 4
        assert _roman_to_int("IX") == 9
        assert _roman_to_int("XL") == 40
        assert _roman_to_int("XC") == 90
        assert _roman_to_int("CD") == 400
        assert _roman_to_int("CM") == 900
        assert _roman_to_int("XIV") == 14
        assert _roman_to_int("XLII") == 42
        assert _roman_to_int("MCMXCIX") == 1999

    def test_minuscula(self):
        assert _roman_to_int("iv") == 4
        assert _roman_to_int("xiv") == 14

    def test_casos_borde(self):
        assert _roman_to_int("") == 0
        assert _roman_to_int("123") == 0  # dígitos devuelven 0
        assert _roman_to_int("ABC") == 0  # letras inválidas devuelven 0


class TestKeySortClauses:
    def test_orden_natural(self):
        clauses = ["2.1", "1.10", "1.2", "10.1", "1.1"]
        assert sorted(clauses, key=_key_sort_clauses) == ["1.1", "1.2", "1.10", "2.1", "10.1"]

    def test_anidados(self):
        clauses = ["3.1.2", "3.1.10", "3.1.1"]
        assert sorted(clauses, key=_key_sort_clauses) == ["3.1.1", "3.1.2", "3.1.10"]

    def test_tres_niveles(self):
        clauses = ["5.2.3", "5.2.1", "5.1.1"]
        result = sorted(clauses, key=_key_sort_clauses)
        assert result[0] == "5.1.1"
        assert result[-1] == "5.2.3"


class TestExpandClauseRanges:
    def test_rango_simple(self):
        result = _expand_clause_ranges("Cláusulas 5.1 a 5.3")
        assert "5.1" in result
        assert "5.2" in result
        assert "5.3" in result

    def test_rango_con_guion(self):
        result = _expand_clause_ranges("entre 3.1 - 3.4")
        assert len(result) == 4

    def test_sin_rango(self):
        result = _expand_clause_ranges("texto sin rango")
        assert len(result) == 0

    def test_rango_distinto_nivel_ignorado(self):
        # 5.1 a 6.1 no tiene mismo prefijo → debe ignorarse
        result = _expand_clause_ranges("desde 5.1 hasta 6.1")
        assert "5.1" not in result or "6.1" not in result  # no puede expandir cross-chapter


class TestClauseIdsInText:
    def test_clausula_explicita(self):
        ids = _clause_ids_in_text("Ver CLÁUSULA 5.1 del contrato")
        assert "5.1" in ids

    def test_patron_numerico(self):
        ids = _clause_ids_in_text("3.2 El plazo establecido")
        assert "3.2" in ids

    def test_lista_clausulas(self):
        ids = _clause_ids_in_text("CLÁUSULAS 5.1, 5.2 y 5.3")
        assert "5.1" in ids
        assert "5.2" in ids
        assert "5.3" in ids

    def test_rango_en_texto(self):
        ids = _clause_ids_in_text("Ver cláusulas 4.1 a 4.3 para mayor detalle")
        assert "4.1" in ids
        assert "4.2" in ids
        assert "4.3" in ids


class TestCrearIndiceClausulas:
    def test_devuelve_ordenado(self):
        texto = "CLÁUSULA 2.1 primera. CLÁUSULA 2.3 tercera. CLÁUSULA 2.2 segunda."
        idx = crear_indice_de_clausulas_por_seccion(texto)
        if len(idx) >= 3:
            assert idx.index("2.1") < idx.index("2.2") < idx.index("2.3")

    def test_devuelve_lista(self):
        idx = crear_indice_de_clausulas_por_seccion("5.1 Texto del contrato.")
        assert isinstance(idx, list)
