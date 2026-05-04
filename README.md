# Plan de mejoras — `Multiagente_ContractIA_vs18.ipynb`

> **Alcance respetado:** no se modifica el procedimiento global del pipeline
> (carga → segmentación → auditoría de secuencia → grafo → multi-agente → chat)
> ni el código de regex de la celda 3 (`_norm_text`, `_CAP_RX`, `_ANEXO_RX`,
> `_CLAUSE_RX`, `_CLAUSE_LIST_RX`, `_RANGE_RX`, `_truncate_upper_block`,
> `_is_proper_caps_title`, `_extraer_*`, `_find_sections`, `_post_secciones`,
> `_clause_ids_in_text`, `_expand_clause_ranges`, etc.).

## Estado

- 📄 **Plan completo**: este documento.
- ✅ **P0 implementado** en [`Multiagente_ContractIA_vs19.ipynb`](./Multiagente_ContractIA_vs19.ipynb).
- ✅ **P1 implementado** en el mismo notebook (paralelización async, Pydantic structured output, persistencia, ego-graph, GraphRAG real en chat, métricas de tokens). La única excepción es `3.3` (context caching de Vertex), que se deja fuera porque la API actual de `langchain-google-vertexai` no la integra.
- ⏳ **P2–P3**: pendientes.

---

## Tabla de contenidos

1. [Bugs y riesgos de correctitud](#1-bugs-y-riesgos-de-correctitud)
2. [GraphRAG — análisis dedicado](#2-graphrag--análisis-dedicado)
3. [Rendimiento y costo](#3-rendimiento-y-costo)
4. [Robustez de E/S y JSON](#4-robustez-de-es-y-json)
5. [Configuración y secretos](#5-configuración-y-secretos)
6. [Observabilidad](#6-observabilidad)
7. [Calidad del código](#7-calidad-del-código)
8. [Mejoras del informe final](#8-mejoras-del-informe-final)
9. [Roadmap priorizado](#9-roadmap-priorizado)

---

## 1. Bugs y riesgos de correctitud

| # | Ubicación | Problema | Mejora |
|---|---|---|---|
| 1.1 | Celda 8, `_build_llm` | `MODELO_PRINCIPAL == MODELO_FALLBACK` ⇒ el `try/except` **no es un fallback real**, si Pro falla el fallback también falla. | Usar `gemini-2.5-flash` como fallback distinto, o eliminar el bloque. |
| 1.2 | Celda 6, `verificar_seguridad_documento` | En `except` retorna `True, ""` (fail-open). Un error de red **silenciosamente** marca el doc como seguro. | Cambiar a fail-closed: en error, abortar la auditoría o exigir confirmación explícita. |
| 1.3 | Celda 6, `verificar_seguridad_documento` | Se envía el **texto completo** del contrato en una sola llamada. Contratos largos exceden el context window y se truncan sin aviso. | Trocear por secciones (reusar `secciones`) y agregar OR de resultados. |
| 1.4 | Celda 5, `auditar_consistencia` | `_crear_agentes(llm)` se invoca **en cada sección** (54 veces). Reconstruye 3 `PromptTemplate` + 3 chains por sección. | Construir los agentes una sola vez fuera del loop y pasarlos como parámetro. |
| 1.5 | Celda 4, `construir_grafo_conocimiento` | `time.sleep(1)` por sección + sin retry: un único 429/500 en una sección **pierde esa parte del grafo** sin reintento. | Envolver en retry con backoff exponencial (`tenacity`) y eliminar el `sleep` fijo. |
| 1.6 | Celda 6, `ejecutar_auditoria_contrato` | `time.sleep(2)` por sección → 54 × 2s = 108 s de espera ociosa en el run real (40 min log). | Reemplazar por rate-limiter real basado en cuota, o eliminar si Vertex no lo exige. |
| 1.7 | Celda 7, `consultar_contrato_graphrag` | Inyecta `textos_completos` (todo el contrato) **en cada pregunta** del chat. Costo y latencia altísimos por turno. | Recuperar sólo las secciones relevantes vía el grafo + índice. |
| 1.8 | Celda 7, chat | No mantiene historial de conversación: cada pregunta es aislada. | Acumular `chat_history` y pasarlo al prompt. |

---

## 2. GraphRAG — análisis dedicado

### 2.1 Arquitectura actual

El GraphRAG vive en tres puntos del notebook:

| Etapa | Función | Celda |
|---|---|---|
| Construcción | `construir_grafo_conocimiento` | 4 |
| Recuperación (auditoría) | `obtener_contexto_grafo` | 4 |
| Recuperación (chat) | `consultar_contrato_graphrag` | 7 |

**Modelo de grafo:** `nx.DiGraph` con nodos = strings libres devueltos por el
LLM, aristas con atributos `relacion` y `contexto`. Los únicos nodos con
metadato `tipo` son los de sección (CAPITULO/ANEXO).

**Schema declarado en el prompt** (no validado en código):

- Entidades válidas: Cláusulas, Plazos, Roles, Entregables, Penalidades.
- Relaciones válidas: REFERENCIA_A, ESTABLECE_PLAZO, MODIFICA_A, DEPENDE_DE, OBLIGA_A.
- Auto-añadida en código: CONTIENE.

---

### 2.2 Problemas críticos de identidad de nodos

#### 2.2.1 Doble fallo: fragmentación + colisión simultáneas

El nodo "Cláusula 7.1" depende del string libre que devuelva el LLM, sin
canonicalización. Dos patologías coexisten:

- **Fragmentación**: variantes ortográficas crean nodos distintos para la
  misma entidad.

  ```text
  "Cláusula 7.1 (Capítulo VII)"
  "Cláusula 7.1 (Capitulo VII)"     # sin tilde
  "Cláusula N° 7.1 (Cap. VII)"
  "Cláusula 7.1 del Capítulo VII"
  ```

- **Colisión**: cuando el LLM **omite** la desambiguación,
  `"Cláusula 7.1"` del Capítulo VII y `"Cláusula 7.1"` del Anexo 22 se
  colapsan en el mismo nodo.

> **Impacto:** el grafo no representa fielmente la red de relaciones. Es
> ruido + entidades fusionadas indebidamente.

#### 2.2.2 La regla de desambiguación del prompt es estructuralmente débil

El prompt pide formato `"Cláusula 7.1 (Capítulo VII)"`, pero al LLM **no se le
pasa el número de capítulo como variable**, sólo el `<texto_seccion>`. Tiene
que inferirlo del propio contenido. En secciones largas o con múltiples
menciones de "Capítulo X" (referencias internas), el LLM puede usar el
capítulo equivocado.

> **Fix de bajo costo:** añadir al prompt un campo `<seccion_contenedora>`
> con el título y el número ya extraídos por `_extraer_num_cap` /
> `_extraer_num_anexo` (sin tocar regex; sólo se *consume* su salida).
> Así el LLM no infiere — copia.

#### 2.2.3 La canonicalización está disponible y no se usa

Ya existe `mapa_clausula_a_seccion` con todas las cláusulas en su forma
normalizada (`"7.1"`, `"3.3.1"`...) y su sección de origen.
**El grafo debería usar ese diccionario como autoridad** y mapear cualquier
string del LLM a un ID canónico antes de añadirlo como nodo. Hoy no existe
esta etapa de normalización.

---

### 2.3 Problemas en la construcción (`construir_grafo_conocimiento`)

#### 2.3.1 `DiGraph` se queda corto: hace falta `MultiDiGraph`

```python
G.add_edge(origen, destino, relacion=..., contexto=...)
G.add_edge(titulo_seccion, origen, relacion="CONTIENE", ...)
```

`DiGraph` permite **una sola arista por par (u, v)**. Si dos secciones
distintas reportan que la cláusula X se relaciona con Y por motivos
diferentes, **la segunda sobreescribe a la primera silenciosamente**. Mismo
problema con `CONTIENE` cuando un nodo aparece extraído desde dos secciones.

> **Fix:** `nx.MultiDiGraph()`.

#### 2.3.2 No se valida el schema de relaciones

El prompt enumera 5 relaciones válidas, pero el código acepta cualquier
string en `t["relacion"]`. Si el LLM devuelve `"VINCULA"`, `"AFECTA_A"` o
variantes españolas, entran al grafo y rompen los conteos por tipo. No hay
un `if t["relacion"] not in {...}: skip|raise`.

#### 2.3.3 Tipos de nodo perdidos

```python
G.add_node(titulo_seccion, tipo=sec.get("tipo", "DESCONOCIDO"))
```

Solo a las secciones se les anota `tipo`. Cláusulas, Roles, Plazos,
Entregables, Penalidades **no reciben atributo `tipo`** porque se crean
implícitamente vía `add_edge`. El prompt fuerza al LLM a clasificar
entidades, pero esa clasificación se descarta. No se puede después filtrar
el grafo por tipo (p. ej. "muéstrame todos los Plazos del contrato").

> **Fix:** pedir al LLM que devuelva también `tipo_origen` y `tipo_destino`
> por tripleta y persistirlo en `G.nodes[n]['tipo']`.

#### 2.3.4 Aristas a cláusulas inventadas

El LLM puede generar `destino = "Cláusula 9.99"` aunque esa cláusula no
exista en el contrato. Como el grafo no se cruza con
`mapa_clausula_a_seccion`, esas aristas viven en el grafo y luego se sirven
como contexto al auditor, contaminando los hallazgos.

> **Fix:** post-validación. Para cualquier nodo cuyo string contenga un
> patrón `\d+(?:\.\d+)+`, comprobar contra el set de cláusulas conocidas.
> Marcar como `inventado=True` o descartar.

#### 2.3.5 La arista `CONTIENE` se sobre-aplica

Por cada tripleta cuyo origen aparece en la sección, se añade
`(seccion → origen, CONTIENE)`. Esto crea CONTIENE incluso para entidades
que **no** son del propio capítulo (p. ej. Roles globales como
"Concesionario"). El grafo dice "Capítulo VII contiene Concesionario", lo
cual es semánticamente falso.

> **Fix:** emitir CONTIENE solo cuando el origen sea una cláusula con
> prefijo coincidente con el número del capítulo (ese prefijo ya está
> disponible en el flujo).

---

### 2.4 Problemas en la recuperación (`obtener_contexto_grafo`)

#### 2.4.1 Búsqueda lineal con falso positivo de prefijo

```python
for n in G.nodes():
    if re.search(rf'\b{re.escape(cid)}\b', str(n)):
        nodos_grafo.append(n)
```

Dos defectos:

- **Coste**: O(|V|) por cada cláusula local, ejecutado por cada sección ⇒
  O(N · |V| · S). Con |V| en cientos y S = 54 secciones, la fase de
  recuperación se vuelve lenta sin justificación.
- **Falso positivo sutil**: `\b3.3\b` no matchea en `"13.3"` (gracias al
  `\b` izquierdo) **pero sí matchea en `"3.3.1"`** porque el `.` no es `\w`
  y el `\b` derecho se satisface ahí. Buscar la cláusula `3.3` devolverá
  nodos de la cláusula `3.3.1`, `3.3.2`, etc. — y al revés no.

> **Fix sin tocar regex de la celda 3:** mantener un
> `dict[str, list[str]]` de `cid → [nombre_de_nodo, ...]` construido al
> final de la fase 1.5, indexado por igualdad estricta del ID canónico.
> La búsqueda pasa a O(1) y elimina el falso positivo.

#### 2.4.2 Solo 1-hop ⇒ deja fuera relaciones multi-salto

```python
for sucesor in G.successors(nodo):
for predecesor in G.predecessors(nodo):
```

Si la inconsistencia está en `A → B → C` (la cláusula que importa es C),
el contexto del grafo nunca la verá. Para un GraphRAG real conviene un
**ego-graph de profundidad k**, con `k = 2` por defecto y posibilidad de
llegar a 3 cuando la cláusula tiene grado bajo.

#### 2.4.3 Texto recuperado sin deduplicación

```python
texto_ref = mapa_textos[id_ref]["texto"]
contexto.append(f"  [TEXTO RECUPERADO DE {sucesor}]:\n{texto_ref}\n")
```

Si varios sucesores apuntan al mismo `id_ref`, el texto entero de esa
cláusula se concatena varias veces. En secciones densas el contexto puede
crecer mucho y desplazar al texto que importa dentro del context window.

> **Fix:** `set` de `id_ref` ya emitidos en la sección actual.

#### 2.4.4 Ignora el atributo `tipo` que sí podría guiar la recuperación

Aunque tipo de nodo está casi vacío hoy (problema 2.3.3), una vez corregido
se podría priorizar recuperar primero los nodos tipo Plazo cuando el
contexto se manda al cronista, tipo Rol cuando se manda al jurista, etc.
Hoy los tres agentes reciben **el mismo** contexto del grafo. Es desperdicio.

---

### 2.5 Problemas en el chat (`consultar_contrato_graphrag`)

#### 2.5.1 No es GraphRAG — es "dump everything"

```python
resumen_grafo = "\n".join(... todas las aristas ...)
textos_completos = "\n\n".join(... contenido de TODAS las secciones ...)
```

Se concatenan **todas las aristas** y **todas las secciones** del contrato
en cada turno del chat. El grafo no se usa para *seleccionar*; se usa como
decoración. Anula el propósito del GraphRAG (recuperar lo relevante) y
maximiza tokens por turno.

> **Fix de fondo:**
>
> 1. Extraer entidades de la pregunta (nombres de cláusulas, roles, palabras clave).
> 2. Localizar los nodos seed en el grafo.
> 3. Expandir ego-graph (k = 2).
> 4. Pasar al prompt **sólo** los textos de las secciones que contienen esos nodos + la sub-vista del grafo.

#### 2.5.2 Sin historial conversacional

El chat reinicia contexto en cada `input()`. No hay multi-turn.

---

### 2.6 Problemas transversales del grafo

#### 2.6.1 No hay métricas de calidad del grafo

Tras construirlo no se reporta nada útil para diagnosticar. Sería barato
calcular y mostrar:

- **Cobertura**: `% cláusulas en mapa_clausula_a_seccion presentes como nodo`.
- **Inventadas**: `% nodos cláusula que NO están en mapa_clausula_a_seccion`.
- **Fragmentación**: número de variantes por ID canónico (cuántas formas distintas del mismo `7.1`).
- **Distribución de relaciones por tipo** (detectaría LLM saliéndose del schema).
- **Componentes débilmente conexos** (¿el grafo está roto en islas?).
- **Top-N nodos por degree** (sanity check: deberían ser cláusulas centrales o roles principales).

#### 2.6.2 No persiste

Reconstruir el grafo cuesta ~40 min según los logs (54 secciones × 1 LLM
call con `time.sleep(1)`). No se serializa con `nx.write_graphml(G, ...)`
ni `pickle`. Cada ejecución del notebook (incluso solo para usar el chat)
lo recalcula.

#### 2.6.3 Visualización no escala

`spring_layout` sobre el grafo completo da un *hairball* ilegible cuando
hay >50 nodos. No hay:

- Color por `tipo`.
- Tamaño por `degree`.
- Filtro por componente conexa.
- Layout jerárquico (que aprovecharía la relación `CONTIENE`).

#### 2.6.4 Latencia: serial + sleep fijo

`construir_grafo_conocimiento` es estrictamente secuencial con
`time.sleep(1)` por sección. La extracción de tripletas por sección es
**independiente** ⇒ candidato natural a `ainvoke` con un `Semaphore` para
respetar la cuota.

---

## 3. Rendimiento y costo

| # | Ubicación | Mejora |
|---|---|---|
| 3.1 | Celdas 4 y 6 | **Paralelizar** las llamadas LLM por sección con `asyncio` + `ainvoke` o `ThreadPoolExecutor`. Hoy son estrictamente seriales. El log muestra 40 min para 54 secciones ⇒ con paralelismo 5–10× se baja a < 10 min. |
| 3.2 | Celda 5 | Las 3 llamadas (jurista, auditor, cronista) por sección son independientes ⇒ ejecutarlas en paralelo dentro de cada sección. |
| 3.3 | Celda 1 / 8 | `temperature=0.0` y prompts repetitivos ⇒ activar **context caching** de Vertex (Gemini 2.5 lo soporta) sobre las partes invariantes de cada prompt. Ahorro grande de tokens. |
| 3.4 | Celda 5 | Cronista podría correr con **gemini-2.5-flash** (cálculos de plazos no requieren razonamiento profundo) y reservar Pro para jurista/auditor. |
| 3.5 | Celda 4 | Persistir el grafo (`pickle` / `graphml`) y el `mapa_clausula_a_seccion` en disco para no recomputar entre runs del mismo contrato. |
| 3.6 | Celda 6 | Checkpoint de hallazgos por sección a disco (`jsonl`) ⇒ si falla en la sección 50, reanuda desde ahí. |

---

## 4. Robustez de E/S y JSON

| # | Ubicación | Mejora |
|---|---|---|
| 4.1 | Celda 4, `_parse_json_seguro` | Si el LLM devuelve JSON malformado, retorna `{}` silenciosamente ⇒ se pierden hallazgos sin trazabilidad. Loggear el `raw_output` truncado en un archivo de errores. |
| 4.2 | Celda 5, agentes | Definir **modelos Pydantic** (`Hallazgo`, `RespuestaAgente`) y usar `with_structured_output(...)` de LangChain. Elimina parseo frágil. |
| 4.3 | Celda 2, `procesar_documentos_carpeta` | Concatena `texto_completo` sin separador entre archivos, lo que puede fusionar el final de un PDF con el inicio del siguiente y romper la detección de secciones. Insertar `\n\n=== ARCHIVO: {file_name} ===\n\n`. |
| 4.4 | Celda 2 | `os.listdir` no es determinista entre OS; los archivos se procesan en orden arbitrario. Ordenar (`sorted(...)`) para resultados reproducibles. |
| 4.5 | Celda 0 | Si no existe el JSON de credenciales, sólo imprime un mensaje y continúa ⇒ todo lo siguiente fallará en cascada. Hacer `raise` o `sys.exit(1)`. |

---

## 5. Configuración y secretos

| # | Ubicación | Mejora |
|---|---|---|
| 5.1 | Celda 0 / 1 / 8 | `PROJECT_ID`, `LOCATION`, `MODELO_*`, `RUTA_CONTRATO_NUEVO`, `REPORTE_MD`, nombre del JSON de credenciales están **hardcoded**. Mover a variables de entorno o a un dict `CONFIG` en una sola celda. |
| 5.2 | Celda 0 | Subir un JSON de service account a un notebook es riesgo de fuga si el `.ipynb` se versiona. Documentar el uso de Application Default Credentials (`gcloud auth application-default login`) como alternativa. |
| 5.3 | Celda 0 | `instalar_dependencias` mezcla detección y reinstalación; en Colab ya están casi todas. Fijar versiones (`pydantic>=2`, `langchain-google-vertexai>=2`) en un `requirements.txt`. |

---

## 6. Observabilidad

| # | Mejora |
|---|---|
| 6.1 | Sustituir `print(...)` con emojis por `logging` (niveles INFO / WARN / ERROR) y un handler que escriba a archivo. Hoy todo el log se mezcla con la salida del notebook y se pierde. |
| 6.2 | Medir y reportar **tokens consumidos** por agente y por sección (callbacks de LangChain) → indispensable para controlar costo. |
| 6.3 | Después del grafo, en lugar de imprimir todos los nodos / aristas (puede ser enorme), imprimir un resumen y guardar el detalle en `grafo.json`. |
| 6.4 | El `tqdm` "Auditando Secciones" no muestra cuántos hallazgos lleva acumulados; añadir `pbar.set_postfix(hallazgos=N)`. |

---

## 7. Calidad del código

| # | Mejora |
|---|---|
| 7.1 | Celdas 3 y 4 mezclan helpers privados, lógica de negocio y prints. Separar en módulos `.py` (`segmentacion.py`, `grafo.py`, `agentes.py`, `pipeline.py`) e importarlos desde el notebook. Facilita testear. |
| 7.2 | Tests unitarios para las funciones que **no** son regex: `_roman_to_int`, `_validar_secuencia_clausulas`, `_parse_json_seguro`, `obtener_contexto_grafo`, `render_auditoria_markdown`. Hoy no hay ninguno. |
| 7.3 | Type hints incompletos: varias funciones devuelven `Dict` o `Any` cuando podrían devolver dataclasses / Pydantic models. |
| 7.4 | `try/except Exception` muy genéricos en `auditar_consistencia` y `construir_grafo_conocimiento`; capturar excepciones específicas y propagar las críticas. |
| 7.5 | `iniciar_chat_interactivo` usa `input()`: en Jupyter funciona pero bloquea el kernel. Considerar `ipywidgets.Text` para una UX no bloqueante. |

---

## 8. Mejoras del informe final

| # | Mejora |
|---|---|
| 8.1 | `render_auditoria_markdown` no incluye estadísticas por severidad (cuántos ALTA / MEDIA / BAJA). Añadir un mini-dashboard al inicio. |
| 8.2 | No deduplica hallazgos: si jurista y auditor reportan lo mismo en términos parecidos, aparecen dos veces. Añadir un paso de deduplicación por similitud (hash del par `clausula_afectada + tipo`). |
| 8.3 | El informe no incluye el grafo (al menos un thumbnail PNG embebido o un enlace al `.graphml` exportado). |

---

## 9. Roadmap priorizado

### P0 — bugs / correctitud (sesión corta, impacto alto) — ✅ implementado en `Multiagente_ContractIA_vs19.ipynb`

- [x] **1.2** Fail-closed en `verificar_seguridad_documento`.
- [x] **1.3** Trocear el escaneo de seguridad por secciones.
- [x] **1.4** Construir agentes una sola vez fuera del loop.
- [x] **1.5** Retry con backoff en construcción de grafo (`tenacity`, 4 intentos, exp 2–30s).
- [x] **2.3.1** Migrar a `nx.MultiDiGraph`.
- [x] **2.2.2** Pasar `<seccion_contenedora>` al prompt de extracción.
- [x] **2.2.3** Etapa de canonicalización post-LLM contra `mapa_clausula_a_seccion` (`_canonicalizar_nodo`).
- [x] **2.4.1** Índice `dict[cid → list[nodo]]` para `obtener_contexto_grafo` (`construir_indice_nodos_por_cid`).
- [x] **2.4.3** *Bonus*: deduplicación de textos recuperados en `obtener_contexto_grafo`.

### P1 — rendimiento y robustez (impacto grande en tiempo y tokens) — ✅ implementado

- [x] **3.1 / 3.2** Paralelización async (`asyncio` + `Semaphore`): secciones del grafo, agentes por sección, secciones de la auditoría y escaneo de seguridad.
- [ ] **3.3** Context caching de Vertex — *no implementado*: la API de `CachedContent` aún no se integra en `langchain-google-vertexai` de forma estable; se deja para una iteración futura usando el SDK de Vertex directamente.
- [x] **3.5 / 3.6** Persistencia: grafo en `cache/grafo_<hash>.pkl` y checkpoint de hallazgos por sección en `cache/hallazgos_<hash>.jsonl` (auditoría resumible).
- [x] **2.3.3** `tipo` persistido en `G.nodes[n]['tipo']` (lo emite el LLM con `tipo_origen`/`tipo_destino`) y `relacion` validada contra `RELACIONES_VALIDAS`; las que no encajan se descartan con conteo.
- [x] **2.5.1** Chat reescrito como GraphRAG real: extrae cláusulas/capítulos/anexos de la pregunta → seeds → ego-graph k=2 → solo manda al prompt el sub-grafo + textos de las secciones tocadas. Soporta multi-turn (últimas 3 vueltas).
- [x] **2.4.2 / 2.4.3** `obtener_contexto_grafo` usa `nx.ego_graph` profundidad 2 con deduplicación por `(u, v, relacion)` y por id de cláusula referenciada.
- [x] **4.2** `with_structured_output(SchemaPydantic)` para los 3 agentes, la extracción del grafo y el escaneo de seguridad. Modelos: `Hallazgo`, `RespuestaJurista`, `RespuestaAuditor`, `RespuestaCronista`, `RespuestaSeguridad`, `TripletaGrafo`, `RespuestaExtraccion`.
- [x] **6.2** `TokenCounterCallback` propagado a todas las llamadas (tag por agente: `jurista`, `auditor`, `cronista`, `grafo`, `seguridad`, `chat`); resumen incluido en el informe Markdown final.

### P2 — calidad y mantenibilidad

- [ ] **5.1 / 7.1** Modularización en `.py` y configuración externa.
- [ ] **2.6.1** Métricas de calidad del grafo (cobertura, fragmentación, inventadas).
- [ ] **6.1** Migrar a `logging` con handler a archivo.
- [ ] **8.1 / 8.2** Dashboard de severidades + deduplicación de hallazgos.
- [ ] **7.2** Tests unitarios para funciones no-regex.

### P3 — pulido

- [ ] **2.6.3** Visualización: color por tipo, tamaño por degree, layout jerárquico.
- [ ] **2.3.5** Restringir `CONTIENE` a cláusulas con prefijo coincidente.
- [ ] **8.3** Embeber grafo en el informe final.
- [ ] **1.8 / 2.5.2** Historial multi-turn en el chat.
- [ ] **7.5** UX de chat con `ipywidgets`.
