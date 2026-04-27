# RAG: fundamentos

Documento de referencia personal para volver a entender el pipeline RAG desde cero. Se escribió consolidando una sesión de aprendizaje guiada, así que está pensado para releerse, no como tutorial paso a paso.

---

## Tabla de contenidos

1. [Qué es RAG y por qué existe](#1-qué-es-rag-y-por-qué-existe)
2. [Visión global del pipeline](#2-visión-global-del-pipeline)
3. [Las fases del pipeline básico (Q&A aislado)](#3-las-fases-del-pipeline-básico-qa-aislado)
4. [La distinción crítica: construcción vs consulta](#4-la-distinción-crítica-construcción-vs-consulta)
5. [Conceptos clave (los matices que duelen si no los sabes)](#5-conceptos-clave)
6. [RAG conversacional: añadir memoria al pipeline](#6-rag-conversacional-añadir-memoria-al-pipeline)
7. [Trampas y errores conocidos](#7-trampas-y-errores-conocidos)

---

## 1. Qué es RAG y por qué existe

**RAG = Retrieval-Augmented Generation** (generación aumentada por recuperación).

Un LLM "puro" tiene dos limitaciones grandes:

- **No sabe nada que no estuviera en sus datos de entrenamiento** (tu CV, tus documentos internos, datos de hoy...).
- **Si no sabe algo, se lo inventa** (alucinaciones), porque está entrenado para producir texto plausible, no para decir "no lo sé".

RAG resuelve las dos cosas con una idea simple: **antes de pedirle al LLM que responda, le buscamos los fragmentos relevantes en una base de conocimiento y se los pegamos al prompt**. El LLM ya no responde de memoria, responde leyendo.

> **Analogía:** un LLM puro es un estudiante muy listo haciendo un examen sin apuntes. Se le da bien razonar, pero si no se sabe la materia, fabula. RAG es ese mismo estudiante con **acceso a apuntes seleccionados durante el examen**: no contesta de memoria, lee primero lo relevante.

El nombre lo dice todo:

- **R**etrieval → recuperar fragmentos relevantes (lo hace FAISS u otra vector store).
- **A**ugmented → "aumentado": el prompt del LLM lleva esos fragmentos pegados.
- **G**eneration → generación de la respuesta final (lo hace el LLM).

---

## 2. Visión global del pipeline

El pipeline RAG es **una cadena de transformaciones**. Cada fase recibe una cosa y produce otra distinta. Si entiendes qué entra y qué sale en cada paso, entiendes RAG.

```
profile.md
    │
    │ (2) ingesta
    ▼
Document
    │
    │ (3) chunking
    ▼
list[Document]   ←── trozos pequeños
    │
    │ (4) embeddings
    ▼
list[vector[384]]
    │
    │ (5) vector store
    ▼
índice FAISS    ←── todo lo anterior se hace UNA SOLA VEZ
═══════════════════════════════════════════════════════════
                    ↓ a partir de aquí, una vez por consulta
pregunta del usuario
    │
    │ (6) retriever
    ▼
list[chunk relevante]
    │
    │ (7) build_context
    ▼
str (CONTEXTO formateado)
    │
    │ (8) generate_answer (LLM)
    ▼
respuesta en español
```

Las fases 1-9 son el RAG "Q&A aislado" (una pregunta, una respuesta, sin memoria). Las fases 10-12 añaden la capa conversacional encima.

---

## 3. Las fases del pipeline básico (Q&A aislado)

### Fase 1 — Configuración (`src/config.py`)

Constantes globales y carga de `.env` para que `HF_TOKEN` esté disponible. Es donde se centralizan los valores que el resto del pipeline importa:

```python
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RAG_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CHUNK_SIZE      = 500
DEFAULT_CHUNK_OVERLAP   = 100
DEFAULT_RETRIEVAL_K     = 3
DEFAULT_MAX_TOKENS      = 500
DEFAULT_TEMPERATURE     = 0.2
```

**Por qué importa centralizar:** ¿quieres probar otro modelo de embeddings? Cambias una constante y se aplica en todo el pipeline. Si los valores estuvieran hardcodeados, sería un infierno experimentar.

### Fase 2 — Ingesta (`src/ingestion.py`)

**Entra:** ruta a un archivo (en este caso `profile.md`).
**Sale:** un `Document` de LangChain con `page_content` (el texto) y `metadata` (origen, título, tipo).

```python
Document {
    page_content: "# Ismael - Perfil profesional\n## Resumen\n...",
    metadata: {"source": "...", "title": "profile", "type": "markdown"}
}
```

**Por qué `Document` y no string crudo:** los metadatos viajan con el texto por todo el pipeline. Cuando al final el retriever te devuelve un chunk, sabes de dónde salió. Esa trazabilidad es oro para depurar y para citar fuentes en la UI.

> **En proyectos reales** la ingesta es donde se complica todo: cargar PDFs, HTMLs, transcripciones, etc. La salida sigue siendo un `Document` con la misma estructura — el resto del pipeline no se entera de la complicación.

### Fase 3 — Chunking (`src/chunking.py`)

**Entra:** lista de `Document`.
**Sale:** lista de `Document` más pequeños.

Trocea cada documento usando `RecursiveCharacterTextSplitter`. **Tres razones para trocear:**

1. **Precisión semántica:** un solo vector para todo el CV sería una "sopa" mezclada de muchos temas. Con chunks pequeños, cada uno tiene un vector concentrado en un tema → match limpio en búsqueda.
2. **Límite de contexto del LLM:** los modelos tienen techo de tokens. No puedes meterles 50 páginas.
3. **Coste:** mandar al LLM solo los 3 chunks relevantes en lugar de las 50 páginas es ~100x más barato.

**El solapamiento (`chunk_overlap=100`) es un seguro contra cortes en sitios malos.** Cada chunk comparte sus últimos 100 caracteres con el siguiente, para que una frase importante que cayera justo entre dos chunks no se parta a la mitad.

`RecursiveCharacterTextSplitter` es "recursivo" porque intenta primero partir por dobles saltos de línea (entre párrafos), si los chunks aún son grandes parte por saltos simples, después por puntos, etc. Respeta la estructura del texto lo más posible.

### Fase 4 — Embeddings (`src/embeddings.py`)

**Entra:** texto (string o chunk).
**Sale:** vector de 384 números (con `all-MiniLM-L6-v2`).

**Definición correcta de embedding:** representación de un texto como vector de números, **construida de forma que distancias en el espacio reflejan similitudes de significado**. *"rey"* y *"reina"* quedan cerca; *"rey"* y *"plebeyo"* quedan lejos.

Esa propiedad es lo que hace al embedding útil. Sin ella, sería un hash arbitrario. Con ella, FAISS puede hacer su magia.

> **Detalle clave:** en RAG no vectorizamos palabras sueltas, vectorizamos **chunks completos** (frases o párrafos). El modelo "lee" el chunk entero y produce un único vector que representa su significado global.

#### Tipos de embeddings — la división histórica

| | Estáticos (1ª gen, 2013-2017) | Contextuales (2018-presente) |
|---|---|---|
| Ejemplos | Word2Vec, GloVe, FastText | BERT, MiniLM, Sentence-Transformers |
| Vector por palabra | Fijo, siempre el mismo | Depende del contexto |
| Captura ambigüedad? | No | Sí |
| Buenos para frases? | Mediocre | Sí |
| Tu pipeline | ❌ | ✅ `all-MiniLM-L6-v2` |

**Por qué importa la diferencia:** la palabra `"banco"` puede ser mueble, entidad financiera o grupo de peces. Word2Vec le da el mismo vector siempre. BERT y derivados le dan vectores distintos según la frase. **Para RAG necesitas contextuales y de frase**, porque comparas párrafos enteros.

#### Cómo identificar qué tipo de embedding usa un modelo

| Pista | Qué te dice |
|---|---|
| Prefijo `sentence-transformers/...` en HuggingFace | Contextual + entrenado para frases |
| Prefijo `BAAI/bge-...`, `intfloat/e5-...` | Contextuales, estado del arte |
| Dimensiones 100-300 | Sospecha estático (Word2Vec, GloVe) |
| Dimensiones 384, 768, 1024+ | Casi siempre contextuales |

> **Analogía mental:** los estáticos son como un diccionario en papel: cada palabra tiene una entrada fija. Los contextuales son como un humano leyendo: la misma palabra significa cosas distintas según la frase, y "lo entiende" mirando el contexto.

### Fase 5 — Vector store (`src/vector_store.py`)

**Entra:** chunks + función de embeddings.
**Sale:** un índice FAISS (en memoria o en disco).

FAISS guarda **dos cosas en paralelo**:

```
Índice de vectores              Mapa interno
─────────────────────           ──────────────────────────────────
posición 0: [0.11, -0.42, ...] ─┬─► Document(page_content="# Ismael...", metadata={...})
posición 1: [0.33,  0.18, ...] ─┼─► Document(page_content="## Experiencia...", metadata={...})
posición 2: [0.51, -0.07, ...] ─┼─► Document(page_content="## Proyectos...", metadata={...})
posición 3: [0.22,  0.15, ...] ─┴─► Document(page_content="## Stack tecnico...", metadata={...})
```

**FAISS no es solo "almacén" — es indexación + búsqueda eficiente.** Si solo guardase los vectores tendrías que comparar uno a uno con un bucle (muy lento con millones de chunks). FAISS usa estructuras especializadas (clusterings, particiones, cuantización) para encontrar los `k` más parecidos en milisegundos sobre millones de vectores.

> **El nombre lo delata:** FAISS = **F**acebook **AI S**imilarity **S**earch. Su razón de existir es buscar por similitud rápido, no almacenar.

> **Analogía:** un diccionario en papel **podría** ser una lista desordenada — pero está alfabetizado para que vayas directo a la sección "E" sin leer las 4 secciones anteriores. FAISS hace lo mismo en un espacio de 384 dimensiones, donde no se puede ordenar alfabéticamente: divide el espacio en zonas y va directo a la zona correcta.

**Vector store y "base de datos vectorial" son sinónimos** (con un matiz informal: vector store suele ser librería en proceso, sin servidor; vector database suele ser producto cliente-servidor como Pinecone o Qdrant). LangChain los unifica todos bajo la interfaz `VectorStore`, así que cambiar de FAISS a Chroma, Qdrant o Pinecone es cambiar una sola línea.

### Fase 6 — Retriever (`src/retriever.py`)

**Entra:** vectorstore + pregunta del usuario + `k` (cuántos chunks).
**Sale:** los `k` chunks más relevantes para esa pregunta, ordenados por similitud.

Internamente:

```
Paso 1: Vectoriza la pregunta con el MISMO modelo de embeddings
        "¿Qué tecnologías domina Ismael?" → [0.07, 0.01, -0.01, ...]

Paso 2: FAISS compara ese vector con todos los del índice
        Calcula similitud (coseno o L2) con cada chunk.

Paso 3: Devuelve los k mejores como Documents (texto + metadatos),
        ordenados por similitud descendente.
```

> **El "mismo modelo de embeddings" es crítico.** Chunks y pregunta deben vivir en el mismo espacio vectorial. Si vectorizas chunks con `all-MiniLM-L6-v2` y preguntas con OpenAI, los espacios no son comparables y FAISS devuelve ruido aleatorio sin avisar de error.

`normalize_sources()` es la segunda función de la fase 6: transforma los `Document` de LangChain en **dicts planos** (`{"id", "title", "content", "metadata"}`) listos para serializar a JSON o pintar en una UI. Aísla la API de las peculiaridades de LangChain.

### Fase 7 — Contexto (`src/context.py`)

**Entra:** `sources` (lista de dicts del retriever).
**Sale:** un único string formateado, listo para meter en el prompt del LLM.

```python
def build_context(sources):
    return "\n\n".join(
        f"[{index}] {source['content']}" for index, source in enumerate(sources)
    )
```

Resultado:

```
[0] ## Stack tecnico
- Python
- FastAPI
...

[1] ## Proyectos
### RAG de porfolio
...

[2] # Ismael - Perfil profesional
...
```

#### El detalle más importante de la fase 7: `list[dict]` vs `str`

La fase 6 produce **datos estructurados** (`list[dict]`). La fase 7 los aplana a **texto plano** (`str`). Es lo que parece duplicación pero **no lo es** — hay dos audiencias distintas:

| | Para qué sirve | Quién la consume |
|---|---|---|
| `sources` (list[dict]) | UI de fuentes, API JSON, logs | Tu código, tu frontend |
| `context` (str) | Prompt del LLM | El modelo |

> **¿El LLM no entiende JSON?** Sí lo entiende — pero rinde peor. Pasarle JSON crudo cuesta más tokens (y pasta), distrae al modelo con sintaxis irrelevante, y se aleja del estilo natural con el que está entrenado. Aplanar a texto plano formateado es **claramente mejor** para tareas de comprensión y síntesis.

> **Los marcadores `[0]`, `[1]`, `[2]` no son decorativos.** Separan visualmente los chunks (el modelo no los confunde como un único bloque), permiten al modelo citar (*"según [2]..."*), y facilitan tu depuración al revisar logs.

> **Analogía:** `sources` son **ingredientes en táperes separados** (lechuga, tomate, atún — cada uno con etiqueta). El LLM no come ingredientes sueltos, come platos preparados. La fase 7 es el cocinero que monta la **ensalada** lista para servir. Misma comida, distinto formato.

### Fase 8 — Generación (`src/generation.py`)

**Entra:** `context` (string) + `question` (string).
**Sale:** la respuesta del LLM en lenguaje natural.

Es donde se llama al LLM. Cuatro piezas:

#### `SYSTEM_PROMPT` — instrucciones al modelo

```python
SYSTEM_PROMPT = """
Eres un asistente RAG para responder preguntas sobre el perfil profesional de Ismael.
Responde en español, de forma clara y directa.
Usa exclusivamente el CONTEXTO proporcionado.
Si el contexto no contiene información suficiente, dilo claramente y no inventes datos.
""".strip()
```

Cada línea hace un trabajo:

| Línea | Función |
|---|---|
| `Eres un asistente RAG...` | Identidad/rol |
| `Responde en español, de forma clara y directa` | Idioma + estilo |
| `Usa exclusivamente el CONTEXTO proporcionado` | **Anti-alucinación** — la regla más importante |
| `Si el contexto no contiene información suficiente, dilo claramente` | El "permiso" para decir "no lo sé" |

> **Esa última línea es de las más importantes en RAG.** Sin ella, los LLMs tienden a inventarse algo plausible cuando no encuentran la respuesta en el contexto. Con ella, dicen "no tengo esa información" y te ahorran alucinaciones.

#### `build_user_input()` — plantilla del mensaje del usuario

```python
def build_user_input(context, question):
    return f"""
CONTEXTO:
{context}

PREGUNTA:
{question}
""".strip()
```

Une contexto y pregunta en un único string que va dentro de `{"role": "user", "content": ...}`. **No es la UI** — es solo una plantilla de texto. La interfaz gráfica (Gradio) está en la fase 12, no aquí.

#### `extract_message_text()` — adaptador de la respuesta

La API de Hugging Face puede devolver la respuesta como string simple o como lista de bloques. Esta función abstrae esa diferencia para que el resto del código siempre reciba un string limpio.

#### Hiperparámetros: `max_tokens` y `temperature`

| Parámetro | Qué hace | Valor por defecto |
|---|---|---|
| `max_tokens` | Techo de tokens generables | 500 (≈ 3-4 párrafos) |
| `temperature` | Cuánto azar mete al elegir palabras | 0.2 (casi determinista) |

> **Truco mental para `temperature`:** alta = poeta (improvisa), baja = abogado (cita literalmente). En RAG quieres abogado.

#### Los roles del API de chat: `system`, `user`, `assistant`

| Rol | Quién es | Qué contiene |
|---|---|---|
| `system` | Tú, el desarrollador | Instrucciones generales: personalidad, reglas, tono. El modelo las lee al principio y las "graba" durante todo el chat. |
| `user` | El usuario humano | Lo que la persona escribe. La pregunta del momento. |
| `assistant` | El propio LLM (en turnos previos) | Lo que el modelo respondió antes. Su memoria del chat. |

En la fase 8 (Q&A aislado) **solo aparecen `system` y `user`**, sin `assistant`, porque cada pregunta se trata como independiente. `assistant` aparece a partir de la fase 10 (conversacional) cuando hay turnos previos que recordar.

### Fase 9 — Pipeline (`src/pipeline.py`)

Empaqueta el `vectorstore` + función de generación en una `dataclass`. Su método `answer_question()` hace el recorrido completo de las fases 6, 7 y 8.

```python
@dataclass
class RagPipeline:
    vectorstore: Any
    answer_generator: AnswerGenerator = generate_answer

    def answer_question(self, question, k=DEFAULT_RETRIEVAL_K):
        docs = retrieve_documents(self.vectorstore, question, k=k)  # fase 6
        sources = normalize_sources(docs)                           # fase 6
        context = build_context(sources)                            # fase 7
        answer = self.answer_generator(context, question)           # fase 8
        return {"answer": answer, "sources": sources}
```

**`build_rag_pipeline()` ejecuta las fases 2-5 una sola vez** (cargar Markdown, trocear, embeddings, FAISS) y deja el `RagPipeline` listo. Es la separación que se explica en la sección siguiente y que define cómo se diseña una app RAG real.

---

## 4. La distinción crítica: construcción vs consulta

El pipeline **no hace las 9 fases del tirón cada vez que llega una pregunta**. Hace dos cosas distintas en dos momentos distintos:

| Fases | Cuándo se ejecutan | Cuántas veces |
|---|---|---|
| 1 | Al importar el módulo | Una vez (constantes globales) |
| **2-5** | En `build_rag_pipeline()` | **Una vez al arrancar** |
| **6-8** | En `answer_question()` | **Una vez por pregunta** |

### Por qué importa

Si reconstruyeras el FAISS en cada pregunta:

```
Usuario pregunta → cargar Markdown → trocear → vectorizar → construir FAISS → buscar → responder
                  └────────────── 5-10 segundos ──────────────┘   └─ 1s ─┘
```

Cada respuesta tardaría 5-10 segundos solo en preparar el índice. Con 1.000 chunks serían minutos. Inutilizable.

Con la separación correcta:

```
Arranque del servidor → cargar → trocear → vectorizar → FAISS listo (5-10s, una sola vez)

Usuario pregunta → buscar en FAISS → responder
                  └─── 1-2s ────┘
```

**El FAISS se construye una vez y se reutiliza** para miles de preguntas. Por eso se empaqueta en la `dataclass` — el `vectorstore` queda vivo en memoria dentro del objeto.

> **Analogía:** una biblioteca. **Construir el catálogo** (fases 2-5) es leer todos los libros, clasificarlos, ponerlos en estanterías ordenadas. Tarda días, se hace una vez al inaugurar. **Atender una consulta** (fases 6-8) es ir a la sección correcta y traer los libros relevantes. Tarda segundos, se hace miles de veces. Sería absurdo reordenar toda la biblioteca cada vez que llega un visitante.

---

## 5. Conceptos clave

### 5.1. Embeddings — la propiedad que importa

**Mala definición:** "convertir texto en vectores".
**Definición correcta:** convertir texto en vectores **construidos de forma que distancias reflejen similitudes de significado**.

Sin esa propiedad, podría ser cualquier hash. Con ella, comparar vectores equivale a comparar significados.

### 5.2. El "mismo modelo de embeddings" para chunks y pregunta

Esto es de lo que más se les pasa a los principiantes y rompe el sistema sin avisar.

- Chunks vectorizados con `all-MiniLM-L6-v2` en la fase 4.
- La pregunta en la fase 6 **TAMBIÉN** debe vectorizarse con `all-MiniLM-L6-v2`.

Si por error usas modelos distintos, los espacios vectoriales no son comparables. El sistema **no te avisa con un error** — devuelve chunks irrelevantes silenciosamente. LangChain evita esto recordando con qué embedder se construyó el vectorstore y reutilizándolo.

### 5.3. FAISS recupera, el LLM responde

Confusión típica: pensar que FAISS "responde" o "le pasa la respuesta al usuario".

**FAISS no responde nada.** FAISS solo recupera fragmentos de texto. **El LLM es el único que genera lenguaje** y produce la respuesta final.

```
Usuario  ──pregunta──►  pipeline
                          │
                          ├──► Embedder vectoriza la pregunta
                          ├──► FAISS devuelve k chunks (interno, usuario no lo ve)
                          ├──► build_context monta el bloque [0] [1] [2]
                          ├──► LLM redacta la respuesta
                          ▼
Usuario  ◄──respuesta──   pipeline
```

> **Analogía del estudiante en el examen.** Tú (el usuario) le haces una pregunta. El estudiante (LLM) es muy listo y redacta bien, pero **no se sabe los apuntes**. FAISS es el bibliotecario: el estudiante le pide "tráeme las páginas relevantes" y FAISS le pone los trozos en la mesa. El estudiante **lee** y **escribe** la respuesta con sus propias palabras. La nota del examen evalúa lo que escribió el estudiante. El bibliotecario no escribió nada — solo trajo libros.

### 5.4. Anti-alucinación — el "no sé" en el system prompt

Sin instrucciones específicas, los LLMs alucinan cuando no saben algo. Las dos líneas que minimizan eso:

```
Usa exclusivamente el CONTEXTO proporcionado.
Si el contexto no contiene información suficiente, dilo claramente y no inventes datos.
```

> El LLM está entrenado para producir texto plausible, no verdadero. Si le das permiso explícito para decir "no lo sé", lo dirá. Si no se lo das, fabula.

### 5.5. Roles `system` / `user` / `assistant`

La API de chat espera una lista de mensajes etiquetados:

```python
messages = [
    {"role": "system",    "content": "Eres un asistente RAG..."},
    {"role": "user",      "content": "¿Qué tecnologías domina?"},
    {"role": "assistant", "content": "Python, FastAPI..."},     # solo si hay turnos previos
    {"role": "user",      "content": "¿Y qué proyectos tiene?"}
]
```

- **`system`** se manda **una sola vez** al principio.
- **`user`** y **`assistant`** se alternan formando el historial.
- **Sin `assistant`** = primera pregunta de la conversación o Q&A aislado (fase 8).
- **Con `assistant`** = el LLM tiene memoria de turnos previos (fase 10+).

---

## 6. RAG conversacional: añadir memoria al pipeline

El RAG básico (fases 1-9) trata cada pregunta como independiente. Para convertirlo en chat hay que inyectar **historial en dos sitios distintos**:

### 6.1. Los dos historiales

#### Historial en la generación (lo fácil)

El LLM recibe los turnos previos como mensajes `user`/`assistant`:

```python
messages = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "¿Qué tecnologías domina Ismael?"},
    {"role": "assistant", "content": "Python, FastAPI, LangChain..."},
    {"role": "user",      "content": "¿Y qué proyectos tiene con eso?"},
]
```

El modelo entiende perfectamente "con eso" porque ve los turnos previos. Esto resuelve referencias anafóricas (*"amplíame eso"*, *"el segundo punto"*).

#### Historial en la recuperación (lo crítico)

**FAISS no entiende historial.** Solo recibe **un string** y le calcula su embedding. Si le mandas literalmente *"¿Y qué proyectos tiene con eso?"*, el embedding de esa frase es genérico (no contiene "Ismael", no contiene "proyectos" como tema central) y no recupera nada útil.

**Solución:** antes de buscar, llamar al LLM para que **reformule** la pregunta a una versión autocontenida. Esto es el `condense_question`:

```
Original    : "¿Y qué proyectos tiene con eso?"
Condensada  : "¿Cuáles son los proyectos en los que utiliza Ismael estas tecnologías?"
```

La condensada **sí** tiene términos ancla y su embedding cae cerca de los chunks correctos.

### 6.2. El flujo completo

```
1. usuario:    "¿Y qué proyectos tiene con eso?"
                          │
                          ▼
2. condense_question(history, pregunta_original)
                          │
                          ▼  (LLM reescribe la pregunta usando el historial)
3. pregunta_condensada: "¿Cuáles son los proyectos en los que utiliza Ismael estas tecnologías?"
                          │
                          ▼
4. embed(pregunta_condensada) ──> vector
                          │
                          ▼
5. FAISS busca chunks parecidos al vector
                          │
                          ▼
6. chunks recuperados
                          │
                          ▼
7. LLM recibe { system + chunks, history, pregunta ORIGINAL }
                          │
                          ▼
8. respuesta natural en español
                          │
                          ▼
9. history.append({"role": "user",      "content": pregunta_original})
   history.append({"role": "assistant", "content": respuesta})
```

### 6.3. Reglas críticas del flujo conversacional

#### `condense_question` solo alimenta a FAISS

**La pregunta condensada NO va al LLM de generación.** Su único uso es como argumento de `faiss.search()`. Una vez recuperados los chunks, **se descarta**.

```python
# Único sitio donde se usa la condensada:
chunks = faiss.search(embed(pregunta_condensada), k=4)
# A partir de aquí, pregunta_condensada nunca más aparece.
```

¿Por qué? Si pasaras también la condensada al LLM de generación:
- Sería información redundante (el historial ya da ese contexto al modelo).
- Confundiría al modelo (¿son dos preguntas distintas?).
- Rompería la naturalidad de la respuesta (sonaría artificial).

#### Lo que se guarda en el historial es siempre la pregunta original

```python
history.append({"role": "user", "content": pregunta_ORIGINAL})  # ✅
# NO esto:
history.append({"role": "user", "content": pregunta_condensada})  # ❌
```

Razones:
1. **Fidelidad a lo que el usuario realmente dijo.** El historial es el registro de la conversación, no de lo que tu pipeline hizo por dentro.
2. **El LLM funciona mejor con flujo natural.** Está entrenado con conversaciones humanas con anáforas y abreviaturas.
3. **Las reformulaciones acumulan ruido.** Si guardas la condensada, los siguientes turnos arrastran toda la basura previa (efecto bola de nieve).
4. **Si condense se equivoca un día, no contaminas el historial para siempre.**

#### Reparto de qué va dónde

| Componente | Qué recibe |
|---|---|
| `condense_question` (LLM) | history + pregunta original → produce pregunta condensada |
| FAISS | embed(pregunta condensada) → devuelve chunks |
| `generate_answer_with_history` (LLM) | system + chunks + history + pregunta **original** → respuesta |
| Historial guardado | pregunta **original** + respuesta |

> **Frase mental:** *"`condense_question` produce un string desechable de un solo uso, exclusivamente para alimentar al retriever. El historial guarda lo que el usuario y el asistente realmente dijeron."*

### 6.4. Componentes del pipeline conversacional

#### `CONDENSE_SYSTEM_PROMPT`

```python
CONDENSE_SYSTEM_PROMPT = """
Dada una conversación previa y una pregunta de seguimiento, reformula la pregunta
como una pregunta independiente en español, incluyendo el contexto necesario del
historial para que pueda entenderse por sí sola.

Reglas:
- No respondas a la pregunta, solo reescríbela.
- Si la pregunta ya es autónoma, devuélvela sin cambios.
- No añadas explicaciones, etiquetas ni comillas. Devuelve solo la pregunta reformulada.
""".strip()
```

#### `condense_question()`

- Mismo LLM que la generación, prompt específico, **temperatura 0** (queremos reformulación estable, no creativa).
- Si no hay historial, devuelve la pregunta tal cual y se ahorra la llamada.

#### `generate_answer_with_history()`

Variante de `generate_answer()` que inserta el historial entre el `system` y la pregunta actual:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages.extend(history)
messages.append({"role": "user", "content": build_user_input(context, question)})
```

El **contexto RAG sigue pegado a la pregunta** (en el último mensaje `user`) para que el modelo no lo confunda con un mensaje del usuario.

#### `ConversationalRagPipeline`

Empaqueta vectorstore + lista de turnos. Su método `chat()` devuelve también la pregunta autónoma para depurar:

```python
return {
    "answer": answer,
    "sources": sources,
    "standalone_question": standalone,
}
```

> **Sobre el límite de historial:** el pipeline lo deja crecer sin recortar deliberadamente para observar cuándo degrada la calidad o cuándo la API se queja por tokens. Qwen2.5-7B-Instruct admite hasta 32k tokens, pero el efecto *lost in the middle* aparece antes. Para cortar: `self.history = self.history[-N:]` al final de `chat()`.

---

## 7. Trampas y errores conocidos

### 7.1. `TypeError: ChatInterface.__init__() got an unexpected keyword argument 'type'`

**Causa:** Gradio 6 ya no acepta `type="messages"`. Ese formato era opcional en Gradio 4/5 (cuando el default era `tuples`) y obligatorio para historiales tipo OpenAI. En Gradio 6 es el único formato soportado, así que el parámetro `type` ha desaparecido.

**Solución:** quitar la línea. El resto sigue igual:

```python
demo = gr.ChatInterface(
    fn=gradio_chat,
    # type="messages",   ← quitar esto en Gradio 6
    title="...",
    ...
)
```

### 7.2. Mismo modelo de embeddings para chunks y pregunta

Si usas modelos distintos para vectorizar chunks (fase 4) y pregunta (fase 6), los espacios vectoriales no son comparables y FAISS devuelve ruido sin avisar de error. **No es un error que detecte un linter** — es un fallo silencioso en la calidad del retrieval.

### 7.3. Olvidarse del "no sé" en el system prompt

Sin la línea *"Si el contexto no contiene información suficiente, dilo claramente"*, el LLM alucina cuando no encuentra la respuesta. Es la diferencia entre un asistente útil y uno que se inventa cosas.

### 7.4. Guardar en el historial la pregunta condensada en lugar de la original

Síntoma: la conversación se ve rara en la UI (texto reformulado en lugar de lo que escribió el usuario), y a partir de varios turnos las respuestas empiezan a degradarse porque el historial acumula reformulaciones sobre reformulaciones.

**Regla:** `history` guarda siempre la pregunta **original** del usuario.

### 7.5. Reconstruir el FAISS en cada consulta

Si el pipeline está mal diseñado y construye el FAISS dentro de `answer_question` en vez de en el constructor, cada respuesta tarda 5-10 segundos en preparación. **Las fases 2-5 deben ejecutarse una sola vez al arrancar.** Las fases 6-8 son las únicas por consulta.

---

## Resumen mental — para llevar

- **RAG = Retrieval + Augmented + Generation.** Buscar fragmentos relevantes y pegarlos al prompt para que el LLM no aluciné.
- **El pipeline es una cadena de transformaciones.** Cada fase recibe una cosa y produce otra.
- **Construcción (2-5) se hace una vez. Consulta (6-8) se hace por pregunta.** Esa separación es lo que hace la app rápida.
- **Embeddings contextuales (no estáticos)** son los que se usan en RAG moderno. La familia Sentence-Transformers es el estándar.
- **Mismo modelo de embeddings** para chunks y para la pregunta. Si no, ruido silencioso.
- **FAISS recupera, el LLM responde.** FAISS no genera lenguaje. El usuario nunca ve los chunks crudos.
- **El contexto se aplana a `str`** porque al LLM le rinde mejor texto plano que JSON, aunque ambos los entienda.
- **Roles `system` / `user` / `assistant`** estructuran el prompt. `assistant` solo aparece cuando hay turnos previos.
- **Anti-alucinación:** instruir explícitamente al modelo para que diga "no lo sé".
- **RAG conversacional añade dos historiales:** uno para el LLM (genera con memoria) y otro reformulado (`condense_question`) que solo alimenta a FAISS y se descarta.
- **El historial guarda siempre la pregunta original**, nunca la condensada.
