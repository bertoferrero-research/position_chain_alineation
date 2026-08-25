# Alineación de trayectorias ArUco — corrección de error de posición

## Qué problema resuelve esto

Se grabó un objeto moviéndose por un carril/trayectoria **recta** y a **velocidad constante
(o casi)**. Un sistema basado en marcadores ArUco estima, frame a frame, la posición del objeto
(`rawX`, `rawY`) a partir de los marcadores visibles en cada instante. Esa estimación tiene ruido:
error de detección del marcador, error de triangulación, jitter, etc.

Como *sabemos* que la trayectoria real es una línea recta entre un punto de inicio y un punto de
fin conocidos (`realX`, `realY` en los CSV, o los extremos fijados a mano en `method1_v2.py`), se
puede usar esa información como "verdad de terreno" para:

1. Corregir/proyectar las estimaciones ruidosas sobre la recta real → obtener una posición
   corregida más fiable que la estimación bruta.
2. Medir cuánto se desvía la estimación ArUco de la posición real (el objetivo final de la tesis:
   cuantificar el error del sistema de posicionamiento).

Los tres scripts (`method1.py`, `method1_v2.py`, `method2.py`) son **tres formas distintas de
hacer esa corrección/alineación**, no tres pasos de un pipeline. Son alternativas que se comparan
entre sí.

## Idea común a los tres métodos

- Tenemos una nube de puntos estimados por ArUco: `P_est` (uno por frame).
- Tenemos una recta real definida por dos extremos: `P0_real` (inicio) y `Pn_real` (fin).
- Como el movimiento es rectilíneo, cualquier posición real intermedia se puede escribir como
  `P0_real + s * dirección_unitaria`, donde `s` es un escalar (distancia recorrida a lo largo de
  la recta, entre `0` y `length`).
- El problema se reduce a: **para cada punto estimado, ¿qué valor de `s` (posición sobre la
  recta) explica mejor esa estimación?**

Donde difieren los métodos es en cómo se calcula ese `s` para cada punto, y en cómo se
correlaciona cada estimación con su "hueco" en el tiempo/espacio real.

## Método 1 (`method1.py`) — mínimos cuadrados, extremos fijos

- Fuente de datos: `samples.csv` (en la raíz del repo), separador `;`, decimales con `,`.
- El CSV contiene muchas configuraciones distintas del sistema de captura, identificadas por
  `sampleSpaceMillis` (cada cuántos ms se muestrea) y `multipleMarkersBehaviour` (cómo se combina
  la posición cuando hay varios marcadores visibles a la vez: `CLOSEST`, `WEIGHTED_AVERAGE`,
  `AVERAGE`, `WEIGHTED_MEDIAN`, `MEDIAN`). El script recorre **todas las combinaciones** presentes.
- Para cada combinación:
  - Se toman `P0_real`/`Pn_real` como el primer y último `realX/realY` de ese subconjunto de filas.
  - Se optimiza (con `scipy.optimize.minimize`, método L-BFGS-B) el valor de `s` de cada punto
    **intermedio**, minimizando la suma de distancias al cuadrado entre el punto proyectado sobre
    la recta y el punto estimado.
  - **Los extremos (primer y último punto) se fijan** a `s=0` y `s=length` — no se optimizan,
    se asume que ahí la estimación coincide con el extremo real conocido.
- Resultado: `optimized_positions.csv`, con todas las combinaciones acumuladas (una fila por
  frame y combinación).
- Además dibuja, por cada combinación, un plot con: la recta real, los puntos reales
  interpolados, las estimaciones brutas, los puntos corregidos, y los marcadores ArUco
  (de `distribucion_markers_1_rev1.json`) usados en cada frame. **Cada plot bloquea la ejecución**
  hasta que se cierra la ventana.

## Método 1 v2 (`method1_v2.py`) — igual que el 1, pero genérico y sin fijar extremos

Es una evolución de `method1.py` pensada para **reutilizar el mismo método de mínimos cuadrados
con cualquier conjunto de datos nuevo**, sin tener que tocar el código cada vez.

La diferencia **no es solo de entrada/salida** — hay un cambio real en el ajuste matemático.
Diferencias clave respecto a `method1.py`, de más a menos importante:

- **No se fija ningún extremo (el cambio importante)**: `method1.py` fuerza `s=0` y `s=length`
  en el primer y último punto (los deja pegados al extremo real, sin optimizar). `method1_v2.py`
  optimiza el `s` de **todos** los puntos, incluidos el primero y el último, sin forzar nada.
  Esto significa que el resultado corregido puede diferir aunque le dieras el mismo `samples.csv`
  a ambos: `method1.py` asume que el primer/último frame capturado coincide exactamente con el
  inicio/fin real; `method1_v2.py` no hace esa suposición y deja que el ruido de esos frames
  también se corrija.
- **De dónde sale la recta real (`P_real`)**: en `method1.py` se lee de los datos (primer/último
  `realX/realY` de cada combinación filtrada). En `method1_v2.py` está **hardcodeada** como
  constante al principio del fichero (líneas ~21-24) — hay que **editarla a mano** cada vez que
  cambie la recta real del experimento.
- **Entrada/salida por lote**: en vez de un único `samples.csv` fijo, procesa **todos los CSV que
  haya en `input/`**, y escribe un CSV de salida por cada uno en `output/`, con el mismo nombre.
  Esas carpetas están en `.gitignore` (solo se versiona la carpeta, no su contenido) — hay que
  copiar ahí los CSV de la tanda de experimentos que se quiera procesar.
- **Dialecto de CSV distinto**: separador `,` y decimales con `.` (al revés que
  `method1.py`/`method2.py`, que usan `;` y `,`). Cuidado al mezclar ficheros de una fuente u
  otra.
- El CSV de salida conserva **todas las columnas originales** del CSV de entrada y añade
  `alineatedRealX` / `alineatedRealY` (nombre con un typo: "alineated" en vez de "aligned").

En resumen: usar `method1_v2.py` cuando se quiera aplicar el método de mínimos cuadrados a datos
nuevos sin tocar el script cada vez (solo hay que ajustar la constante `P_real` y poner los CSV en
`input/`).

## Método 2 (`method2.py`) — alineación de secuencias (Needleman-Wunsch)

Enfoque distinto: en vez de proyectar cada estimación sobre la recta por mínimos cuadrados, se
plantea como un problema de **alineación de secuencias** (el mismo algoritmo que se usa para
alinear cadenas de ADN).

- Fuente de datos: igual que `method1.py` (`samples.csv`, `;` / `,`).
- Se fija una **secuencia de referencia**: la trayectoria real interpolada con
  `sampleSpaceMillis=0` y el primer comportamiento de la lista (`WEIGHTED_MEDIAN` tal y como está
  configurado ahora). Esa referencia representa "dónde debería estar el objeto en cada instante
  real, muestreado lo más fino posible".
- Para cada combinación (`sampleSpaceMillis`, `multipleMarkersBehaviour`), se alinean las
  estimaciones ArUco de esa combinación contra la referencia real, usando Needleman-Wunsch:
  - La "similitud" entre un punto real y uno estimado es la distancia euclídea en negativo (cuanto
    más cerca, mejor puntuación).
  - El algoritmo encuentra la correspondencia global óptima entre ambas secuencias, permitiendo
    "huecos" (gaps) cuando un punto no tiene pareja razonable en la otra secuencia (p. ej. si el
    muestreo estimado es más disperso que la referencia).
- Lo que se guarda en `optimized_positions_method2.csv` **no es la estimación corregida**, sino
  **el punto de la trayectoria real de referencia que quedó emparejado con cada estimación**. Es
  decir, este método resincroniza el "reloj" de la referencia real con el de las estimaciones,
  en vez de mover las estimaciones hacia la recta.

Esto lo hace conceptualmente distinto de los métodos 1: mientras que el método 1 asume que cada
frame ya tiene un `realX/realY` de referencia y solo corrige su posición sobre la recta, el
método 2 no asume una correspondencia 1-a-1 previa entre frames estimados y frames reales, y la
construye explícitamente mediante alineación.

## Cuál usar / cómo interpretarlos juntos

- **Método 1**: bueno cuando confías en que el frame inicial y final de la estimación ArUco
  corresponden exactamente al inicio/fin real del recorrido (los fija). Da la corrección más
  "ajustada" en esos extremos.
- **Método 1 v2**: igual matemáticamente, pero sin esa suposición en los extremos, y pensado para
  correr sobre lotes de experimentos nuevos sin tocar el script (parametrizado por carpeta y por
  la constante `P_real`).
- **Método 2**: útil cuando el muestreo temporal de las estimaciones no es fiable o está
  desalineado respecto al tiempo real (p. ej. frames perdidos, timestamps con jitter), porque no
  asume una correspondencia directa índice-a-índice, sino que la resuelve por similitud espacial.

Para la tesis, comparar los tres da tres estimaciones distintas del error de posicionamiento del
sistema ArUco bajo distintos supuestos de correspondencia temporal — el objetivo no es que
"ganen" uno, sino entender cómo cambia el error medido según el supuesto metodológico.

## Cómo ejecutar

Dependencias: `numpy`, `scipy`, `pandas`, `matplotlib` (no hay `requirements.txt` todavía).

```bash
python method1.py       # lee samples.csv, escribe optimized_positions.csv
python method1_v2.py    # lee input/*.csv, escribe output/<mismo nombre>.csv
python method2.py       # lee samples.csv, escribe optimized_positions_method2.csv
```

Cada combinación de parámetros abre una ventana de matplotlib que **bloquea la ejecución** hasta
que se cierra. Si solo interesa el CSV de salida y no las gráficas, comentar las llamadas a
`plt.show()` (o las secciones de plotting) antes de lanzar corridas largas con muchas
combinaciones.

## Columnas necesarias en los CSV de entrada

**`method1.py` / `method2.py`** (formato `samples.csv`: separador `;`, decimal `,`):

| Columna | Para qué se usa |
|---|---|
| `sampleSpaceMillis` | agrupar/filtrar las combinaciones a procesar |
| `multipleMarkersBehaviour` | agrupar/filtrar las combinaciones a procesar |
| `rawX`, `rawY` | posición estimada por ArUco (`P_est`) |
| `realX`, `realY` | posición real interpolada (`P_real`); en `method1.py` de aquí salen también los extremos de la recta |
| `timestamp` | se conserva en el CSV de salida (en `method2.py` además identifica cada fila alineada) |
| `markers_info` | de aquí se extraen los `markerId=N` para dibujar los marcadores en el plot |

**`method1_v2.py`** (formato `input/*.csv`: separador `,`, decimal `.`):

| Columna | Para qué se usa |
|---|---|
| `rawX`, `rawY` | posición estimada (`P_est`) |
| `markers_info` | igual que arriba, para el plot |

Solo hacen falta esas dos — el resto de columnas del CSV de entrada no se leen, simplemente se
copian tal cual al output. Dos cosas a tener en cuenta:

- La recta real (`P_real`) **no** se lee del CSV en este script — es la constante hardcodeada al
  principio del fichero (líneas ~21-24). No hace falta ninguna columna `realX`/`realY` en el
  input para que corra.
- `load_positions()` acepta un parámetro `timestamp_column` (por defecto `'timestamp'`), pero
  **no se usa en el cuerpo de la función** — es un parámetro muerto. No hace falta esa columna
  para que el script funcione, aunque probablemente debería usarse si en algún momento se quiere
  que el output lleve timestamp (ahora mismo no lo lleva).
- Los nombres de columna de `rawX`/`rawY`/`markers_info` son parametrizables en
  `load_positions()`, pero el script los llama sin overrides (`load_positions(input_file_name=file_name)`),
  así que tal y como está hoy, esos son los nombres exactos que tiene que tener el CSV.

## Ficheros de datos

| Fichero | En git | Usado por | Formato |
|---|---|---|---|
| `samples.csv` | sí | method1, method2 | `;` sep, `,` decimal |
| `optimized_positions.csv` | sí | salida de method1 | `;` sep, `,` decimal |
| `optimized_positions_method2.csv` | sí | salida de method2 | `;` sep, `,` decimal |
| `input/*.csv` | **no** (solo la carpeta) | entrada de method1_v2 | `,` sep, `.` decimal |
| `output/*.csv` | **no** (solo la carpeta) | salida de method1_v2 | `,` sep, `.` decimal |
| `distribucion_markers_1_rev1.json` | sí | los tres, para dibujar marcadores | posición/rotación por `id` de marcador |

La columna `markers_info` de los CSV de origen trae el texto tal cual lo generó el sistema
(algo como `PositionFromMarker(markerId=5, x=..., y=..., z=..., distance=...)`, posiblemente
varios por fila); los tres scripts extraen solo los `markerId` con una regex
(`markerId=(\d+)`), no parsean el resto de campos de esa cadena.
