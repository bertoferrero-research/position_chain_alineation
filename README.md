# Alineación de trayectorias ArUco — corrección de error de posición

## Qué problema resuelve esto

Se grabó un objeto moviéndose por un carril/trayectoria **recta** y a **velocidad constante
(o casi)**. Un sistema basado en marcadores ArUco estima, frame a frame, la posición del objeto
(`rawX`, `rawY`) a partir de los marcadores visibles en cada instante. Esa estimación tiene ruido:
error de detección del marcador, error de triangulación, jitter, etc.

Como *sabemos* que la trayectoria real es una línea recta entre un punto de inicio y un punto de
fin conocidos (`realX`, `realY` en los CSV, o los extremos fijados a mano en `method1_v2_least_squares_batch.py`), se
puede usar esa información como "verdad de terreno" para:

1. Corregir/proyectar las estimaciones ruidosas sobre la recta real → obtener una posición
   corregida más fiable que la estimación bruta.
2. Medir cuánto se desvía la estimación ArUco de la posición real (el objetivo final de la tesis:
   cuantificar el error del sistema de posicionamiento).

Los tres scripts (`method1_least_squares_fixed_endpoints.py`, `method1_v2_least_squares_batch.py`, `method2_needleman_wunsch_alignment.py`) son **tres formas distintas de
hacer esa corrección/alineación**, no tres pasos de un pipeline. Son alternativas que se comparan
entre sí.

## De dónde sale la posición "real" (`realX`/`realY`) si no se puede medir en cada frame

Esto puede confundir a primera vista: si el problema es precisamente que no se puede medir la
posición real con precisión frame a frame (por eso hace falta ArUco), ¿cómo puede el CSV traer ya
una columna `realX`/`realY`?

La respuesta es que **no se mide, se calcula** — se aprovecha que el movimiento está controlado
(línea recta, velocidad ~constante) para derivar la posición esperada en cualquier instante a
partir de solo cuatro números que sí son fáciles de medir **una vez por pasada**: los dos extremos
físicos del recorrido. `samples.csv` trae explícitas las columnas que hacen esa cuenta:
`xi, yi` (extremo inicial), `xf, yf` (extremo final), `ti, tf` (instante de inicio/fin del
recorrido) y `alpha`. La relación (comprobada contra los datos) es una interpolación lineal simple:

```
alpha = (timestamp - ti) / (tf - ti)
realX = xi + alpha * (xf - xi)
realY = yi + alpha * (yf - yi)
```

Es decir: `realX`/`realY` es una posición **teórica**, calculada bajo el supuesto de velocidad
constante entre dos puntos conocidos — no una medición independiente de la posición del objeto.
Su fiabilidad depende de que ese supuesto se cumpla razonablemente bien (si el objeto acelera o
frena, esta "posición real" ya carga su propio error, que no es el error de ArUco).

Esa cuenta **no la hace nada de este repo** — llega ya calculada dentro de `samples.csv`. Ese CSV
es la salida de un sistema externo (la app/pipeline que corre ArUco + Kalman durante el
experimento, a juzgar por columnas como `RT_ransac_threshold`, `kalmanQ`, `kalmanR`), que conoce
de antemano los extremos y los instantes de cada pasada y calcula `alpha`/`realX`/`realY` al
escribir cada fila. `method1_least_squares_fixed_endpoints.py`, `method1_v2_least_squares_batch.py` y `method2_needleman_wunsch_alignment.py` **solo consumen** esa columna;
ninguno la genera ni la recalcula.

**Importante:** `method1_least_squares_fixed_endpoints.py` y `method1_v2_least_squares_batch.py` *no* usan esta interpolación por tiempo para los
puntos intermedios. Solo toman los dos extremos (de `realX`/`realY`, o la constante `P_real`
hardcodeada en `method1_v2_least_squares_batch.py`) para definir la recta, y luego cada punto intermedio se recalcula
por mínimos cuadrados ajustándolo a la estimación ArUco bruta — no al valor de `realX`/`realY` de
esa fila. Es decir, esos métodos solo asumen "está en la recta", no "se movió a velocidad
constante"; el supuesto de velocidad constante queda reservado a los dos extremos (que ahí sí
coinciden, por construcción, con `alpha=0` y `alpha=1`).

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

## Método 1 (`method1_least_squares_fixed_endpoints.py`) — mínimos cuadrados, extremos fijos

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

## Método 1 v2 (`method1_v2_least_squares_batch.py`) — igual que el 1, pero genérico y sin fijar extremos

Es una evolución de `method1_least_squares_fixed_endpoints.py` pensada para **reutilizar el mismo método de mínimos cuadrados
con cualquier conjunto de datos nuevo**, sin tener que tocar el código cada vez.

La diferencia **no es solo de entrada/salida** — hay un cambio real en el ajuste matemático.
Diferencias clave respecto a `method1_least_squares_fixed_endpoints.py`, de más a menos importante:

- **No se fija ningún extremo (el cambio importante)**: `method1_least_squares_fixed_endpoints.py` fuerza `s=0` y `s=length`
  en el primer y último punto (los deja pegados al extremo real, sin optimizar). `method1_v2_least_squares_batch.py`
  optimiza el `s` de **todos** los puntos, incluidos el primero y el último, sin forzar nada.
  Esto significa que el resultado corregido puede diferir aunque le dieras el mismo `samples.csv`
  a ambos: `method1_least_squares_fixed_endpoints.py` asume que el primer/último frame capturado coincide exactamente con el
  inicio/fin real; `method1_v2_least_squares_batch.py` no hace esa suposición y deja que el ruido de esos frames
  también se corrija.
- **De dónde sale la recta real (`P_real`)**: en `method1_least_squares_fixed_endpoints.py` se lee de los datos (primer/último
  `realX/realY` de cada combinación filtrada). En `method1_v2_least_squares_batch.py` está **hardcodeada** como
  constante al principio del fichero (líneas ~21-24) — hay que **editarla a mano** cada vez que
  cambie la recta real del experimento.
- **Entrada/salida por lote**: en vez de un único `samples.csv` fijo, procesa **todos los CSV que
  haya en `input/`**, y escribe un CSV de salida por cada uno en `output/`, con el mismo nombre.
  Esas carpetas están en `.gitignore` (solo se versiona la carpeta, no su contenido) — hay que
  copiar ahí los CSV de la tanda de experimentos que se quiera procesar.
- **Dialecto de CSV distinto**: separador `,` y decimales con `.` (al revés que
  `method1_least_squares_fixed_endpoints.py`/`method2_needleman_wunsch_alignment.py`, que usan `;` y `,`). Cuidado al mezclar ficheros de una fuente u
  otra.
- El CSV de salida conserva **todas las columnas originales** del CSV de entrada y añade
  `alineatedRealX` / `alineatedRealY` (nombre con un typo: "alineated" en vez de "aligned"), más
  `errorX`, `errorY` y `euclideanError` — la distancia entre la estimación bruta
  (`rawX`/`rawY`) y su proyección sobre la recta real. **Es el error perpendicular a la línea
  (cross-track)**, no el error total: como no se usa información de tiempo (a diferencia de la
  interpolación por `alpha`/`ti`/`tf` de `samples.csv`), no captura si la estimación iba
  adelantada o atrasada a lo largo de la recta respecto a donde debería estar según velocidad
  constante — solo cuánto se aleja de la trayectoria recta conocida.

En resumen: usar `method1_v2_least_squares_batch.py` cuando se quiera aplicar el método de mínimos cuadrados a datos
nuevos sin tocar el script cada vez (solo hay que ajustar la constante `P_real` y poner los CSV en
`input/`).

## Método 2 (`method2_needleman_wunsch_alignment.py`) — alineación de secuencias (Needleman-Wunsch)

Enfoque distinto: en vez de proyectar cada estimación sobre la recta por mínimos cuadrados, se
plantea como un problema de **alineación de secuencias** (el mismo algoritmo que se usa para
alinear cadenas de ADN).

- Fuente de datos: igual que `method1_least_squares_fixed_endpoints.py` (`samples.csv`, `;` / `,`).
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

## Método 3 (`method3_constant_velocity_time_estimation.py`) — posición esperada por aproximación temporal

Este método es distinto de los anteriores en un punto clave: **no ajusta nada a los datos**. Los
métodos 1 y 1 v2 encuentran, para cada punto, la posición sobre la línea que mejor explica la
estimación bruta — es decir, dejan libre la componente a lo largo de la línea (`u`) y solo pueden
detectar error en la componente perpendicular (`v`). Este método, en cambio, calcula la posición
donde el objeto **debería** estar en cada instante, sin mirar en absoluto la estimación de esa
fila, así que el error resultante recoge las dos componentes (`u` y `v`).

- Fuente de datos: procesa **todos los CSV de `input/`**, igual que `method1_v2_least_squares_batch.py`
  (separador `,`, decimal `.`). No necesita `markers_info` — solo `timestamp`, `rawX`, `rawY`.
- Toma el **primer y el último punto del fichero completo** como los dos extremos reales (`P0`,
  `Pn`) y sus timestamps como instante de inicio/fin del recorrido (`t0`, `t1`) — no hay una
  constante `P_real` que editar a mano, se deriva de los propios datos de cada fichero.
- `--n-rows N` (parámetro de línea de comandos) limita **solo qué filas se calculan, informan y
  guardan** — no toca `P0`/`Pn`/`t0`/`t1`, que siempre se derivan del fichero completo. Esto es
  importante: si `P0`/`Pn`/`t0`/`t1` se recalcularan con cada subconjunto, el error de una fila
  fija cambiaría según `--n-rows` sin que eso signifique nada sobre la trayectoria real — sería
  solo un artefacto de la fórmula (`alpha` de esa fila tiende a 0 cuanto más se amplía `t1`, así
  que su posición esperada converge hacia `P0`). Al fijar el modelo al fichero completo, el error
  de cada fila es siempre el mismo sin importar con cuántas filas se ejecute el script.
- Para cada fila calcula `alpha = (timestamp - t0) / (t1 - t0)` y la posición esperada
  `P0 + alpha * (Pn - P0)`, asumiendo velocidad constante — la misma fórmula que ya vimos que usa
  `samples.csv` para calcular `realX`/`realY` internamente (ver la sección de arriba), solo que
  aquí se calcula explícitamente en el script en vez de venir precalculada en el CSV de origen.
- Guarda `expectedX`, `expectedY`, `errorX`, `errorY` y `euclideanError` en
  `output/<nombre>_temporal.csv` (con el sufijo `_temporal` para no chocar con la salida de
  `method1_v2_least_squares_batch.py` sobre el mismo fichero de entrada).
- Imprime por consola un resumen (media, mediana, desviación típica, máximo) del error euclídeo.
- Dibuja un plot con la línea real, la trayectoria calculada y la esperada, y una línea punteada
  por cada punto uniendo ambas (el vector de error completo).

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
- **Método 3**: el único que mide el error **total** (a lo largo de la línea + perpendicular a
  ella), porque no ajusta nada — solo asume velocidad constante entre el primer y el último punto
  de la secuencia. Útil cuando no tienes (o no confías en) una columna `realX`/`realY` ya calculada,
  y quieres saber si el sistema, además de mantenerse sobre la línea, va también sincronizado en
  el tiempo con el movimiento real.

Para la tesis, comparar los cuatro (los tres métodos + la columna `errorX`/`errorY` ya presente en
`samples.csv`) da estimaciones distintas del error de posicionamiento bajo distintos supuestos de
correspondencia temporal — el objetivo no es que "gane" uno, sino entender cómo cambia el error
medido según el supuesto metodológico. En concreto, los métodos 1/1 v2 solo pueden medir el error
perpendicular a la línea (`v`); si además quieres el error a lo largo de ella (`u`), necesitas el
método 3 o la columna `realX`/`realY` ya calculada por tiempo.

## Cómo ejecutar

Dependencias: `numpy`, `scipy`, `pandas`, `matplotlib` (no hay `requirements.txt` todavía).

```bash
python method1_least_squares_fixed_endpoints.py       # lee samples.csv, escribe optimized_positions.csv
python method1_v2_least_squares_batch.py               # lee input/*.csv, escribe output/<mismo nombre>.csv
python method2_needleman_wunsch_alignment.py           # lee samples.csv, escribe optimized_positions_method2.csv
python method3_constant_velocity_time_estimation.py     # lee input/*.csv, escribe output/<nombre>_temporal.csv
python method3_constant_velocity_time_estimation.py --n-rows 100   # solo evalúa/guarda las 100 primeras filas (el modelo P0/Pn/t0/t1 sigue siendo el del fichero completo)
```

Cada combinación de parámetros abre una ventana de matplotlib que **bloquea la ejecución** hasta
que se cierra. Si solo interesa el CSV de salida y no las gráficas, comentar las llamadas a
`plt.show()` (o las secciones de plotting) antes de lanzar corridas largas con muchas
combinaciones.

## Columnas necesarias en los CSV de entrada

**`method1_least_squares_fixed_endpoints.py` / `method2_needleman_wunsch_alignment.py`** (formato `samples.csv`: separador `;`, decimal `,`):

| Columna | Para qué se usa |
|---|---|
| `sampleSpaceMillis` | agrupar/filtrar las combinaciones a procesar |
| `multipleMarkersBehaviour` | agrupar/filtrar las combinaciones a procesar |
| `rawX`, `rawY` | posición estimada por ArUco (`P_est`) |
| `realX`, `realY` | posición real interpolada (`P_real`); en `method1_least_squares_fixed_endpoints.py` de aquí salen también los extremos de la recta |
| `timestamp` | se conserva en el CSV de salida (en `method2_needleman_wunsch_alignment.py` además identifica cada fila alineada) |
| `markers_info` | de aquí se extraen los `markerId=N` para dibujar los marcadores en el plot |

**`method1_v2_least_squares_batch.py`** (formato `input/*.csv`: separador `,`, decimal `.`):

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

**`method3_constant_velocity_time_estimation.py`** (formato `input/*.csv`: separador `,`, decimal `.`):

| Columna | Para qué se usa |
|---|---|
| `timestamp` | define `t0`/`t1` y el `alpha` de interpolación por fila |
| `rawX`, `rawY` | posición calculada (`P_est`); su primera y última fila son además `P0`/`Pn` |

Solo estas tres — no necesita `markers_info` (a diferencia de los métodos 1/1 v2, este script no
dibuja marcadores ArUco en absoluto).

## Ficheros de datos

| Fichero | En git | Usado por | Formato |
|---|---|---|---|
| `samples.csv` | sí | method1, method2 | `;` sep, `,` decimal |
| `optimized_positions.csv` | sí | salida de method1 | `;` sep, `,` decimal |
| `optimized_positions_method2.csv` | sí | salida de method2 | `;` sep, `,` decimal |
| `input/*.csv` | **no** (solo la carpeta) | entrada de method1_v2 y method3 | `,` sep, `.` decimal |
| `output/*.csv` | **no** (solo la carpeta) | salida de method1_v2 (mismo nombre) y method3 (sufijo `_temporal`) | `,` sep, `.` decimal |
| `distribucion_markers_1_rev1.json` | sí | method1 y method1_v2, para dibujar marcadores | posición/rotación por `id` de marcador |

La columna `markers_info` de los CSV de origen trae el texto tal cual lo generó el sistema
(algo como `PositionFromMarker(markerId=5, x=..., y=..., z=..., distance=...)`, posiblemente
varios por fila); method1, method1_v2 y method2 extraen solo los `markerId` con una regex
(`markerId=(\d+)`), no parsean el resto de campos de esa cadena. `method3` no usa esta columna.
