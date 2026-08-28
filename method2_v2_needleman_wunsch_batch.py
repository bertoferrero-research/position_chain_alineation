import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Método 2 v2 - Alineación de secuencias (Needleman-Wunsch) contra una referencia generada internamente

Es un EVALUADOR: no corrige nada, no genera ninguna posición corregida. Generaliza `method2` para
ficheros de tipo `input/*.csv` (una sola secuencia timestamp+posición calculada, sin columna de
referencia interpolada como `realX`/`realY`) — en vez de leer la referencia real de `samples.csv`,
la genera aquí mismo: un conjunto DENSO de puntos sobre la recta entre el primer y el último punto
del fichero (`P0`, `Pn`), muestreados uniformemente en el tiempo entre `t0` y `t1` (mismo modelo
que `method3_constant_velocity_time_estimation.py`, pero mucho más fino — `--ref-points` puntos en
vez de uno por fila de entrada).

OJO — esto NO responde a "¿dónde debería estar el sujeto en el instante exacto de esta captura?".
Empareja por PARECIDO ESPACIAL (distancia euclídea), no por timestamp: el punto de referencia
elegido para cada fila no tiene por qué corresponder al mismo instante que esa fila. Si lo que
quieres es esa pregunta de tiempo, usa `method3_constant_velocity_time_estimation.py` directamente
— es más simple y no necesita ninguna búsqueda. Este script solo tiene sentido si específicamente
no te fías de que el timestamp de cada fila sea fiable (frames perdidos, desincronización). Ver
README.md, sección "La pregunta clave: ¿qué tipo de punto de comparación busca cada método?".

Por qué no basta con `method1_v2` ni con `method3` para esto:
- `method1_v2_least_squares_batch.py` proyecta cada punto LIBREMENTE sobre una recta continua, así
  que solo puede medir el error perpendicular a la línea — la componente a lo largo de ella se
  absorbe siempre en el ajuste, sin ninguna restricción de orden temporal entre puntos.
- `method3_constant_velocity_time_estimation.py` sí mide el error en las dos componentes, pero
  compara fila a fila por timestamp — es frágil si la secuencia calculada tiene saltos, reordenes
  o algún timestamp puntual desincronizado.
Aquí, al alinear contra un conjunto DISCRETO de puntos (no una recta continua con libertad total),
sobrevive error en las dos componentes; y al ser una alineación GLOBAL que preserva el orden (como
el `method2` original), es robusto frente a saltos/reordenes en la secuencia calculada.

El proceso es el siguiente:
1. Carga la secuencia calculada (timestamp, rawX, rawY) de cada CSV de input/.
2. Toma el primer y el último punto como extremos reales (P0, Pn) y sus timestamps como instante
   de inicio/fin (t0, t1).
3. Genera una referencia densa de `--ref-points` puntos sobre la recta P0-Pn, muestreados
   uniformemente en el tiempo entre t0 y t1.
4. Alinea la secuencia calculada contra esa referencia densa con Needleman-Wunsch (similaridad =
   distancia euclídea en negativo), permitiendo huecos donde no hay pareja razonable.
5. Guarda, para cada fila con pareja, sus columnas originales + el punto de referencia emparejado
   (`matchedRealX/Y`) y el error entre ambos (`errorX/Y`, `euclideanError`) en
   output/<nombre>_nw.csv.

Columnas necesarias en cada CSV de input/ (separador ',', decimal '.'):
  - timestamp   -> define t0/t1 y el instante de cada fila para generar la referencia densa
  - rawX, rawY  -> posición calculada (P_est); su primera y última fila son además P0/Pn
No hace falta markers_info (no se dibujan marcadores). Los nombres de columna son parametrizables
en load_positions(), pero el script los llama sin overrides, así que estos son los nombres exactos
que tiene que tener el CSV tal y como está hoy.

Uso: python method2_v2_needleman_wunsch_batch.py [--ref-points N]
  --ref-points N   Densidad de la referencia sintética (por defecto, 2000).
'''

csv_separator = ','
csv_decimal = '.'

parser = argparse.ArgumentParser(description='Método 2 v2 - alineación Needleman-Wunsch contra una referencia generada internamente')
parser.add_argument('--ref-points', type=int, default=2000,
help='Densidad (número de puntos) de la referencia sintética sobre la recta P0-Pn (por defecto, 2000)')
args = parser.parse_args()
ref_points_count = args.ref_points


def load_positions(input_file_name,
estimated_position_x_column='rawX',
estimated_position_y_column='rawY',
timestamp_column='timestamp'):

    import pandas as pd
    import sys

    csv_path = os.path.join(os.path.dirname(__file__), 'input', input_file_name)
    if not os.path.exists(csv_path):
        print(f"Error: El archivo {csv_path} no existe.")
        sys.exit(1)
    df = pd.read_csv(csv_path, sep=csv_separator, decimal=csv_decimal)

    p_est = df[[estimated_position_x_column, estimated_position_y_column]].values
    timestamps = df[timestamp_column].values

    return p_est, timestamps


def save_positions(input_file_name, output_file_name, matched_est_rows, p_ref_matched, error):
    import pandas as pd

    csv_path = os.path.join(os.path.dirname(__file__), 'output', output_file_name)

    # Leemos el archivo de entrada original para conservar sus columnas, solo para las filas
    # que encontraron pareja en la alineación.
    input_csv_path = os.path.join(os.path.dirname(__file__), 'input', input_file_name)
    df_input = pd.read_csv(input_csv_path, sep=csv_separator, decimal=csv_decimal)
    df = df_input.iloc[matched_est_rows].reset_index(drop=True)

    df['matchedRealX'] = p_ref_matched[:, 0]
    df['matchedRealY'] = p_ref_matched[:, 1]
    df['errorX'] = error[:, 0]
    df['errorY'] = error[:, 1]
    df['euclideanError'] = np.sqrt(error[:, 0]**2 + error[:, 1]**2)

    df.to_csv(csv_path, sep=csv_separator, decimal=csv_decimal, index=False)


def needleman_wunsch(seq1, seq2, match_score=1, gap_cost=-1, mismatch_cost=-1):
    """
    Algoritmo Needleman-Wunsch para alineación global de dos secuencias.
    seq1, seq2: listas de puntos (por ejemplo, posiciones [x, y])
    Devuelve los índices alineados de seq1 y seq2 (con None para gaps).
    """
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n+1, m+1))
    pointer = np.zeros((n+1, m+1), dtype=int)
    for i in range(1, n+1):
        score[i, 0] = gap_cost * i
    for j in range(1, m+1):
        score[0, j] = gap_cost * j
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Similaridad inversa a la distancia euclídea
            sim = -np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
            match = score[i-1, j-1] + sim
            delete = score[i-1, j] + gap_cost
            insert = score[i, j-1] + gap_cost
            score[i, j] = max(match, delete, insert)
            if score[i, j] == match:
                pointer[i, j] = 1  # diagonal
            elif score[i, j] == delete:
                pointer[i, j] = 2  # up
            else:
                pointer[i, j] = 3  # left
    # Backtracking
    i, j = n, m
    aligned_seq1 = []
    aligned_seq2 = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pointer[i, j] == 1:
            aligned_seq1.append(i-1)
            aligned_seq2.append(j-1)
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or pointer[i, j] == 2):
            aligned_seq1.append(i-1)
            aligned_seq2.append(None)
            i -= 1
        else:
            aligned_seq1.append(None)
            aligned_seq2.append(j-1)
            j -= 1
    return aligned_seq1[::-1], aligned_seq2[::-1]


input_dir = os.path.join(os.path.dirname(__file__), 'input')
files_name = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]

for file_name in files_name:

    P_est, timestamps = load_positions(input_file_name=file_name)
    n = P_est.shape[0]

    # Extremos y ventana temporal tomados de la propia secuencia (primer/último punto)
    P0 = P_est[0]
    Pn = P_est[-1]
    t0 = timestamps[0]
    t1 = timestamps[-1]

    # Referencia densa: N puntos sobre la recta P0-Pn, uniformes en tiempo entre t0 y t1
    alpha_ref = np.linspace(0.0, 1.0, ref_points_count)
    P_ref = P0 + np.outer(alpha_ref, (Pn - P0))

    # Needleman-Wunsch: alineamos la secuencia calculada contra la referencia densa
    aligned_ref_idx, aligned_est_idx = needleman_wunsch(P_ref, P_est)

    # Nos quedamos solo con las posiciones de la alineación donde ambos lados tienen pareja
    matched_est_rows = []
    matched_ref_points = []
    for ref_i, est_j in zip(aligned_ref_idx, aligned_est_idx):
        if ref_i is not None and est_j is not None:
            matched_est_rows.append(est_j)
            matched_ref_points.append(P_ref[ref_i])

    print(f"{file_name}: {len(matched_est_rows)}/{n} filas emparejadas")

    if matched_est_rows:
        P_est_matched = P_est[matched_est_rows]
        P_ref_matched = np.array(matched_ref_points)
        error = P_est_matched - P_ref_matched
        euclidean_error = np.linalg.norm(error, axis=1)

        output_file_name = os.path.splitext(file_name)[0] + '_nw.csv'
        save_positions(input_file_name=file_name, output_file_name=output_file_name,
        matched_est_rows=matched_est_rows, p_ref_matched=P_ref_matched, error=error)

        print(f"{file_name}: error euclídeo medio={euclidean_error.mean():.4f}, "
              f"mediana={np.median(euclidean_error):.4f}, std={euclidean_error.std():.4f}, "
              f"max={euclidean_error.max():.4f}")

        # Visualización
        plt.figure(figsize=(10, 6))

        plt.plot([P0[0], Pn[0]], [P0[1], Pn[1]], linestyle='--', color='gray', linewidth=1,
        label='Línea real (recta entre primer y último punto)')
        plt.scatter(*P0, color='green', s=80, label='Inicio')
        plt.scatter(*Pn, color='red', s=80, label='Fin')

        plt.plot(P_est[:, 0], P_est[:, 1], color='blue', marker='x', linewidth=0.5,
        label='Posición calculada')
        plt.scatter(P_ref_matched[:, 0], P_ref_matched[:, 1], color='orange', marker='o', s=20,
        label='Punto de referencia emparejado (NW)')

        for p_est, p_ref in zip(P_est_matched, P_ref_matched):
            plt.plot([p_est[0], p_ref[0]], [p_est[1], p_ref[1]],
            linestyle=':', color='gray', linewidth=0.6)

        plt.title(f'Método 2 v2 — {file_name}\nAlineación Needleman-Wunsch contra referencia densa ({ref_points_count} puntos)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
