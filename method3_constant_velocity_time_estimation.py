import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Método 3 - Estimación de la posición esperada por aproximación temporal (velocidad constante)

Es un EVALUADOR PURO: no corrige nada, no genera ninguna posición corregida — a diferencia de los
métodos 1 y 1 v2 (que corrigen la estimación bruta proyectándola sobre la recta, ajustando
libremente su posición a lo largo de ella, y por tanto solo detectan el error perpendicular a la
recta), este método calcula la posición donde "debería" estar el objeto en cada instante,
asumiendo que se mueve a velocidad constante en línea recta entre el primer y el último punto de
la secuencia. Como esa posición esperada no se ajusta a los datos (se deriva solo del tiempo
transcurrido), el error resultante frente a la posición calculada recoge las dos componentes:
a lo largo de la línea y perpendicular a ella. Es la misma idea que ya usa `samples.csv` para
calcular `realX`/`realY` (ver README), solo que aquí no hay una interpolada de partida: los
extremos y los instantes se sacan del propio fichero de entrada.
El proceso es el siguiente:
1. Carga la secuencia de posiciones calculadas (timestamp, rawX, rawY) desde un CSV.
2. Toma el primer y el último punto de la secuencia como extremos reales (P0, Pn), y sus
   timestamps como instante de inicio/fin del recorrido.
3. Para cada fila calcula alpha = (timestamp - t0) / (t1 - t0) y la posición esperada
   P0 + alpha * (Pn - P0).
4. Calcula el error (errorX, errorY, euclideanError) entre la posición calculada y la esperada.

Uso: python method3_constant_velocity_time_estimation.py [--n-rows N]
  --n-rows N   Procesa solo las N primeras filas de cada CSV de input/ (por defecto, todas).
'''

csv_separator = ','
csv_decimal = '.'

parser = argparse.ArgumentParser(description='Método 3 - estimación de posición esperada por aproximación temporal')
parser.add_argument('--n-rows', type=int, default=None,
help='Procesa solo las N primeras filas de cada CSV de input/ (por defecto, todas)')
args = parser.parse_args()
n_rows = args.n_rows


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


def save_positions(input_file_name, output_file_name, p_expected, error, clean_if_exists=False,
n_rows=None):
    import pandas as pd

    csv_path = os.path.join(os.path.dirname(__file__), 'output', output_file_name)

    # Leemos el archivo de entrada original para conservar todas las columnas
    input_csv_path = os.path.join(os.path.dirname(__file__), 'input', input_file_name)
    df_input = pd.read_csv(input_csv_path, sep=csv_separator, decimal=csv_decimal)
    if n_rows is not None:
        df_input = df_input.iloc[:n_rows]
    df = df_input.copy()
    df['expectedX'] = p_expected[:, 0]
    df['expectedY'] = p_expected[:, 1]
    df['errorX'] = error[:, 0]
    df['errorY'] = error[:, 1]
    df['euclideanError'] = np.sqrt(error[:, 0]**2 + error[:, 1]**2)

    if os.path.exists(csv_path) and not clean_if_exists:
        df_existing = pd.read_csv(csv_path, sep=csv_separator, decimal=csv_decimal)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_path, sep=csv_separator, decimal=csv_decimal, index=False)


input_dir = os.path.join(os.path.dirname(__file__), 'input')
files_name = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]

for file_name in files_name:

    P_est_full, timestamps_full = load_positions(input_file_name=file_name)

    # El modelo (extremos P0/Pn y ventana temporal t0/t1) se define SIEMPRE con el fichero
    # completo, para que no cambie según cuántas filas se evalúen. --n-rows solo recorta qué
    # filas se calculan/reportan/guardan, no el modelo contra el que se comparan.
    P0 = P_est_full[0]
    Pn = P_est_full[-1]
    t0 = timestamps_full[0]
    t1 = timestamps_full[-1]

    if n_rows is not None:
        P_est = P_est_full[:n_rows]
        timestamps = timestamps_full[:n_rows]
    else:
        P_est = P_est_full
        timestamps = timestamps_full
    n = P_est.shape[0]

    # Posición esperada por interpolación lineal en el tiempo (velocidad constante)
    alpha = (timestamps - t0) / (t1 - t0)
    P_expected = P0 + np.outer(alpha, (Pn - P0))

    error = P_est - P_expected
    euclidean_error = np.linalg.norm(error, axis=1)

    output_file_name = os.path.splitext(file_name)[0] + '_temporal.csv'
    save_positions(input_file_name=file_name, output_file_name=output_file_name,
    p_expected=P_expected, error=error, clean_if_exists=True, n_rows=n_rows)

    print(f"{file_name}: n={n}, error euclídeo medio={euclidean_error.mean():.4f}, "
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
    plt.plot(P_expected[:, 0], P_expected[:, 1], color='orange', marker='o', markersize=3,
    linewidth=0.5, label='Posición esperada (aprox. temporal, velocidad constante)')

    for i in range(n):
        plt.plot([P_est[i, 0], P_expected[i, 0]], [P_est[i, 1], P_expected[i, 1]],
        linestyle=':', color='gray', linewidth=0.6)

    plt.title(f'Método 3 — {file_name}\nComparativa: calculado vs. esperado por aproximación temporal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
