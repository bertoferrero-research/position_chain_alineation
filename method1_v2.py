import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

#Método 1 - Ajustamos la posición estimada real empleando mínimos cuadrados
#Versión 2 - Definimos de manera separada los extremos reales


P_real = [
    [3.733, 0.691],  # Inicio real
    [0.693, 0.882],  # Final real
]
csv_separator = ','
csv_decimal = '.'

def load_positions(input_file_name, 
estimated_position_x_column='rawX', 
estimated_position_y_column='rawY', 
timestamp_column='timestamp',
marker_info_column='markers_info'):

    # Comenzamos cargando el csv con panda
    import pandas as pd
    import sys
    
    # Definimos la ruta del archivo CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'input', input_file_name)
    if not os.path.exists(csv_path):
        print(f"Error: El archivo {csv_path} no existe.")
        sys.exit(1)
    # Cargamos el CSV
    df = pd.read_csv(csv_path, sep=csv_separator, decimal=csv_decimal)

    # Obtenemos las posiciones estimadas y reales
    p_est = df[[estimated_position_x_column, estimated_position_y_column]].values

    # Extraer ids de marcadores de la columna markers_info
    import re
    def extract_ids(marker_info_str):
        # Busca markerId=NUMERO en la cadena y extrae los números
        return [int(m) for m in re.findall(r'markerId=(\d+)', str(marker_info_str))]
    marker_ids_list = df[marker_info_column].apply(extract_ids).tolist()

    return p_est, marker_ids_list

def save_positions(input_file_name, output_file_name, p_est_optimized, clean_if_exists = False):
    # Guardamos las posiciones optimizadas en un CSV
    import pandas as pd
    import os
    
    # Definimos la ruta del archivo CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'output', output_file_name)

    # Creamos un DataFrame con las posiciones optimizadas
    # Leemos el archivo de entrada original para conservar todas las columnas
    input_csv_path = os.path.join(os.path.dirname(__file__), 'input', input_file_name)
    df_input = pd.read_csv(input_csv_path, sep=csv_separator, decimal=csv_decimal)
    df = df_input.copy()
    df['alineatedRealX'] = p_est_optimized[:, 0]
    df['alineatedRealY'] = p_est_optimized[:, 1]

    # Si el archivo existe, añadimos; si no, lo creamos
    if os.path.exists(csv_path) and not clean_if_exists:
        df_existing = pd.read_csv(csv_path, sep=csv_separator, decimal=csv_decimal)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_path, sep=csv_separator, decimal=csv_decimal, index=False)

def load_marker_positions(json_path=None):
    """
    Carga el archivo distribucion_markers_1_rev1.json y devuelve un diccionario
    donde la clave es el id del marcador y el valor es una lista [x, y].
    """
    import json
    import os
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'distribucion_markers_1_rev1.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        markers = json.load(f)
    marker_dict = {m['id']: [m['position']['x'], m['position']['y']] for m in markers}
    return marker_dict


P_real = np.array(P_real)
markers_data = load_marker_positions()

input_dir = os.path.join(os.path.dirname(__file__), 'input')
files_name = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]

for file_name in files_name:

    P_est, marker_ids_list = load_positions(input_file_name=file_name)

    # Suponé que ya tenés esto:
    # P_est: (n, 2) interpolación inicial
    # P_real: (n, 2) posiciones reales
    n = P_est.shape[0]
    d = 2  # 2D

    # Paso 1: definir extremos reales
    P0_real = P_real[0]
    Pn_real = P_real[-1]

    # Paso 2: vector dirección unitario de la línea real
    direction = Pn_real - P0_real
    length = np.linalg.norm(direction)
    direction_unit = direction / length

    # Paso 3: inicializar escalares (posición a lo largo de la recta)
    # Proyectamos las estimaciones sobre la línea como punto de partida
    def project_scalars(P, origin, dir_unit):
        return np.dot(P - origin, dir_unit)

    s_init = project_scalars(P_est, P0_real, direction_unit)

    # Paso 4: definir función de coste (optimiza todos los puntos, sin forzar extremos)
    def cost(s_all):
        P_proj = P0_real + np.outer(s_all, direction_unit)
        return np.sum(np.linalg.norm(P_proj - P_est, axis=1)**2)

    # Paso 5: optimizar escalares para todos los puntos
    res = minimize(cost, s_init, method='L-BFGS-B')

    # Paso 6: reconstruir puntos corregidos sobre la línea
    s_opt = res.x
    P_corrected = P0_real + np.outer(s_opt, direction_unit)

    # Guardar las posiciones optimizadas
    save_positions(input_file_name=file_name,
    output_file_name=file_name,
    p_est_optimized=P_corrected,
    clean_if_exists=True)

    # Visualización del resultado
    plt.figure(figsize=(10, 6))

    # Línea de movimiento real
    plt.plot([P_real[0, 0], P_real[-1, 0]],
            [P_real[0, 1], P_real[-1, 1]],
            linestyle='--', color='gray', label='Línea de movimiento', linewidth=1)

    # Puntos extremos reales
    plt.scatter(*P_real[0], color='green', s=80, label='Inicio real')
    plt.scatter(*P_real[-1], color='red', s=80, label='Fin real')

    # Puntos estimados originales
    plt.scatter(P_est[:, 0], P_est[:, 1], color='blue', label='Estimaciones (por frame)', marker='x')

    # Puntos corregidos
    plt.scatter(P_corrected[:, 0], P_corrected[:, 1], color='orange', label='Posiciones real corregidas', marker='o')
    plt.plot(P_corrected[:, 0], P_corrected[:, 1], color='orange', linewidth=1)

    # Líneas desde puntos corregidos a estimados
    for i in range(len(P_est)):
        # Línea desde corrección a estimación
        plt.plot([P_corrected[i, 0], P_est[i, 0]],
                [P_corrected[i, 1], P_est[i, 1]],
                linestyle=':', color='orange', linewidth=0.8)

    plt.title('Trayectoria alineada por mínimos cuadrados')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
