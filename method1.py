import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt



def load_positions(sample_space_millis, multiple_markers_behaviour):

    # Comenzamos cargando el csv con panda
    import pandas as pd
    import os
    import sys
    
    # Definimos la ruta del archivo CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'samples.csv')
    if not os.path.exists(csv_path):
        print(f"Error: El archivo {csv_path} no existe.")
        sys.exit(1)
    # Cargamos el CSV
    df = pd.read_csv(csv_path, sep=';',decimal=',')

    # Filtramos
    df = df[(df['sampleSpaceMillis'] == sample_space_millis) & (df['multipleMarkersBehaviour'] == multiple_markers_behaviour)]

    # Obtenemos las posiciones estimadas y reales
    p_est = df[['rawX', 'rawY']].values
    p_real = df[['realX', 'realY']].values
    timestamps = df['timestamp'].values

    # Extraer ids de marcadores de la columna markers_info
    import re
    def extract_ids(marker_info_str):
        # Busca markerId=NUMERO en la cadena y extrae los números
        return [int(m) for m in re.findall(r'markerId=(\d+)', str(marker_info_str))]
    marker_ids_list = df['markers_info'].apply(extract_ids).tolist()

    return p_est, p_real, timestamps, marker_ids_list

def save_positions(p_est_optimized, sample_space_millis, multiple_markers_behaviour, timestamps, clean_if_exists = False):
    # Guardamos las posiciones optimizadas en un CSV
    import pandas as pd
    import os
    
    # Definimos la ruta del archivo CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'optimized_positions.csv')

    # Creamos un DataFrame con las posiciones optimizadas
    df = pd.DataFrame(p_est_optimized, columns=['optimizedRawX', 'optimizedRawY'])
    df['sampleSpaceMillis'] = sample_space_millis
    df['multipleMarkersBehaviour'] = multiple_markers_behaviour
    df['timestamp'] = timestamps

    # Si el archivo existe, añadimos; si no, lo creamos
    if os.path.exists(csv_path) and not clean_if_exists:
        df_existing = pd.read_csv(csv_path, sep=';', decimal=',')
        df = pd.concat([df_existing, df], ignore_index=True)
        # Ordenar por sampleSpaceMillis y luego por timestamp
        
    df = df.sort_values(by=['sampleSpaceMillis', 'timestamp'])
    df.to_csv(csv_path, sep=';', decimal=',', index=False)

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

sample_space_millis_set = [0, 1000, 2000]
multiple_markers_behaviour_set = ['WEIGHTED_MEDIAN']#['CLOSEST','WEIGHTED_AVERAGE','AVERAGE','WEIGHTED_MEDIAN','MEDIAN']
markers_data = load_marker_positions()
first_exec = True


for sample_space_millis in sample_space_millis_set:
    for multiple_markers_behaviour in multiple_markers_behaviour_set:

        P_est, P_real, timestamps, marker_ids_list = load_positions(sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour)

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

        # Forzar extremos
        s_init[0] = 0.0
        s_init[-1] = length

        # Paso 4: definir función de coste
        def cost(s_middle):
            s_full = np.concatenate(([0.0], s_middle, [length]))
            P_proj = P0_real + np.outer(s_full, direction_unit)
            return np.sum(np.linalg.norm(P_proj - P_est, axis=1)**2)

        # Paso 5: optimizar escalares intermedios
        res = minimize(cost, s_init[1:-1], method='L-BFGS-B')

        # Paso 6: reconstruir puntos corregidos sobre la línea
        s_opt = np.concatenate(([0.0], res.x, [length]))
        P_corrected = P0_real + np.outer(s_opt, direction_unit)

        # Guardar las posiciones optimizadas
        save_positions(P_corrected, sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour, timestamps=timestamps, clean_if_exists=first_exec)


        plt.figure(figsize=(10, 6))

        # Línea de movimiento real
        plt.plot([P_real[0, 0], P_real[-1, 0]],
                [P_real[0, 1], P_real[-1, 1]],
                linestyle='--', color='gray', label='Línea de movimiento', linewidth=1)

        # Puntos reales interpolados
        plt.scatter(P_real[:, 0], P_real[:, 1], color='gray', label='Puntos reales (interpolados)', marker='.', s=30)

        # Puntos extremos reales
        plt.scatter(*P_real[0], color='green', s=80, label='Inicio real')
        plt.scatter(*P_real[-1], color='red', s=80, label='Fin real')

        # Puntos estimados
        plt.scatter(P_est[:, 0], P_est[:, 1], color='blue', label='Estimaciones (por frame)', marker='x')

        # Puntos corregidos
        plt.scatter(P_corrected[:, 0], P_corrected[:, 1], color='orange', label='Estimaciones corregidas', marker='o')
        plt.plot(P_corrected[:, 0], P_corrected[:, 1], color='orange', linewidth=1)

        # Líneas desde puntos reales y corregidos a estimados
        for i in range(len(P_est)):
            # Línea desde interpolación inicial a estimación
            plt.plot([P_real[i, 0], P_est[i, 0]],
                    [P_real[i, 1], P_est[i, 1]],
                    linestyle=':', color='gray', linewidth=0.8)

            # Línea desde corrección a estimación
            plt.plot([P_corrected[i, 0], P_est[i, 0]],
                    [P_corrected[i, 1], P_est[i, 1]],
                    linestyle=':', color='orange', linewidth=0.8)

            # Dibujar marcadores y líneas a estimado
            for marcador_id in marker_ids_list[i]:
                marcador = markers_data.get(marcador_id)
                # Dibujar el marcador como un cuadrado
                plt.scatter(marcador[0], marcador[1], color='purple', marker='s', s=40)
                # Línea desde el marcador a la posición estimada correspondiente
                plt.plot([marcador[0], P_est[i, 0]],
                        [marcador[1], P_est[i, 1]],
                        linestyle='-', color='purple', linewidth=0.8)

        plt.title('Trayectoria y marcadores por frame')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        first_exec = False