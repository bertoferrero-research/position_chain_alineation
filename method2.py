import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Método 2 - Ajustamos la posición estimada real sincronizandola con la estimada por aruco mediante Needleman-Wunsch

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
    csv_path = os.path.join(os.path.dirname(__file__), 'optimized_positions_method2.csv')

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

def needleman_wunsch(seq1, seq2, match_score=1, gap_cost=-1, mismatch_cost=-1):
    """
    Algoritmo Needleman-Wunsch para alineación global de dos secuencias.
    seq1, seq2: listas de puntos (por ejemplo, posiciones [x, y])
    Devuelve los índices alineados de seq1 y seq2 (con None para gaps).
    """
    import numpy as np
    n = len(seq1)
    m = len(seq2)
    # Inicialización de la matriz de puntuación
    score = np.zeros((n+1, m+1))
    pointer = np.zeros((n+1, m+1), dtype=int)
    # Inicialización de bordes
    for i in range(1, n+1):
        score[i, 0] = gap_cost * i
    for j in range(1, m+1):
        score[0, j] = gap_cost * j
    # Rellenar la matriz
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

sample_space_millis_set = [0, 1000, 2000]
multiple_markers_behaviour_set = ['WEIGHTED_MEDIAN']#['CLOSEST','WEIGHTED_AVERAGE','AVERAGE','WEIGHTED_MEDIAN','MEDIAN']
markers_data = load_marker_positions()
first_exec = True

# Obtener la referencia real interpolada a ms=0 (solo una vez)
P_est_ref, P_real_ref, timestamps_ref, marker_ids_list_ref = load_positions(0, multiple_markers_behaviour_set[0])

for sample_space_millis in sample_space_millis_set:
    for multiple_markers_behaviour in multiple_markers_behaviour_set:
        P_est, P_real, timestamps, marker_ids_list = load_positions(sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour)

        # Needleman-Wunsch: siempre alineamos contra la referencia real interpolada ms=0
        aligned_real_idx, aligned_est_idx = needleman_wunsch(P_real_ref, P_est)

        # Construir listas alineadas (con None donde hay gaps)
        P_real_aligned = [P_real_ref[i] if i is not None else None for i in aligned_real_idx]
        P_est_aligned = [P_est[j] if j is not None else None for j in aligned_est_idx]
        timestamps_aligned = [timestamps_ref[i] if i is not None else None for i in aligned_real_idx]
        marker_ids_aligned = [marker_ids_list_ref[i] if i is not None else None for i in aligned_real_idx]

        print(f"Alineados: {len(P_real_aligned)} pares para sampleSpaceMillis={sample_space_millis}, behaviour={multiple_markers_behaviour}")
        plt.figure(figsize=(10, 6))

        # Guardar las posiciones alineadas (solo las que tienen ambos valores)
        # Ahora se guarda la posición real estimada alineada (P_real_aligned), no la estimada por aruco
        aligned_points = [(real, est, ts) for real, est, ts in zip(P_real_aligned, P_est_aligned, timestamps_aligned) if est is not None and real is not None]
        if aligned_points:
            P_real_save = np.array([real for real, _, _ in aligned_points])
            timestamps_save = np.array([ts for _, _, ts in aligned_points])
            save_positions(P_real_save, sample_space_millis, multiple_markers_behaviour, timestamps_save, clean_if_exists=first_exec)
            # Imprimir en el plot los puntos P_real_save
            plt.scatter(P_real_save[:, 0], P_real_save[:, 1], color='yellow', label='P_real_save (guardado)', marker='*', s=80, zorder=5)
            first_exec = False

        # Plot de las trayectorias alineadas
        real_x = [p[0] if p is not None else np.nan for p in P_real_aligned]
        real_y = [p[1] if p is not None else np.nan for p in P_real_aligned]
        est_x = [p[0] if p is not None else np.nan for p in P_est_aligned]
        est_y = [p[1] if p is not None else np.nan for p in P_est_aligned]

        real_valid = [(i, p) for i, p in enumerate(P_real_aligned) if p is not None]
        if real_valid:
            idxs, vals = zip(*real_valid)
            plt.plot([vals[0][0], vals[-1][0]], [vals[0][1], vals[-1][1]], linestyle='--', color='gray', label='Línea de movimiento', linewidth=1)
        plt.scatter(real_x, real_y, color='gray', label='Puntos reales (interpolados, alineados)', marker='.', s=30)
        if real_valid:
            plt.scatter(vals[0][0], vals[0][1], color='green', s=80, label='Inicio real')
            plt.scatter(vals[-1][0], vals[-1][1], color='red', s=80, label='Fin real')
        plt.scatter(est_x, est_y, color='blue', label='Estimaciones Aruco', marker='x')
        plt.scatter(P_real[:, 0], P_real[:, 1], color='orange', label='Puntos reales (sin alinear)', marker='^', s=30)
        
        for i, (p_est, p_real, marker_ids) in enumerate(zip(P_est_aligned, P_real_aligned, marker_ids_aligned)):
            if p_est is not None and p_real is not None:
                plt.plot([p_real[0], p_est[0]], [p_real[1], p_est[1]], linestyle=':', color='gray', linewidth=0.8)
        plt.title(f'Trayectoria y marcadores por frame (Needleman-Wunsch)\nSampleSpaceMillis={sample_space_millis}, Behaviour={multiple_markers_behaviour}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


