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

    return p_est, p_real, timestamps

def save_positions(p_est_optimized, sample_space_millis, multiple_markers_behaviour, timestamps):
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
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, sep=';', decimal=',')
        df = pd.concat([df_existing, df], ignore_index=True)
        # Ordenar por sampleSpaceMillis y luego por timestamp
        
    df = df.sort_values(by=['sampleSpaceMillis', 'timestamp'])
    df.to_csv(csv_path, sep=';', decimal=',', index=False)

sample_space_millis_set = [0, 1000, 2000]
multiple_markers_behaviour_set = ['WEIGHTED_MEDIAN']#['CLOSEST','WEIGHTED_AVERAGE','AVERAGE','WEIGHTED_MEDIAN','MEDIAN']

for sample_space_millis in sample_space_millis_set:
    for multiple_markers_behaviour in multiple_markers_behaviour_set:

        p_est, p_real, timestamps = load_positions(sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour)

        # Suponé que ya tenés esto:
        # P_est: (n, 2) interpolación inicial
        # P_real: (n, 2) posiciones reales
        n = p_est.shape[0]
        d = 2  # 2D

        # Paso 1: obtener extremos reales
        P0_real = p_real[0]
        Pn_real = p_real[-1]

        # Paso 2: vector dirección unitario de la línea recta real
        direction = Pn_real - P0_real
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / direction_norm

        # Paso 3: proyectar puntos estimados sobre la línea recta
        def project_onto_line(P, origin, direction_unit):
            vectors = P - origin  # vectores desde el origen a cada punto
            scalars = np.dot(vectors, direction_unit)  # proyección escalar
            P_proj = origin + np.outer(scalars, direction_unit)  # reconstrucción sobre la línea
            return P_proj, scalars

        P_proj, scalar_init = project_onto_line(p_est, P0_real, direction_unit)

        # Paso 4: preparar optimización sobre los escalares intermedios
        s0 = scalar_init[0]
        sn = scalar_init[-1]

        def scalar_error(s_middle):
            s_full = np.concatenate(([s0], s_middle, [sn]))
            P_corr = P0_real + np.outer(s_full, direction_unit)
            return np.sum(np.linalg.norm(P_corr - p_est, axis=1)**2)

        # Optimizar los valores escalares intermedios
        res = minimize(scalar_error, scalar_init[1:-1], method='L-BFGS-B')

        # Paso 5: reconstruir puntos corregidos
        s_opt = np.concatenate(([s0], res.x, [sn]))
        P_corrected = P0_real + np.outer(s_opt, direction_unit)

        # Guardar las posiciones optimizadas
        save_positions(P_corrected, sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour, timestamps=timestamps)


        plt.figure(figsize=(10, 6))

        # Línea de movimiento real (recta entre extremos reales)
        plt.plot([p_real[0, 0], p_real[-1, 0]],
                [p_real[0, 1], p_real[-1, 1]],
                linestyle='--', label='Línea de movimiento', linewidth=1)

        # Puntos extremos reales
        plt.scatter(*p_real[0], color='green', s=80, label='Inicio real')
        plt.scatter(*p_real[-1], color='red', s=80, label='Fin real')

        # Puntos estimados originales
        plt.scatter(p_est[:, 0], p_est[:, 1], color='blue', label='Estimaciones (por frame)', marker='x')

        # Puntos corregidos
        plt.scatter(P_corrected[:, 0], P_corrected[:, 1], color='orange', label='Estimaciones corregidas', marker='o')

        # Conectar los puntos corregidos para ver la trayectoria
        plt.plot(P_corrected[:, 0], P_corrected[:, 1], color='orange', linewidth=1)

        plt.title('Corrección de trayectoria sobre línea recta')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()