import numpy as np
from scipy.optimize import minimize

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

        # Fijamos los extremos
        P0 = p_est[0]
        Pn = p_est[-1]

        # Aplanamos los puntos intermedios (lo que vamos a optimizar)
        initial_guess = p_est[1:-1].flatten()

        def total_error(P_flat):
            # Reconstruimos el array completo de posiciones con extremos fijos
            P_mid = P_flat.reshape(-1, d)
            P_full = np.vstack([P0, P_mid, Pn])
            
            # Error euclídeo al cuadrado
            error = np.sum(np.linalg.norm(P_full[1:-1] - p_real[1:-1], axis=1)**2)

            # (Opcional) Agregar suavidad:
            # for i in range(1, len(P_full) - 1):
            #     error += lambda_ * np.linalg.norm(P_full[i-1] - 2 * P_full[i] + P_full[i+1])**2
            
            return error

        # Ejecutar optimización
        res = minimize(total_error, initial_guess, method='L-BFGS-B')

        # Reconstruir las posiciones corregidas
        P_optimized = np.vstack([P0, res.x.reshape(-1, d), Pn])

        # Guardar las posiciones optimizadas
        save_positions(P_optimized, sample_space_millis=sample_space_millis, multiple_markers_behaviour=multiple_markers_behaviour, timestamps=timestamps)

