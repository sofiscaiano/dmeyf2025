import pandas as pd
import logging
import pyarrow.csv as pv
import pyarrow.parquet as pq
import numpy as np


logger = logging.getLogger(__name__)
def cargar_datos(path: str, columns: list = None, use_threads: bool = True) -> pd.DataFrame | None:
    '''
    Carga un CSV o Parquet desde 'path' con pyarrow y retorna un pd.DataFrame.
    Versión optimizada con soporte para columnas específicas y multithreading.

    Args:
        path: Ruta al archivo CSV o Parquet
        columns: Lista de columnas a cargar (None = todas)
        use_threads: Si True, usa multithreading para lectura

    Returns:
        pd.DataFrame con los datos cargados
    '''

    logger.info(f'Cargando dataset desde {path}')
    try:
        if path.endswith('.parquet'):
            # Cargar parquet con optimizaciones
            tabla_pyarrow = pq.read_table(
                path,
                columns=columns,
                use_threads=use_threads
            )
        else:
            # Cargar CSV
            tabla_pyarrow = pv.read_csv(path)
            if columns is not None:
                tabla_pyarrow = tabla_pyarrow.select(columns)

        df = tabla_pyarrow.to_pandas()
        logger.info(f'Dataset cargado con {df.shape[0]:,} filas y {df.shape[1]} columnas')
        return df
    except Exception as e:
        logger.error(f'Error al cargar el dataset: {e}')
        raise


def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1

    Args:
        df: DataFrame con columna 'clase_ternaria'

    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Contar valores originales para logging (before modification)
    n_continua_orig = (df['target'] == 'CONTINUA').sum()
    n_baja1_orig = (df['target'] == 'BAJA+1').sum()
    n_baja2_orig = (df['target'] == 'BAJA+2').sum()

    # Convertir clase_ternaria a binario respetando mi objetivo principal que son los baja+2
    df['target_test'] = df['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 0,
        'BAJA+2': 1
    })

    # Convertir clase_ternaria a binario en el mismo atributo considerando todas las bajas para training
    df['target'] = df['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })

    # Log de la conversión
    n_ceros = (df['target'] == 0).sum()
    n_unos = (df['target'] == 1).sum()

    # Log de la conversión
    n_ceros_test = (df['target_test'] == 0).sum()
    n_unos_test = (df['target_test'] == 1).sum()

    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos / (n_ceros + n_unos) * 100:.2f}% casos positivos")
    logger.info(f"  Real BAJA+2 -> Binario - 0: {n_ceros_test}, 1: {n_unos_test}")

    return df


def reduce_mem_usage(df, verbose=True):
    """
    Optimiza el uso de memoria del dataframe reduciendo los tipos de dato.
    Versión optimizada con mejor manejo de float16 y procesamiento por tipo.

    Args:
        df: DataFrame a optimizar
        verbose: Si True, muestra información de progreso

    Returns:
        DataFrame optimizado
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        logger.info(f'Uso de memoria inicial del dataframe: {start_mem:.2f} MB')

    # Procesar columnas numéricas int
    int_cols = df.select_dtypes(include=['int64', 'int32', 'int16']).columns
    for col in int_cols:
        c_min = df[col].min()
        c_max = df[col].max()

        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

    # Procesar columnas numéricas float (evitar float16 por pérdida de precisión en ML)
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        c_min = df[col].min()
        c_max = df[col].max()

        # Solo usar float32 para reducir memoria, evitar float16
        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)

    # Convertir columnas object a category si tienen pocos valores únicos
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        # Solo convertir a category si tiene menos del 50% de valores únicos
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        logger.info(f'Uso de memoria después de la optimización: {end_mem:.2f} MB')
        logger.info(f'Reducción de memoria: {reduction:.1f}%')

    return df
