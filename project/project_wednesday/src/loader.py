import pandas as pd
import logging
import pyarrow.csv as pv
import numpy as np


logger = logging.getLogger(__name__)
def cargar_datos(path: str) -> pd.DataFrame | None:

    '''
    Carga un CSV desde 'path' con pyarrow y retorna un pd.DataFrame
    '''

    logger.info(f'Cargando dataset desde {path}')
    try:
        tabla_pyarrow = pv.read_csv(path)
        df = tabla_pyarrow.to_pandas()
        logger.info(f'Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas')
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
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()

    # Contar valores originales para logging
    n_continua_orig = (df_result['target'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['target'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['target'] == 'BAJA+2').sum()

    # Convertir clase_ternaria a binario respetando mi objetivo principal que son los baja+2
    df_result['target_test'] = df_result['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 0,
        'BAJA+2': 1
    })

    # Convertir clase_ternaria a binario en el mismo atributo considerando todas las bajas para training
    df_result['target'] = df_result['target'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })


    # Log de la conversión
    n_ceros = (df_result['target'] == 0).sum()
    n_unos = (df_result['target'] == 1).sum()

    # Log de la conversión
    n_ceros_test = (df_result['target_test'] == 0).sum()
    n_unos_test = (df_result['target_test'] == 1).sum()

    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos / (n_ceros + n_unos) * 100:.2f}% casos positivos")
    logger.info(f"  Real BAJA+2 -> Binario - 0: {n_ceros_test}, 1: {n_unos_test}")

    return df_result


def reduce_mem_usage(df):
    """ Itera sobre todas las columnas de un dataframe y modifica el tipo de dato
        para reducir el uso de memoria.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Uso de memoria del dataframe es {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Convierte las columnas de tipo 'object' a 'category'
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Uso de memoria después de la optimización: {:.2f} MB'.format(end_mem))
    print('Disminución de {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
