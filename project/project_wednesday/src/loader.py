import pandas as pd
import logging
import numpy as np
import polars as pl


logger = logging.getLogger(__name__)

def cargar_datos(path: str) -> pl.DataFrame:
    """
    Carga un archivo Parquet en un DataFrame de Polars.

    Parámetros:
        ruta (str): Ruta del archivo Parquet a cargar.

    Retorna:
        pl.DataFrame: DataFrame con los datos cargados.
    """
    try:
        df = pl.read_parquet(path)
        print(f"✅ Archivo cargado correctamente: {path}")
        print(f"Filas: {df.height}, Columnas: {df.width}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: no se encontró el archivo '{path}'.")
    except Exception as e:
        print(f"⚠️ Error al cargar el archivo Parquet: {e}")

def cargar_datos_csv(path: str, sep: str = ",", infer_schema_length: int = 10000) -> pl.DataFrame:
    """
    Carga un archivo CSV comprimido (.csv.gz) en un DataFrame de Polars.

    Parámetros:
        ruta (str): Ruta del archivo CSV comprimido.
        sep (str): Separador de columnas (por defecto ',').
        infer_schema_length (int): Número de filas a usar para inferir el esquema.

    Retorna:
        pl.DataFrame: DataFrame con los datos cargados.
    """
    try:
        df = pl.read_csv(
            path,
            separator=sep,
            infer_schema_length=infer_schema_length,
            has_header=True,
            try_parse_dates=True
        )
        print(f"✅ Archivo CSV.gz cargado correctamente: {path}")
        print(f"Filas: {df.height}, Columnas: {df.width}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: no se encontró el archivo '{path}'.")
    except Exception as e:
        print(f"⚠️ Error al cargar el CSV.gz: {e}")


def convertir_clase_ternaria_a_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte 'target' a binario de forma eficiente para datasets grandes:
      - target_test: solo BAJA+2 = 1
      - target: BAJA+1 y BAJA+2 = 1, CONTINUA = 0
    """
    # Contar valores originales antes de convertir
    counts_orig = df.select([
        (pl.col("target") == "CONTINUA").sum().alias("n_continua_orig"),
        (pl.col("target") == "BAJA+1").sum().alias("n_baja1_orig"),
        (pl.col("target") == "BAJA+2").sum().alias("n_baja2_orig"),
    ]).to_dict(as_series=False)

    # Crear columnas binarias de forma vectorizada
    df = df.with_columns([
        (pl.col("target") == "BAJA+2").cast(pl.Int8).alias("target_test"),
        (pl.col("target").is_in(["BAJA+1", "BAJA+2"])).cast(pl.Int8).alias("target")
    ])

    # Contar 0s y 1s después de la conversión
    counts_bin = df.select([
        (pl.col("target") == 0).sum().alias("n_ceros"),
        (pl.col("target") == 1).sum().alias("n_unos"),
        (pl.col("target_test") == 0).sum().alias("n_ceros_test"),
        (pl.col("target_test") == 1).sum().alias("n_unos_test")
    ]).to_dict(as_series=False)

    logger.info("Conversión completada:")
    logger.info(f"  Original - CONTINUA: {counts_orig['n_continua_orig'][0]}, "
                f"BAJA+1: {counts_orig['n_baja1_orig'][0]}, BAJA+2: {counts_orig['n_baja2_orig'][0]}")
    logger.info(f"  Binario - 0: {counts_bin['n_ceros'][0]}, 1: {counts_bin['n_unos'][0]}")
    total = counts_bin['n_ceros'][0] + counts_bin['n_unos'][0]
    logger.info(f"  Distribución: {counts_bin['n_unos'][0] / total * 100:.2f}% casos positivos")
    logger.info(f"  Real BAJA+2 -> Binario - 0: {counts_bin['n_ceros_test'][0]}, 1: {counts_bin['n_unos_test'][0]}")

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
