import pandas as pd
import logging
import numpy as np
import polars as pl
from typing import List


logger = logging.getLogger(__name__)

def cargar_datos(path: str, lazy: bool, months: List[int] = None) -> pl.DataFrame:
    """
    Carga un archivo Parquet en un DataFrame de Polars.
    """

    if lazy:
        try:
            df_lazy = pl.scan_parquet(path)

            df = (
                df_lazy
                .filter(pl.col("foto_mes").is_in(months))
                .sort(["numero_de_cliente", "foto_mes"])
                .collect()
            )

            logging.info(f"✅ Archivo cargado correctamente: {path}")
            logging.info(f"Shape: {df.shape}")
            return df

        except Exception as e:
            logging.info(f"⚠️ Error al cargar el archivo Parquet: {e}")

    else:
        try:
            df = pl.read_parquet(path)

            logging.info(f"Filas: {df.height}, Columnas: {df.width}")
            return df

        except Exception as e:
            logging.info(f"⚠️ Error al cargar el archivo Parquet: {e}")

def cargar_datos_csv(path: str, sep: str = ",", infer_schema_length: int = None, schema_overrides: dict = None, columns=None) -> pl.DataFrame:
    """
    Carga un archivo CSV comprimido (.csv.gz) en un DataFrame de Polars.

    Parámetros:
        ruta (str): Ruta del archivo CSV comprimido.
        sep (str): Separador de columnas (por defecto ',').
        infer_schema_length (int): Número de filas a usar para inferir el esquema.
        columns (list): Lista con las columnas que quiero cargar.

    Retorna:
        pl.DataFrame: DataFrame con los datos cargados.
    """
    try:
        df = pl.read_csv(
            path,
            separator=sep,
            infer_schema_length=infer_schema_length,
            has_header=True,
            try_parse_dates=True,
            columns=columns
        )
        logging.info(f"✅ Archivo CSV.gz cargado correctamente: {path}")
        logging.info(f"Filas: {df.height}, Columnas: {df.width}")
        return df
    except FileNotFoundError:
        logging.info(f"❌ Error: no se encontró el archivo '{path}'.")
    except Exception as e:
        logging.info(f"⚠️ Error al cargar el CSV.gz: {e}")


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



def load_dataset_undersampling_efficient(
        path: str,
        months: list[int],
        seed: int,
        fraction: float = 0.1
) -> pl.DataFrame:
    """
    Lee el dataset, crea columnas objetivo y aplica undersampling sobre CONTINUA.

    Args:
        path: Ruta al archivo parquet
        months: Lista de meses o mes único a cargar
        fraction: Fracción de CONTINUA a mantener (ej: 0.1 = 10%)
        seed: Semilla para reproducibilidad

    Returns:
        pl.DataFrame con undersampling aplicado
    """
    if isinstance(months, str):
        months = [months]

    if path.endswith('.parquet'):
        scan_func = pl.scan_parquet(path, low_memory=True)
    elif path.endswith('.csv.gz') or path.endswith('.csv'):
        scan_func = pl.scan_csv(path, low_memory=True)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {path}")

    logger.info(f"Cargando dataset desde {path} y creando variables target")

    # Base lazy con columnas objetivo
    df_lazy = (
        scan_func
        .filter(pl.col("foto_mes").is_in(months))
        .with_columns([
            pl.when(pl.col("target") == "CONTINUA").then(0).otherwise(1).alias("target_train"),
            pl.when(pl.col("target") == "BAJA+2").then(1).otherwise(0).alias("target_test"),
            pl.when(pl.col("target") == "CONTINUA").then(1)
            .when(pl.col("target") == "BAJA+1").then(1.00001)
            .when(pl.col("target") == "BAJA+2").then(1.00002)
            .otherwise(None)
            .alias("w_train")
        ])
    )

    logger.info("Aplicando undersampling...")

    # 1. Separar lógica de filtros (esto sigue siendo una query plan, no data real)
    df_mayoritaria = df_lazy.filter(
        pl.col("target") == "CONTINUA")
    df_minoritaria = df_lazy.filter(pl.col("target") != "CONTINUA")

    # Extraer sólo lo necesario para muestrear
    clientes_df = (
        df_mayoritaria
        .select(["numero_de_cliente"])
        .unique(maintain_order=True)
        .collect()
    )

    # Sample de los continua
    clientes_sampled = (
        clientes_df.sample(fraction=fraction, seed=seed)
    )

    cliente_ids = clientes_sampled["numero_de_cliente"].to_list()

    df_mayoritaria_sampled = df_mayoritaria.filter(
        pl.col("numero_de_cliente").is_in(cliente_ids)
    )

    df_lazy = pl.concat([df_mayoritaria_sampled, df_minoritaria])

    df = df_lazy.collect()

    logger.info("Calculando conteos post-undersampling por mes y clase...")
    counts_df = df.group_by("foto_mes", "target").agg(
        pl.count().alias("registros")
    ).sort("foto_mes", "target")
    with pl.Config(tbl_rows=-1, tbl_width_chars=120):
        counts_str = str(counts_df)
    logger.info(
        f"--- Conteos Finales Post-Undersampling ---\n{counts_str}\n--------------------------------------------")
    num_cols = df.width
    logger.info(f"Columnas del dataset muestreado: {num_cols}")
    num_rows = df.height
    logger.info(f"Filas del dataset muestreado: {num_rows}")

    return df
