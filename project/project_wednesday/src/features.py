import pandas as pd
import numpy as np
import duckdb
import logging
from .config import SEMILLA

logger = logging.getLogger(__name__)

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    # logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    # print(df.head())

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


def undersample(df, sample_fraction):
    """
    Realiza un undersampling de la clase mayoritaria.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        sample_fraction (float): Fracción de la clase mayoritaria a conservar (entre 0 y 1).
    Returns:
        pd.DataFrame: El DataFrame resultante submuestreado.
    """

    # Función lambda para aplicar a cada grupo (clase)
    # Si el grupo es la clase mayoritaria, se aplica el muestreo con la fracción.
    # Para el resto de clases, se conserva el 100% de los datos (frac=1).
    df_undersampled = df.groupby('target', group_keys=False).apply(
        lambda x: x.sample(
            frac=sample_fraction,
            random_state=SEMILLA[1]
        ) if x.name == 0 else x
    ).reset_index(drop=True)

    prop_continua = (df_undersampled['target'] == 0).sum() / (df['target'] == 0).sum()
    prop_baja = (df_undersampled['target'] == 1).sum() / (df['target'] == 1).sum()

    # Imprimir estadísticas para verificar la reducción
    logging.info(f"Tamaño original del DataFrame: {len(df)}")
    logging.info(f"Tamaño final del DataFrame: {len(df_undersampled)}")
    logging.info(f"Proporcion final de clase mayoritaria: {prop_continua:.2f}")
    logging.info(f"Proporcion final de clase minoritaria: {prop_baja:.2f}")

    return df_undersampled
