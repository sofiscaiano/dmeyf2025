import pandas as pd
import os
import logging
from datetime import datetime
from .config import STUDY_NAME

logger = logging.getLogger(__name__)


def guardar_predicciones_finales(resultados_df: pd.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.

    Args:
        resultados_df: DataFrame con numero_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)

    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)

    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME

    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"

    # Guardar archivo
    resultados_df.to_csv(ruta_archivo, index=False)

    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {list(resultados_df.columns)}")
    logger.info(f"  Registros: {len(resultados_df):,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{resultados_df.head()}")

    return ruta_archivo

def exportar_envios_bot(resultados_df: pd.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.

    Args:
        resultados_df: DataFrame con numero_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)

    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)

    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME

    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"

    # Guardar archivo
    df_filtrado = resultados_df[resultados_df["Predicted"] == 1][["numero_de_cliente"]]
    df_filtrado.to_csv(ruta_archivo, header=False, index=False)

    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {list(df_filtrado.columns)}")
    logger.info(f"  Registros: {len(df_filtrado):,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{df_filtrado.head()}")

    return ruta_archivo