import numpy as np
import pandas as pd
import polars as pl
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger(__name__)


def calcular_ganancias_acumuladas(y_true, y_pred_proba):
    """
    Calcula las ganancias acumuladas ordenando por probabilidad descendente

    Args:
        y_true: Valores reales (0 o 1)
        y_pred_proba: Predicciones en formato probabilidad

    Returns:
        numpy array: Ganancias acumuladas
    """

    # # Target
    # y_true = y_true.get_label()
    # DataFrame con target y predicciones
    df_eval = pl.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    # Ordenado las probabilidades de forma descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    # Con Polars realizo el calculo de ganancia individual de cada registro segun los verdaderos valores
    df_ordenado = df_ordenado.with_columns(
        [pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias(
            'ganancia_individual')])
    # Calculo la ganancia acumulada en cada paso
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')])
    ganancias_acumuladas = df_ordenado['ganancia_acumulada'].to_numpy()

    return ganancias_acumuladas



def ganancia_evaluator(y_pred, y_true):
    """
    Funcion de evaluacion personalizada para LightGBM
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    para encontrar el punto de maxima ganancia
    :param y_pred:Predicciones de probabilidad del modelo
    :param y_true:Dataset de LightGBM con labels verdaderos
    :return: Ganancia total
    """

    # Target
    y_true = y_true.get_label()
    # DataFrame con target y predicciones
    df_eval = pl.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred})
    # Ordenado las probabilidades de forma descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    # Con Polars realizo el calculo de ganancia individual de cada registro segun los verdaderos valores
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])
    # Calculo la ganancia acumulada en cada paso
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')])
    # Obtengo la ganancia maxima de la serie
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()

    return 'ganancia', ganancia_maxima, True

