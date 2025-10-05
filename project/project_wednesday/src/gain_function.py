import numpy as np
import pandas as pd
import polars as pl
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger(__name__)


# def calcular_ganancia(y_true, y_pred):
#     """
#     Calcula la ganancia total usando la función de ganancia de la competencia.
#
#     Args:
#         y_true: Valores reales (0 o 1)
#         y_pred: Predicciones (0 o 1)
#
#     Returns:
#         float: Ganancia total
#     """
#     # Convertir a numpy arrays si es necesario
#     if isinstance(y_true, pd.Series):
#         y_true = y_true.values
#     if isinstance(y_pred, pd.Series):
#         y_pred = y_pred.values
#
#     # Calcular ganancia vectorizada usando configuración
#     # Verdaderos positivos: y_true=1 y y_pred=1 -> ganancia
#     # Falsos positivos: y_true=0 y y_pred=1 -> costo
#     # Verdaderos negativos y falsos negativos: ganancia = 0
#
#     ganancia_total = np.sum(
#         ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +
#         ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)
#     )
#
#     # logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
#     #              f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")
#
#     return ganancia_total

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


# def ganancia_lgb_binary(y_pred, y_true):
#     """
#     Función de ganancia para LightGBM en clasificación binaria.
#     Compatible con callbacks de LightGBM.
#
#     Args:
#         y_pred: Predicciones de probabilidad del modelo
#         y_true: Dataset de LightGBM con labels verdaderos
#
#     Returns:
#         tuple: (eval_name, eval_result, is_higher_better)
#     """
#     # Obtener labels verdaderos
#     y_true_labels = y_true.get_label()
#
#     # Convertir probabilidades a predicciones binarias (umbral 0.5)
#     y_pred_binary = (y_pred > 0.025).astype(int)
#
#     # Calcular ganancia usando configuración
#     ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
#
#     # Retornar en formato esperado por LightGBM
#     return 'ganancia', ganancia_total, True  # True = higher is better

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

