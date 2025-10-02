import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDICT, SEMILLA
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)


def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.

    Args:
        df: DataFrame con todos los datos

    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDICT}")

    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    if isinstance(FINAL_TRAIN, list):
        df_train = df[df['foto_mes'].isin(FINAL_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == FINAL_TRAIN]

    # Datos de predicción: período FINAL_PREDIC
    df_predict = df[df['foto_mes'] == FINAL_PREDICT]

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")

    # Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
    X_train = df_train.drop(['target', 'foto_mes'], axis=1)
    y_train = df_train['target']

    # Preparar features para predicción
    X_predict = df_predict.drop(['target', 'foto_mes'], axis=1)
    clientes_predict = df_predict['numero_de_cliente']

    logger.info(f"Features utilizadas: {len(X_predict.columns)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

    return X_train, y_train, X_predict, clientes_predict


def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna

    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")

    # Hiperparámetros optimizados
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,  # para reducir warnings
        'max_depth': -1,  # -1 significa no limitar,  por ahora lo dejo fijo
        'min_gain_to_split': 0,
        'min_sum_hessian_in_leaf': 0.001,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'max_bin': 31,
        'pos_bagging_fraction': 1,
        'neg_bagging_fraction': 1,
        'is_unbalance': False,
        'scale_pos_weight': 1,
        'extra_trees': False,
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        **mejores_params  # Agregar los mejores hiperparámetros
    }

    logger.info(f"Parámetros del modelo: {params}")

    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    # Entrenar modelo con lgb.train()
    modelo = lgb.train(
        params,
        train_data,
        num_boost_round=mejores_params.get('num_iterations', 1000)
        # callbacks=[
        #     lgb.early_stopping(stopping_rounds=50),
        #     lgb.log_evaluation(period=100)
        # ],
        # feval=ganancia_lgb_binary
    )
    return modelo


def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray,
                                 umbral: float = 0.025) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria

    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    # Generar probabilidades con el modelo entrenado
    probabilidades = modelo.predict(X_predict)

    # Convertir a predicciones binarias con el umbral establecido
    predicciones_binarias = (probabilidades > umbral).astype(int)

    # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predicted': predicciones_binarias
    })

    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['Predicted'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")

    return resultados