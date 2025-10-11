import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .plots import plot_mean_importance
from .config import FINAL_TRAIN, FINAL_PREDICT, SEMILLA
from .config import *
import glob
import gc

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

    # Preparar features y target para entrenamiento
    X_train = df_train.drop(['target', 'target_test'], axis=1)
    y_train = df_train['target']

    # Preparar features para predicción
    X_predict = df_predict.drop(['target', 'target_test'], axis=1)
    clientes_predict = df_predict['numero_de_cliente']

    del df, df_predict, df_train
    gc.collect()

    logger.info(f"Features utilizadas: {len(X_predict.columns)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

    return X_train, y_train, X_predict, clientes_predict


def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> list:
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

    flag_GPU = int(os.getenv('GPU', 0))

    if flag_GPU == 0:
        gpu_dict = {'device': 'cpu'}
    else:
        gpu_dict = {'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0}

    # Hiperparámetros fijos y optimizados
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'n_threads': -1,
        **gpu_dict,
        'feature_pre_filter': PARAMETROS_LGB['feature_pre_filter'],
        'force_row_wise': PARAMETROS_LGB['force_row_wise'],  # para reducir warnings
        'max_bin': PARAMETROS_LGB['max_bin'],
        'seed': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        # 'data_random_seed': SEMILLA[0],
        # 'feature_fraction_seed': SEMILLA[0],
        **mejores_params
    }

    if isinstance(MES_TRAIN, list):
        periodos_validacion = MES_TRAIN
    else:
        periodos_validacion = [MES_TRAIN]

    if MES_VALIDACION is not None:
        periodos_validacion.append(MES_VALIDACION)

    # params['min_data_in_leaf'] = round((len(FINAL_TRAIN)/len(periodos_validacion)) * params['min_data_in_leaf'])

    logger.info(f"Parámetros del modelo: {params}")

    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    del X_train, y_train
    gc.collect()

    logging.info('=== Inicio Entrenamiento del Modelo Final con 5 semillas ===')

    modelos = []
    all_importances = []
    importance_type = 'gain'  # O 'split'

    # Carpeta donde guardar los modelos
    models_dir = "resultados/modelos/"
    os.makedirs(models_dir, exist_ok=True)

    for i, seed in enumerate(SEMILLA):
        logging.info(f'Entrenando modelo con seed = {seed} ({i+1}/{len(SEMILLA)})')

        # Copia de parámetros con la semilla actual
        params_seed = params.copy()
        params_seed['seed'] = seed

        # Entrenamiento
        modelo = lgb.train(
            params_seed,
            train_data,
            num_boost_round=mejores_params.get('num_iterations', 1000)
        )

        # Generamos un DataFrame temporal con la importancia de este modelo
        feature_imp = pd.DataFrame({
            'feature': modelo.feature_name(),
            'importance': modelo.feature_importance(importance_type=importance_type)
        })
        all_importances.append(feature_imp)

        # Guardar el modelo entrenado
        model_path = os.path.join(models_dir, f"{STUDY_NAME}_lgb_seed_{seed}.txt")
        modelo.save_model(model_path)
        logging.info(f'Modelo guardado en: {model_path}')

        modelos.append(modelo)

        # Memory cleanup after each model
        del feature_imp, params_seed
        gc.collect()

    logging.info('=== Inicio Grafico de Importancia ===')
    plot_mean_importance(all_importances, importance_type, type='train')

    logging.info('=== Finaliza Entrenamiento de los 5 Modelos ===')

    return modelos


def generar_predicciones_finales(X_predict: pd.DataFrame, clientes_predict: np.ndarray,
                                 envios: int) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.

    Args:
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        envios: cantidad de envios a realizar

    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    models_dir = "resultados/modelos/"
    model_files = sorted(glob.glob(f"{models_dir}/{STUDY_NAME}_lgb_seed_*.txt"))

    preds = []

    for file in model_files:
        modelo = lgb.Booster(model_file=file)
        feature_names = modelo.feature_name() # obtengo los nombres de las features
        X_predict = X_predict[feature_names]  # respeto el orden por si hice cambios
        preds.append(modelo.predict(X_predict))

    # Ensemble final (promedio)
    y_pred_proba = np.mean(preds, axis=0)

    # Crear un DataFrame para manejar el orden
    resultados = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "probabilidad": y_pred_proba
    })

    # Ordenar por probabilidad descendente
    resultados = resultados.sort_values(by="probabilidad", ascending=False).reset_index(drop=True)

    # 4. Asignar etiquetas: 1 a los primeros envios, 0 al resto
    resultados["Predicted"] = 0
    resultados.loc[:envios, "Predicted"] = 1

    resultados.drop(columns='probabilidad', inplace=True)

    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['Predicted'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Cantidad de envios: {envios}")

    return resultados