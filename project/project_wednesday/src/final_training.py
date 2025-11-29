import pandas as pd
import lightgbm as lgb
import numpy as np
import polars as pl
import logging
import os
from .plots import plot_mean_importance
from .basic_functions import generar_semillas, undersample, train_test_split
from .config import *
import glob
import gc
import mlflow

logger = logging.getLogger(__name__)


def preparar_datos_entrenamiento_final(df: pl.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.

    Args:
        df: DataFrame con todos los datos

    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")

    if UNDERSAMPLING_FINAL_TRAINING:
        X_train, y_train, X_test, y_test, feature_name = train_test_split(df=df, undersampling=True, mes_train=FINAL_TRAIN, mes_test=MES_TEST)
    else:
        X_train, y_train, X_test, y_test, feature_name = train_test_split(df=df, undersampling=False, mes_train=FINAL_TRAIN, mes_test=MES_TEST)

    logger.info(f"Registros de entrenamiento final: {X_train.shape}")
    mlflow.log_param("X_train_final_shape", X_train.shape)

    return X_train, y_train

def entrenar_modelo_final(df: pl.DataFrame, mejores_params: dict) -> list:
    """
    Entrena el modelo final con los mejores hiperparámetros.

    Args:
        df: Dataframe de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna

    Returns:
        modelos: Lista de los modelos entrenados
    """
    logger.info("Iniciando entrenamiento del modelo final")

    X_train, y_train = preparar_datos_entrenamiento_final(df)

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
        **mejores_params
    }

    logger.info(f"Parámetros del modelo: {params}")

    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True, feature_name=feature_names)

    logging.info(f'=== Inicio Entrenamiento del Modelo Final con {KSEMILLERIO} semillas ===')

    modelos = []
    all_importances = []
    importance_type = 'gain'  # O 'split'

    # Carpeta donde guardar los modelos
    models_dir = os.path.join(BUCKET_NAME, "resultados/modelos/")
    os.makedirs(models_dir, exist_ok=True)

    semillas = generar_semillas(SEMILLA[0], KSEMILLERIO)

    for i, seed in enumerate(semillas):
        logging.info(f'Entrenando modelo con seed = {seed} ({i+1}/{len(semillas)})')

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

    logging.info('=== Finaliza Entrenamiento de los modelos ===')

    return modelos


def generar_predicciones_finales(df: pl.DataFrame, envios: int, archivo_base: str = None) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.

    Args:
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        envios: cantidad de envios a realizar

    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info(f"Período de predicción: {FINAL_PREDICT}")
    df_predict = df.filter(pl.col("foto_mes").is_in(FINAL_PREDICT))

    logger.info(f"Registros de predicción: {df_predict.height:,}")
    X_predict = df_predict.drop(['target', 'target_test']).to_pandas()
    clientes_predict = df_predict['numero_de_cliente'].to_pandas()

    logger.info(f"Features utilizadas: {len(X_predict.columns)}")

    logger.info("Generando predicciones finales")

    if archivo_base is None:
        archivo_base = STUDY_NAME

    model_files = sorted(glob.glob(os.path.join(BUCKET_NAME, f"resultados/modelos/{archivo_base}_lgb_seed_*.txt")))
    if not model_files:
        mensaje_error = f"🔍 No se encontraron los modelos correspondientes al experimento '{archivo_base}'."
        raise FileNotFoundError(mensaje_error)

    preds = []

    for file in model_files:
        modelo = lgb.Booster(model_file=file)
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