import optuna
import lightgbm as lgb
# from lightgbm import early_stopping, log_evaluation
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import ganancia_evaluator

logger = logging.getLogger(__name__)


def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.

    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"

    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA[0],
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION,
            'undersampling': UNDERSAMPLING_FRACTION
        }
    }

    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)

    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación para el LIGHTGBM
    Utiliza configuración YAML para períodos y semilla.
    1. Define parametros para el modelo LightGBM
    2. Preparar dataset para entrenamiento y validación
    3. Entrena modelo con función de ganancia personalizada (CV)
    4. Ganancia promedio del CV
    5 .Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """

    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'num_threads': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,

        # 'first_metric_only': False,
        # 'boost_from_average': True,
        'feature_pre_filter': PARAMETROS_LGB['feature_pre_filter'],
        'force_row_wise': PARAMETROS_LGB['force_row_wise'],  # para reducir warnings
        # 'max_depth': -1,  # -1 significa no limitar,  por ahora lo dejo fijo
        # 'min_gain_to_split': 0,
        # 'min_sum_hessian_in_leaf': 0.001,
        # 'lambda_l1': 0.0,
        # 'lambda_l2': 0.0,
        'max_bin': PARAMETROS_LGB['max_bin'],
        # 'pos_bagging_fraction': 1,
        # 'neg_bagging_fraction': 1,
        # 'is_unbalance': False,
        # 'scale_pos_weight': 1,
        # 'extra_trees': False,
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'num_iterations': trial.suggest_int('num_iterations', PARAMETROS_LGB['num_iterations'][0], PARAMETROS_LGB['num_iterations'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1]),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'num_leaves': trial.suggest_int("num_leaves", PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', PARAMETROS_LGB['min_data_in_leaf'][0], PARAMETROS_LGB['min_data_in_leaf'][1]),
        'random_state': SEMILLA[0]
    }

    # MES_TRAIN puede ser un unico mes o una lista de meses
    if isinstance(MES_TRAIN, list):
        periodos_cv = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_cv = [MES_TRAIN, MES_VALIDACION]

    df_train = df[df['foto_mes'].isin(periodos_cv)]
    logging.info(df_train.shape)
    X_train = df_train.drop(['target', 'target_test'], axis=1)
    y_train = df_train['target']

    train_data = lgb.Dataset(X_train, label=y_train)

    cv_results = lgb.cv(
        params,
        train_data,
        feval=ganancia_evaluator,
        stratified=True,
        shuffle=True,
        nfold=5,
        seed=SEMILLA[0],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(0)]
    )

    ganancia_total = np.max(cv_results['valid ganancia-mean'])

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)

    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")

    return ganancia_total


def optimizar(df, n_trials=100, n_jobs=1) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)

    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado.
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0])
        # storage=storage_name,
        # load_if_exists=True,
    )

    study.optimize(lambda t: objetivo_ganancia(t, df), n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study

