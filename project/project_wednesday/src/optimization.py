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
from .gain_function import calcular_ganancia, ganancia_lgb_binary

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
            'mes_validacion': MES_VALIDACION
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

    num_leaves = trial.suggest_int('num_leaves', 8, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3) # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 700)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)
    num_iterations = trial.suggest_int('num_iterations', 500, 1500)

    # Hiperparámetros a optimizar
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
        'max_bin': 31,  # lo debo dejar fijo, no participa de la BO
        'bagging_fraction': bagging_fraction,
        'pos_bagging_fraction': 1,
        'neg_bagging_fraction': 1,
        'is_unbalance': False,
        'scale_pos_weight': 1,
        'extra_trees': False,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'random_state': SEMILLA[0]  # Desde configuración YAML
    }

    # MES_TRAIN puede ser un unico mes o una lista de meses
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]

    # train_data = df[df['foto_mes'] == MES_TRAIN]
    # val_data = df[df['foto_mes'] == MES_VALIDACION]
    # test_data = df[df['foto_mes'] == MES_TEST]

    X_train = df_train.drop(['target', 'foto_mes'], axis=1)
    # print(X_train.shape)
    y_train = df_train['target']
    # print(y_train.value_counts())

    train_data = lgb.Dataset(X_train, label=y_train)

    # X_test = test_data.drop(['target'], axis=1)
    # y_test = test_data['target']

    # X_val = val_data.drop(['target'], axis=1)
    # y_val = val_data['target']

    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=num_iterations,
        # early_stopping_rounds= int(50 + 5 / learning_rate),
        feval=ganancia_lgb_binary,
        stratified=True,
        shuffle=True,
        nfold=5,
        seed=SEMILLA[0],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)]
    )

    ganancia_total = max(cv_results['valid ganancia-mean'])

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)

    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")

    return ganancia_total



def optimizar(df, n_trials=100) -> optuna.Study:
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

    study.optimize(lambda t: objetivo_ganancia(t, df), n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study


def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.

    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna

    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")

    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    X_train = df_train_completo.drop(['target', 'foto_mes'], axis=1)
    y_train = df_train_completo['target']

    X_test = df_test.drop(['target', 'foto_mes'], axis=1)
    y_test = df_test['target']

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Hiperparámetros a optimizar
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
        'max_bin': 31,  # lo debo dejar fijo, no participa de la BO
        'pos_bagging_fraction': 1,
        'neg_bagging_fraction': 1,
        'is_unbalance': False,
        'scale_pos_weight': 1,
        'extra_trees': False,
        'random_state': SEMILLA[0]  # Desde configuración YAML
    }

    final_params = {**params, **mejores_params}

    # Entrenar modelo con mejores parámetros
    modelo = lgb.train(final_params,
                      train_data,
                      num_boost_round = mejores_params.get('num_iterations', 1000)
                       )
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()

    y_pred = modelo.predict(X_test)
    y_pred_binary = (y_pred > 0.025).astype(int)

    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'params': mejores_params
    }

    return resultados

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.

    Args:
        resultados_test: resultados del entrenamiento en test
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_test_results.json"

    # Datos del resultado en test
    test_data = {
        'ganancia': resultados_test['ganancia_test'],
        'total_predicciones': resultados_test['total_predicciones'],
        'predicciones_positivas': resultados_test['predicciones_positivas'],
        'porcentaje_positivas': resultados_test['porcentaje_positivas'],
        'params': resultados_test['params'],
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el entrenamiento se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA[0],
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_TEST
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
    datos_existentes.append(test_data)

    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Test guardada en {archivo}")
    logger.info(f"Ganancia: {resultados_test['ganancia_test']:,.0f}")