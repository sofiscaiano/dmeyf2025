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


def guardar_iteracion(trial, metrica, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.

    Args:
        trial: Trial de Optuna
        metrica: Valor de ganancia o auc obtenido
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
        'user_attrs': trial.user_attrs,
        'value': float(metrica),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA[0],
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION,
            'undersampling': UNDERSAMPLING_FRACTION,
            'metric': PARAMETROS_LGB['metric']
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
    logger.info(f"Ganancia/auc: {metrica:,.4f}" + "---" + "Parámetros: {params}")

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
    3. Entrena modelo con función de ganancia personalizada (CV) o AUC
    4. Ganancia promedio del CV
    5 .Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """

    flag_GPU = int(os.getenv('GPU', 0))

    if flag_GPU == 0:
        gpu_dict = {'device': 'cpu'}
    else:
        gpu_dict = {'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0}

    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        # 'metric': 'None',  # Usamos nuestra métrica personalizada
        'metric': PARAMETROS_LGB['metric'],
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'num_threads': -1,
        **gpu_dict,
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
        'seed': SEMILLA[0]
    }

    # MES_TRAIN puede ser un unico mes o una lista de meses
    if isinstance(MES_TRAIN, list):
        periodos_train = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_train = [MES_TRAIN, MES_VALIDACION]

    # if isinstance(MES_VALIDACION, list):
    #     periodos_val = MES_VALIDACION
    # else:
    #     periodos_val = [MES_VALIDACION]

    df_train = df[df['foto_mes'].isin(periodos_train)]
    # df_val = df[df['foto_mes'].isin(periodos_val)]
    logging.info(df_train.shape)
    # logging.info(df_val.shape)
    X_train = df_train.drop(['target', 'target_test'], axis=1)
    y_train = df_train['target']
    # X_val = df_val.drop(['target', 'target_test'], axis=1)
    # y_val = df_val['target']

    train_data = lgb.Dataset(X_train, label=y_train)
    # val_data = lgb.Dataset(X_val, label=y_val)

    logger.debug(f"Iniciando CV de trial:{trial.number}")
    cv_results = lgb.cv(
        params,
        train_data,
        # feval=ganancia_evaluator,
        stratified=True,
        shuffle=True,
        nfold=5,
        seed=SEMILLA[0],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(0)]
    )

    # Entrenamiento
    # modelo = lgb.train(
    #     params,
    #     train_data,
    #     feval=ganancia_evaluator,
    #     seed=SEMILLA[0],
    #     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(0)]
    # )

    # 1. Determina la métrica y su valor
    if PARAMETROS_LGB['metric'] == 'auc':
        metrica_key = 'valid auc-mean'
    else:
        metrica_key = 'valid ganancia-mean'

    metrica = np.max(cv_results[metrica_key])
    best_num_iterations_cv = len(cv_results[metrica_key])

    # 2. Guarda esta información en el trial para recuperarla después.
    trial.set_user_attr('num_iterations', best_num_iterations_cv)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, metrica)

    logger.debug(f"Trial {trial.number}: Ganancia/AUC = {metrica:,.4f}")

    return metrica


def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.

    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad

    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME

    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)

    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"

    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)

            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")

            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")

            return study

        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")

    # Crear nuevo estudio
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0]),
        storage=storage,
    )

    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")

    return study

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

    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)

    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")

    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        study.optimize(lambda t: objetivo_ganancia(t, df), n_trials=trials_a_ejecutar, show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)

        # Generar el gráfico
        fig_importancia = optuna.visualization.plot_param_importances(study)
        fig_importancia.write_html(f"resultados/{STUDY_NAME}_importancia_parametros.html")

        fig_contour = optuna.visualization.plot_contour(study, params=['num_leaves', 'min_data_in_leaf'])
        fig_contour.write_html(f"resultados/{STUDY_NAME}_contour.html")

        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_html(f"resultados/{STUDY_NAME}_slice.html")

        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")

    return study

