import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import polars as pl
import logging
import json
import os
from datetime import datetime
import mlflow
from .config import *
from matplotlib import pyplot as plt
from .gain_function import calcular_ganancias_acumuladas
from .basic_functions import train_test_split
from .plots import plot_mean_importance
from .basic_functions import generar_semillas
import gc

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def evaluar_en_test(df, mejores_params) -> tuple:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia

    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna

    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")

    X_train, y_train, X_test, y_test, w_train, feature_name = train_test_split(df=df, undersampling=True, mes_train=MES_TRAIN, weight_train=WEIGHT_TRAIN, mes_test=MES_TEST)

    mlflow.log_param("X_train_shape", X_train.shape)
    logging.info(X_train.shape)
    logging.info(X_test.shape)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_name, free_raw_data=True, weight=w_train)

    flag_GPU = int(os.getenv('GPU', 0))

    gpu_dict = {'device': 'gpu'} if flag_GPU else {'device': 'cpu'}

    # Hiperparámetros fijos
    params = {
        'objective': 'binary',
        'metric': 'None',
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'n_threads': -1,
        **gpu_dict,
        'feature_pre_filter': PARAMETROS_LGB['feature_pre_filter'],
        'force_row_wise': PARAMETROS_LGB['force_row_wise'],  # para reducir warnings
        'max_bin': PARAMETROS_LGB['max_bin'],
        'seed': SEMILLA[0]
    }

    final_params = {**params, **mejores_params}
    logger.info(f"Parámetros del modelo: {final_params}")

    logging.info(f'=== Inicio Entrenamiento del Modelo con {KSEMILLERIO} semillas ===')

    modelos = []
    preds = []
    pred_acumulada = np.zeros(len(y_test))
    all_importances = []
    importance_type = 'gain'

    semillas = generar_semillas(SEMILLA[0], KSEMILLERIO)

    for i, seed in enumerate(semillas):
        logging.info(f'Entrenando modelo con seed = {seed} ({i+1}/{len(semillas)})')

        # Copia de parámetros con la semilla actual
        params_seed = final_params.copy()
        params_seed['seed'] = seed

        # Entrenamiento
        modelo = lgb.train(
            params_seed,
            train_data,
            num_boost_round=mejores_params.get('num_iterations', 1000)
        )

        logging.info(f'Fin de entrenamiento del modelo con seed = {seed} ({i + 1}/{len(semillas)})')

        modelos.append(modelo)
        y_pred_actual = modelo.predict(X_test)
        preds.append(y_pred_actual)
        pred_acumulada += y_pred_actual
        y_pred_promedio_parcial = pred_acumulada / (i + 1)

        ganancias_parcial = calcular_ganancias_acumuladas(y_test, y_pred_promedio_parcial)
        max_ganancia_parcial = np.max(ganancias_parcial)
        logging.info(f'Ganancias parcial: {max_ganancia_parcial}')
        mlflow.log_metric("ganancia_parcial", max_ganancia_parcial, step=i)  # loggear la ganancia parcial en mlflow

        # Generamos un DataFrame temporal con la importancia de este modelo
        feature_imp = pd.DataFrame({
            'feature': modelo.feature_name(),
            'importance': modelo.feature_importance(importance_type=importance_type)
        })
        all_importances.append(feature_imp)

        # Memory cleanup after each model
        del feature_imp, params_seed
        gc.collect()

    logging.info('=== Finaliza Entrenamiento de los modelos ===')

    # Ensemble: promedio de predicciones
    y_pred = pred_acumulada / len(semillas)

    auc = roc_auc_score(y_test, y_pred)

    path_resultados = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(path_resultados, exist_ok=True)
    ruta_probabilidades = os.path.join(path_resultados, f"{STUDY_NAME}_probabilidades.csv")
    ndc_index = feature_name.index('numero_de_cliente')
    numero_de_cliente = X_test[:, ndc_index].astype(int)

    df_probabilidades = pd.DataFrame(
        {
            'numero_de_cliente': numero_de_cliente,
            'probabilidad': y_pred,
            'target': y_test
        }
    )

    df_probabilidades.to_csv(ruta_probabilidades, index=False)
    logger.info(f'Probabilidades guardadas en: {ruta_probabilidades}')

    logging.info('=== Inicio Grafico de Importancia ===')
    plot_mean_importance(all_importances, importance_type, type='test')

    logging.info('=== Inicio Calculo de Ganancias Acumuladas en Test ===')

    # Calcular la ganancia de cada modelo individual
    ganancias_acumuladas = []
    for i, pred_semilla in enumerate(preds):
        ganancia = calcular_ganancias_acumuladas(y_test, pred_semilla)
        ganancias_acumuladas.append(ganancia)

    # Calculo la ganancia del ensamble
    ganancia_ensamble = calcular_ganancias_acumuladas(y_test, y_pred)
    ganancias_acumuladas.append(ganancia_ensamble)

    ganancia_ensamble_meseta = (
        pd.Series(ganancia_ensamble)
        .rolling(window=1001, center=True, min_periods=1)
        .mean()
    ).max(skipna=True)

    # Estadísticas básicas del ensamble
    ganancia_test = np.max(ganancia_ensamble)
    total_predicciones = len(ganancia_ensamble)
    predicciones_positivas = np.argmax(ganancia_ensamble)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    resultados = {
        'ganancia_test': float(ganancia_test),
        'ganancia_meseta_test': float(ganancia_ensamble_meseta),
        'auc_test': float(auc),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'params': mejores_params
    }
    logging.info('=== Finaliza Calculo de Ganancias Acumuladas en Test ===')

    logger.info("=== INICIANDO GENERACION DE GRAFICO DE TEST")
    ruta_grafico = crear_grafico_ganancia(y_pred, ganancia_ensamble)
    ruta_grafico_multiple = crear_grafico_multiple_ganancia(ganancias_acumuladas)
    mlflow.log_artifact(ruta_grafico_multiple)
    logger.info("=== GRAFICO DE TEST COMPLETADO")

    ## Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados['ganancia_test']:,.4f}")
    logger.info(f"🎯 Predicciones positivas: {resultados['predicciones_positivas']:,} ({resultados['porcentaje_positivas']:.2f}%)")

    return resultados, y_pred, ganancias_acumuladas

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.

    Args:
        resultados_test: resultados del entrenamiento en test
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    path_resultados = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(path_resultados, exist_ok=True)
    # Nombre del archivo único para todas las iteraciones
    archivo = os.path.join(path_resultados, f"{archivo_base}_test_results.json")

    # Datos del resultado en test
    test_data = {
        'descripcion_experimento': DESCRIPCION,
        'ksemillerio': KSEMILLERIO,
        'ganancia': resultados_test['ganancia_test'],
        'ganancia_meseta': resultados_test['ganancia_meseta_test'],
        'auc_BAJA+2': resultados_test['auc_test'],
        'total_predicciones': resultados_test['total_predicciones'],
        'predicciones_positivas': resultados_test['predicciones_positivas'],
        'porcentaje_positivas': resultados_test['porcentaje_positivas'],
        'params_study': STUDY_HP,
        'params': resultados_test['params'],
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',
        'configuracion': {
            'semilla': SEMILLA[0],
            'mes_train': MES_TRAIN,
            'mes_test': MES_TEST
        }
    }

    mlflow.log_metric("ganancia_test", resultados_test['ganancia_test'])
    mlflow.log_metric("ganancia_meseta_test", resultados_test['ganancia_meseta_test'])
    mlflow.log_metric("mes_test", MES_TEST[0])
    mlflow.log_metric("undersampling", UNDERSAMPLING_FRACTION)
    mlflow.log_metric("semillerio_test", KSEMILLERIO)
    mlflow.log_metric("auc_test", resultados_test['auc_test'])
    mlflow.log_metric("envios_test", resultados_test['predicciones_positivas'])

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


def crear_grafico_multiple_ganancia(ganancias_acumuladas: np.array) -> str:
    """
    Genera un gráfico con la ganancia acumulada de cada semilla y su ensamble y lo guarda como JPG.

    :param ganancias_acumuladas: Lista de ganancias acumuladas
    :return: ruta del archivo de salida
    """

    path_resultados = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(path_resultados, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ruta_archivo = os.path.join(path_resultados, f"{STUDY_NAME}_grafico_multiple_test_{timestamp}.jpg")

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, ganancia in enumerate(ganancias_acumuladas):
        ganancia_maxima = np.max(ganancia)
        indice_maximo = np.argmax(ganancia)

        # umbral para filtrar el grafico
        umbral_ganancia = ganancia_maxima * 0.66
        indices_filtrados = ganancia >= umbral_ganancia
        x_filtrado = np.where(indices_filtrados)[0]
        y_filtrado = ganancia[indices_filtrados]

        if i == len(ganancias_acumuladas) - 1:
            ax.plot(x_filtrado, y_filtrado, color='blue', linewidth=2.5, label=f'Ganancia Acumulada Ensamble')
            ax.scatter(indice_maximo, ganancia_maxima, color='red', s=100, zorder=5, label='Ganancia Máxima Ensamble')
            ax.annotate(f'Ganancia Máxima\n{ganancia_maxima:,.0f}',
                         xy=(indice_maximo, ganancia_maxima),
                         xytext=(indice_maximo + len(x_filtrado) * 0.1, ganancia_maxima * 1.05),
                         arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                         fontsize=10, fontweight='bold', color='red',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round, pad=0.3'))
        else:
            label = 'Modelos Individuales' if i == 0 else ""
            ax.plot(x_filtrado, y_filtrado, color='grey', linewidth=1, label=label, alpha=0.6)

    ax.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax.set_ylabel('Ganancia Acumulada', fontsize=12)
    ax.set_title(f'Ganancia acumulada por orden de predicción (filtrada) - {STUDY_NAME}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Formatear los ejes
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Ajustar y guardar la figura final
    plt.tight_layout()
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f'Archivo guardado: {ruta_archivo}')
    logger.info('Estadísticas del gráfico:')
    logger.info(f'  - Ganancia máxima: {ganancia_maxima:,.0f}')
    logger.info(f'  - Corte ideal por cliente: {indice_maximo:,.0f}')

    return ruta_archivo

def crear_grafico_ganancia(y_pred_proba: np.array, ganancias_acumuladas: np.array) -> str:
    """
    Genera un gráfico de la ganancia acumulada en test y lo guarda como JPG.

    :param y_pred_proba: Probabilidades predichas
    :param ganancias_acumuladas: Vector con ganancias acumuladas
    :return: ruta del archivo de salida
    """

    path_resultados = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(path_resultados, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ruta_probabilidades = os.path.join(path_resultados, f"{STUDY_NAME}_probabilidades_{timestamp}.csv")

    df_probabilidades = pd.DataFrame(
        {
            'probabilidad': y_pred_proba,
            'ganancia_acumulada': ganancias_acumuladas,
            'cliente_ordenado': range(len(y_pred_proba))
        }
    )

    df_probabilidades.to_csv(ruta_probabilidades, index=False)
    logger.info(f'Probabilidades guardadas en: {ruta_probabilidades}')

    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)

    # umbral para filtrar el grafico
    umbral_ganancia = ganancia_maxima * 0.66

    indices_filtrados = ganancias_acumuladas >= umbral_ganancia
    x_filtrado = np.where(indices_filtrados)[0]
    y_filtrado = ganancias_acumuladas[indices_filtrados]

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 8))

    plt.plot(x_filtrado, y_filtrado, color='blue', linewidth=2.5, label='Ganancia Acumulada')

    plt.scatter(indice_maximo, ganancia_maxima, color='red', s=100, zorder=5, label='Ganancia Máxima')

    plt.annotate(f'Ganancia Máxima\n{ganancia_maxima:,.0f}',
                 xy=(indice_maximo, ganancia_maxima),
                 xytext=(indice_maximo + len(x_filtrado) * 0.1, ganancia_maxima * 1.05),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                 fontsize=10, fontweight='bold', color='red',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round, pad=0.3'))

    plt.xlabel('Clientes ordenados por probabilidad', fontsize=12)
    plt.ylabel('Ganancia Acumulada', fontsize=12)
    plt.title(f'Ganancia acumulada por orden de predicción (filtrada) - {STUDY_NAME}',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # ✅ corregido: usar el valor x en el formato
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()

    ruta_archivo = os.path.join(path_resultados, f"{STUDY_NAME}_grafico_test_{timestamp}.jpg")

    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f'Archivo guardado: {ruta_archivo}')
    logger.info('Estadísticas del gráfico:')
    logger.info(f'  - Ganancia máxima: {ganancia_maxima:,.0f}')
    logger.info(f'  - Corte ideal por cliente: {indice_maximo:,.0f}')

    return ruta_archivo


