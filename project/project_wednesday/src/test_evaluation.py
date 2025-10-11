import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from matplotlib import pyplot as plt
from .gain_function import ganancia_evaluator, calcular_ganancias_acumuladas
from .plots import plot_mean_importance
import gc

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def evaluar_en_test(df, mejores_params) -> dict:
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

    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    X_train = df_train_completo.drop(['target', 'target_test'], axis=1)
    y_train = df_train_completo['target']

    X_test = df_test.drop(['target', 'target_test'], axis=1)
    y_test = df_test['target_test']

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    del X_train, y_train, df_train_completo, df_test
    gc.collect()

    flag_GPU = int(os.getenv('GPU', 0))

    if flag_GPU == 0:
        gpu_dict = {'device': 'cpu'}
    else:
        gpu_dict = {'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0}

    # Hiperparámetros fijos
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
        # 'random_state': SEMILLA[0],
        'seed': SEMILLA[0]
        # 'data_random_seed': SEMILLA[0],
        # 'feature_fraction_seed': SEMILLA[0]
    }

    final_params = {**params, **mejores_params}

    logger.info(f"Parámetros del modelo: {final_params}")

    logging.info('=== Inicio Entrenamiento del Modelo con 5 semillas ===')

    modelos = []
    preds = []
    all_importances = []
    importance_type = 'gain'  # O 'split'

    for i, seed in enumerate(SEMILLA):
        logging.info(f'Entrenando modelo con seed = {seed} ({i+1}/{len(SEMILLA)})')

        # Copia de parámetros con la semilla actual
        params_seed = final_params.copy()
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

        modelos.append(modelo)
        preds.append(modelo.predict(X_test))

        # Memory cleanup after each model
        del feature_imp, params_seed
        gc.collect()

    logging.info('=== Finaliza Entrenamiento de los 5 Modelos ===')

    # Ensemble: promedio de predicciones
    y_pred = np.mean(preds, axis=0)

    logging.info('=== Inicio Grafico de Importancia ===')
    plot_mean_importance(all_importances, importance_type, type='test')

    logging.info('=== Inicio Calculo de Ganancias Acumuladas en Test ===')
    # Calcular solo la ganancia
    ganancias_acumuladas = calcular_ganancias_acumuladas(y_test, y_pred)

    # Estadísticas básicas
    ganancia_test = np.max(ganancias_acumuladas)
    total_predicciones = len(ganancias_acumuladas)
    predicciones_positivas = np.argmax(ganancias_acumuladas)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'params': mejores_params
    }
    logging.info('=== Finaliza Calculo de Ganancias Acumuladas en Test ===')


    logger.info("=== INICIANDO GENERACION DE GRAFICO DE TEST")
    ruta_grafico = crear_grafico_ganancia_test(y_pred, ganancias_acumuladas)
    logger.info("=== GRAFICO DE TEST COMPLETADO")

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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime

def crear_grafico_ganancia_test(y_pred_proba: np.array, ganancias_acumuladas: np.array) -> str:
    """
    Genera un gráfico de la ganancia acumulada en test y lo guarda como JPG.

    :param y_pred_proba: Probabilidades predichas
    :param ganancias_acumuladas: Vector con ganancias acumuladas
    :return: ruta del archivo de salida
    """

    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ruta_probabilidades = f"resultados/{STUDY_NAME}_probabilidades_{timestamp}.csv"

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

    # umbral_probabilidad = 0.025
    # clientes_sobre_umbral = np.sum(y_pred_proba >= umbral_probabilidad)

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

    ruta_archivo = f'resultados/{STUDY_NAME}_grafico_test_{timestamp}.jpg'

    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f'Archivo guardado: {ruta_archivo}')
    logger.info('Estadísticas del gráfico:')
    logger.info(f'  - Ganancia máxima: {ganancia_maxima:,.0f}')
    logger.info(f'  - Corte ideal por cliente: {indice_maximo:,.0f}')

    return ruta_archivo

