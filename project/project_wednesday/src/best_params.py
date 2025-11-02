import json
import numpy as np
import logging
from .config import *

logger = logging.getLogger(__name__)


def cargar_mejores_hiperparametros(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna o carga hiperparametros fijados en config ADHOC

    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)

    Returns:
        dict: Mejores hiperparámetros encontrados o ADHOC
    """

    if ADHOC:
        mejores_params = {
            'num_iterations': PARAMETROS_LGB_ADHOC['num_iterations'],
            'learning_rate': PARAMETROS_LGB_ADHOC['learning_rate'],
            'num_leaves': PARAMETROS_LGB_ADHOC['num_leaves'],
            'min_data_in_leaf': PARAMETROS_LGB_ADHOC['min_data_in_leaf'],
            'feature_fraction': PARAMETROS_LGB_ADHOC['feature_fraction'],
	        'bagging_fraction': PARAMETROS_LGB_ADHOC['bagging_fraction']
        }

        return mejores_params

    else:
        if archivo_base is None:
            archivo_base = STUDY_NAME

        archivo = f"resultados/{archivo_base}_iteraciones.json"

        try:
            with open(archivo, 'r') as f:
                iteraciones = json.load(f)

            if not iteraciones:
                raise ValueError("No se encontraron iteraciones en el archivo")

            # Encontrar la iteración con mayor ganancia
            mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
            mejores_params = mejor_iteracion['params']
            best_num_iterations = mejor_iteracion['user_attrs']['num_iterations'] # si hubo early_stopping num_iterations puede ser menor
            mejores_params['num_iterations'] = best_num_iterations
            if not UNDERSAMPLING_FINAL_TRAINING:
                mejores_params['min_data_in_leaf'] = round(mejores_params['min_data_in_leaf'] / UNDERSAMPLING_FRACTION)
            mejor_ganancia = mejor_iteracion['value']

            logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
            logger.info(f"Mejor ganancia/auc encontrada: {mejor_ganancia:,.4f}")
            logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
            logger.info(f"Parámetros: {mejores_params}")

            return mejores_params

        except FileNotFoundError:
            logger.error(f"No se encontró el archivo {archivo}")
            logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
            raise
        except Exception as e:
            logger.error(f"Error al cargar mejores hiperparámetros: {e}")
            raise

def cargar_mejores_envios(archivo_base: str = None) -> int:
    """
    Carga los mejores envios desde el archivo JSON de resultados de test.

    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)

    Returns:
        int: Mejores envios
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    archivo = f"resultados/{archivo_base}_test_results.json"

    try:
        with open(archivo, 'r') as f:
            resultados = json.load(f)

        if not resultados:
            raise ValueError("No se encontraron resultados en el archivo")

        # Encontrar la iteración con mayor ganancia
        envios = resultados[0]["predicciones_positivas"]
        ganancia = resultados[0]["ganancia"]

        logger.info(f"Mejores envios cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada en test: {ganancia:,.0f}")
        logger.info(f"Envios: {envios}")

        return envios

    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la evaluacion en test primero.")
        raise
    except Exception as e:
        logger.error(f"Error al cargar los mejores envios: {e}")
        raise

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna.

    Args:
        archivo_base: Nombre base del archivo

    Returns:
        dict: Estadísticas de la optimización
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    archivo = f"resultados/{archivo_base}_iteraciones.json"

    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)

        ganancias = [iter['value'] for iter in iteraciones]

        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_promedio': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }

        logger.info("Estadísticas de optimización:")
        logger.info(f"  Total trials: {estadisticas['total_trials']}")
        logger.info(f"  Mejor ganancia: {estadisticas['mejor_ganancia']:,.0f}")
        logger.info(f"  Ganancia promedio: {estadisticas['ganancia_promedio']:,.0f}")

        return estadisticas

    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise
