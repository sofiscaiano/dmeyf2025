import pandas as pd
from datetime import datetime
import os
import logging

from src.features import feature_engineering_lag
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.optimization import optimizar
from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.config import *
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.create_target import create_target

# config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/{nombre_log}', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

def main():
    print(">>> Inicio de ejecucion")

    ## Creacion de target
    # create_target()

    ## Carga de Datos
    os.makedirs('data', exist_ok=True)
    path = 'data/competencia_01.csv'
    df = cargar_datos(path)

    ## Feature Engineering
    atributos = list(df.drop(columns=['foto_mes', 'target']).columns)
    cant_lag = 2
    df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
    logger.info(f'Feature Engineering completado: {df.shape}')

    ## Convertir clase ternaria a target binaria
    df = convertir_clase_ternaria_a_target(df)

    ## Ejecutar optimizacion de hiperparametros
    study = optimizar(df, n_trials = 100)

    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
    logger.info(f'Mejores Hiperparametros: {study.best_params}')
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    mejores_params = cargar_mejores_hiperparametros()
    resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df, mejores_params)

    # Guardar resultados de test
    guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)

    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    # Entrenar modelo final
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df)
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)

    # Guardar modelo joblib

    # Generar predicciones
    envios = resultados_test['predicciones_positivas']
    predicciones = generar_predicciones_finales(modelo_final, X_predict, clientes_predict, envios)

    # Guardar predicciones
    salida_kaggle = guardar_predicciones_finales(predicciones)

    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDICT}")
    logger.info(f"📁 Archivo de salida: {salida_kaggle}")
    logger.info(f"📝 Log detallado: logs/{nombre_log}")

    logger.info(f'Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}')

if __name__ == '__main__':
    main()

