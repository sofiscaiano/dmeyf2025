import pandas as pd
from datetime import datetime
import os
import logging
import argparse
import numpy as np
import gc
import polars as pl

from src.features import feature_engineering_lag, undersample, generar_reporte_mensual_html, fix_aguinaldo, feature_engineering_delta, feature_engineering_rank, feature_engineering_trend
from src.loader import cargar_datos_csv, cargar_datos, convertir_clase_ternaria_a_target, reduce_mem_usage
from src.optimization import optimizar
from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.config import *
from src.best_params import cargar_mejores_hiperparametros, cargar_mejores_envios
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.create_target import create_target

# config basico logging
path_logs = os.path.join(BUCKET_NAME, "log")
os.makedirs(path_logs, exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"
log_path = os.path.join(path_logs, nombre_log)
logger.info(f"Guardo archivo de logs en {log_path}")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
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
logger.info(f"UNDERSAMPLING_FRACTION: {UNDERSAMPLING_FRACTION}")
logger.info(f"UNDERSAMPLING_FINAL_TRAINING: {UNDERSAMPLING_FINAL_TRAINING}")
logger.info(f"METRIC: {PARAMETROS_LGB['metric']}")
logger.info(f"DROP FEATURES: {DROP}")
logger.info(f"CANTIDAD DE ENVIOS: {ENVIOS}")
logger.info(f"DESCRIPCION DE EXPERIMENTO: {DESCRIPCION}")

def main():

    parser = argparse.ArgumentParser(description="Entrenamiento con Optuna")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Cantidad de trials para Optuna"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Jobs paralelos para Optuna"
    )

    args = parser.parse_args()

    print(">>> Inicio de ejecucion")

    ## Creacion de target
    # crudo_path = os.path.join(BUCKET_NAME, "datasets/competencia_02_crudo.csv.gz")
    # df = cargar_datos_csv(crudo_path)
    # df = create_target(df=df)

    if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")):
        logger.info("✅ df_fe encontrado")
        data_path = os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")
        months_filter = list(set(MES_TRAIN + MES_VALIDACION + MES_TEST + FINAL_TRAIN + FINAL_PREDICT))
        df = cargar_datos(data_path, lazy=True, months=months_filter)

    else:
        ## Carga de Datos
        logger.info("❌ df_fe no encontrado")
        os.makedirs(f'{BUCKET_NAME}/datasets', exist_ok=True)
        data_path = os.path.join(BUCKET_NAME, DATA_PATH)
        df = cargar_datos(data_path, lazy=False)

        ## Reporte HTML de evolucion de features
        # generar_reporte_mensual_html(df, nombre_archivo='reporte_evolucion_features.html')

        ## Feature Engineering
        atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'numero_de_cliente']]
        cant_lag = 2
        # Fix aguinaldo
        # df = fix_aguinaldo(df)
        # gc.collect()
        # Lag features
        # df = feature_engineering_trend(df, columnas=['ctrx_quarter', 'mpayroll', 'mcaja_ahorro', 'mcuenta_corriente', 'mcuentas_saldo'])
        # df = feature_engineering_rank(df, columnas=atributos) # pandas
        gc.collect()
        df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag) # duckdb
        gc.collect()
        # Delta features
        df = feature_engineering_delta(df, columnas=atributos, cant_lag=cant_lag) # polars
        gc.collect()

        ## Convertir clase ternaria a target binaria
        df = convertir_clase_ternaria_a_target(df)
        data_path = os.path.join(BUCKET_NAME, "datasets", "df_fe.parquet")
        df.write_parquet(data_path, compression="gzip")

    df = df.to_pandas()
    df = reduce_mem_usage(df)
    cols_to_drop = [col for col in df.columns for prefix in DROP if col.startswith(prefix)]
    df.drop(columns=cols_to_drop, inplace=True)
    gc.collect()

    # Realizo undersampling de la clase mayoritaria para agilizar la optimizacion
    reduced_df = undersample(df, UNDERSAMPLING_FRACTION)

    # ## Ejecutar optimizacion de hiperparametros
    # study = optimizar(reduced_df, n_trials = args.n_trials, n_jobs = args.n_jobs)
    #
    # ## 5. Análisis adicional
    # logger.info("=== ANÁLISIS DE RESULTADOS ===")
    # trials_df = study.trials_dataframe()
    # if len(trials_df) > 0:
    #     top_5 = trials_df.nlargest(5, 'value')
    #     logger.info("Top 5 mejores trials:")
    #     for idx, trial in top_5.iterrows():
    #         logger.info(f"  Trial {trial['number']}: {trial['value']:,.4f}")
    # logger.info(f'Mejores Hiperparametros: {study.best_params}')
    # logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    mejores_params = cargar_mejores_hiperparametros('lgb_optimization_competencia27')
    if UNDERSAMPLING_FINAL_TRAINING:
        resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(reduced_df, mejores_params)
        guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)
        entrenar_modelo_final(reduced_df, mejores_params)
    else:
        resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df, mejores_params)
        guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)
        entrenar_modelo_final(df, mejores_params)

    ## Generar predicciones
    if ENVIOS is not None:
        envios = ENVIOS
    else:
        envios = cargar_mejores_envios()

    predicciones = generar_predicciones_finales(df, envios)
    salida_kaggle = guardar_predicciones_finales(predicciones)

    ## Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    # logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDICT}")
    logger.info(f"📁 Archivo de salida: {salida_kaggle}")
    logger.info(f"📝 Log detallado: log/{nombre_log}")

    logger.info(f'Ejecucion finalizada. Revisar log para mas detalles. {nombre_log}')

if __name__ == '__main__':
    main()

