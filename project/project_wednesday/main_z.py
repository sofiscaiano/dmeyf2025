import pandas as pd
from datetime import datetime
import os
import logging
import argparse
import numpy as np
import gc
import polars as pl
import mlflow

from src.features import undersample, feature_engineering_lag, generar_reporte_mensual_html, fix_aguinaldo, feature_engineering_delta, feature_engineering_rank, feature_engineering_trend, fix_zero_sd, create_canaritos
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.optimization import optimizar
from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.config import *
from src.best_params import cargar_mejores_hiperparametros, cargar_mejores_envios
from src.final_training import generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.create_target import create_target
from src.zeroshot import optimizar_zero_shot

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
logger.info(f"MES_TRAIN_BO: {MES_TRAIN_BO}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"KSEMILLERIO_BO: {KSEMILLERIO_BO}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"KSEMILLERIO: {KSEMILLERIO}")
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

    # Configurar MLflow y ejecutar pipeline completo
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow configurado con servidor remoto: {MLFLOW_TRACKING_URI}")

    except Exception as e:
        logger.error(f"Error al configurar MLflow: {e}")
        raise

    def cargar_y_procesar_df():

        data_path = os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")
        months_filter = list(set(MES_TRAIN + MES_VALIDACION + MES_TEST + FINAL_TRAIN + FINAL_PREDICT))
        df = cargar_datos(data_path, lazy=True, months=months_filter)
        logging.info("Elimino atributos:")
        df = df.drop([c for c in df.columns if any(c.startswith(p) for p in DROP)])
        df = create_canaritos(df, qcanaritos=PARAMETROS_ZLGB['qcanaritos'])

        df_train = df.filter(pl.col("foto_mes").is_in(MES_TRAIN))
        df_train = undersample(df_train, sample_fraction=UNDERSAMPLING_FRACTION)
        df_test = df.filter(pl.col("foto_mes").is_in(MES_TEST))

        return df_train, df_test


    with mlflow.start_run(run_name=f"experimento-{STUDY_NAME}"):
        mlflow.set_tags(MLFLOW_TAGS)

        df_train, df_test = cargar_y_procesar_df()
        mejores_params = cargar_mejores_hiperparametros()

        resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df_train, df_test, mejores_params)
        guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)
        # entrenar_modelo_final(df, mejores_params)

        # ## Generar predicciones
        # if ENVIOS is not None:
        #     envios = ENVIOS
        #     logger.info(f"Envios: {envios}")
        # else:
        #     envios = cargar_mejores_envios()
        #
        # predicciones = generar_predicciones_finales(df, envios)
        # salida_kaggle = guardar_predicciones_finales(predicciones)

        ## Resumen final
        logger.info("=== RESUMEN FINAL ===")
        logger.info(f"✅ Entrenamiento final completado exitosamente")
        logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
        # logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
        # logger.info(f"🔮 Período de predicción: {FINAL_PREDICT}")
        # logger.info(f"📁 Archivo de salida: {salida_kaggle}")
        logger.info(f"📝 Log detallado: log/{nombre_log}")

        logger.info(f'Ejecucion finalizada. Revisar log para mas detalles. {nombre_log}')

if __name__ == '__main__':
    main()

