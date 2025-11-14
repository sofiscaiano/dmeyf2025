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

    with mlflow.start_run(run_name=f"experimento-{STUDY_NAME}"):
        mlflow.set_tags(MLFLOW_TAGS)

        if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")):
            logger.info("✅ df_fe encontrado")
            data_path = os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")
            if FLAG_GCP == 1:
                data_path = '~/datasets/df_fe.parquet'
            months_filter = list(set(MES_TRAIN + MES_VALIDACION + MES_TEST + FINAL_TRAIN + FINAL_PREDICT))
            df = cargar_datos(data_path, lazy=True, months=months_filter)
            gc.collect()

        else:
            ## Carga de Datos
            logger.info("❌ df_fe no encontrado")
            os.makedirs(f'{BUCKET_NAME}/datasets', exist_ok=True)
            data_path = os.path.join(BUCKET_NAME, DATA_PATH)
            df = cargar_datos(data_path, lazy=False)

            ## Feature Engineering
            atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'numero_de_cliente']]
            atributos_monetarios = [c for c in df.columns if any(c.startswith(p) for p in ['m', 'Visa_m', 'Master_m'])]
            cant_lag = 2
            ## Fix aguinaldo
            # df = fix_aguinaldo(df)
            gc.collect()

            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos.html')

            df = fix_zero_sd(df, columnas=atributos)
            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos_after_data_quality.html')

            df = feature_engineering_rank(df, columnas=atributos_monetarios) # pandas
            df = feature_engineering_trend(df, columnas=atributos, q=3)
            df = feature_engineering_trend(df, columnas=atributos, q=6)
            # mlflow.log_param("q_trend", '3 y 6m')

            gc.collect()
            df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag) # duckdb
            gc.collect()
            df = feature_engineering_delta(df, columnas=atributos, cant_lag=cant_lag) # polars
            gc.collect()

            ## Convertir clase ternaria a target binaria
            df = convertir_clase_ternaria_a_target(df)
            logging.info("==== Exporto el df_fe.parquet ====")
            data_path = os.path.join(BUCKET_NAME, "datasets", "df_fe.parquet")
            if FLAG_GCP == 1:
                data_path = '~/datasets/df_fe.parquet'
            df.write_parquet(data_path, compression="gzip")

        # df = df.to_pandas()
        # df = reduce_mem_usage(df)
        # Si defini atributos para descartar los elimino ahora
        logging.info("Elimino atributos:")
        df = df.drop([c for c in df.columns if any(c.startswith(p) for p in DROP)]).clone()
        gc.collect()
        mlflow.log_param("df_shape", df.shape)

        if FLAG_ZLIGHTGBM == 0 and STUDY_HP is None and ZEROSHOT == False:
            ## Ejecutar optimizacion de hiperparametros
            mlflow.log_param("undersampling_ratio_BO", UNDERSAMPLING_FRACTION)
            mlflow.log_param("n_trials_BO", args.n_trials)
            mlflow.log_param("ksemillerio_BO", KSEMILLERIO_BO)

            study = optimizar(df, n_trials = args.n_trials, n_jobs = args.n_jobs)

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("ganancia_best_BO", study.best_value)

            ## 5. Análisis adicional
            logger.info("=== ANÁLISIS DE RESULTADOS ===")
            trials_df = study.trials_dataframe()
            if len(trials_df) > 0:
                top_5 = trials_df.nlargest(5, 'value')
                logger.info("Top 5 mejores trials:")
                for idx, trial in top_5.iterrows():
                    logger.info(f"  Trial {trial['number']}: {trial['value']:,.4f}")
            logger.info(f'Mejores Hiperparametros: {study.best_params}')
            logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

        elif FLAG_ZLIGHTGBM == 1:
            df = create_canaritos(df, qcanaritos=PARAMETROS_ZLGB['qcanaritos'])
            gc.collect()

        if ZEROSHOT:
            logger.info("=== ANÁLISIS ZEROSHOT ===")
            resultado_zs = optimizar_zero_shot(df)

            # Desempacar resultados del diccionario
            ganancia_val = resultado_zs["ganancia_validacion"]
            umbral_sugerido = resultado_zs["umbral_sugerido"]
            params_lightgbm = resultado_zs["best_params_lightgbm"]
            hyperparams = resultado_zs["best_params_flaml"]
            paths = resultado_zs["paths"]

            logger.info("=== ANÁLISIS DE RESULTADOS ZEROSHOT ===")
            logger.info(f"✅ Ganancia en validación: {ganancia_val:,.0f}")
            logger.info(f"✅ Umbral sugerido: {umbral_sugerido:.4f}")
            logger.info(f"✅ Parámetros FLAML guardados: {len(hyperparams)} parámetros")
            logger.info(f"✅ Parámetros LightGBM guardados: {len(params_lightgbm)} parámetros")
            logger.info(f"✅ Archivos generados:")
            logger.info(f"   - Iteraciones: {paths['iteraciones']}")
            logger.info(f"   - Best params: {paths['best_params']}")

        elif STUDY_HP is None:
            mejores_params = cargar_mejores_hiperparametros()
        else:
            mejores_params = cargar_mejores_hiperparametros(archivo_base=STUDY_HP)

        df_train = df.filter(pl.col("foto_mes").is_in(MES_TRAIN)).clone()
        df_train = undersample(df_train, sample_fraction=UNDERSAMPLING_FRACTION)

        df_test = df.filter(pl.col("foto_mes").is_in(MES_TEST)).clone()

        del df
        gc.collect()

        import sys
        import inspect

        print("=== VARIABLES QUE QUEDAN REFERENCIANDO DATAFRAMES ===")
        for name, obj in globals().items():
            if isinstance(obj, pl.DataFrame):
                print("GLOBAL:", name, sys.getrefcount(obj))

        for frame_info in inspect.stack():
            frame = frame_info.frame
            for name, obj in frame.f_locals.items():
                if isinstance(obj, pl.DataFrame):
                    print("STACK:", name, sys.getrefcount(obj))

        import psutil
        proc = psutil.Process(os.getpid())
        print("RSS MB:", proc.memory_info().rss / 1024 ** 2)
        print("PL df_train size est (MB):", df_train.estimated_size() / 1024 ** 2)
        print("PL df_test  size est (MB):", df_test.estimated_size() / 1024 ** 2)
        gc.collect()
        print("After GC - RSS MB:", proc.memory_info().rss / 1024 ** 2)

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

