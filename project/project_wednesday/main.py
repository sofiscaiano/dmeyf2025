import pandas as pd
from datetime import datetime
import os
import logging
import argparse
import numpy as np
import gc
import polars as pl
import mlflow
import shutil
from src.features import *
from src.loader import cargar_datos_csv, cargar_datos, convertir_clase_ternaria_a_target
from src.optimization import optimizar
from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.config import *
from src.best_params import cargar_mejores_hiperparametros, cargar_mejores_envios
from src.final_training import generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import exportar_envios_bot
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
logger.info(f"DF_FE: {DF_FE}")
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

        mlflow.log_param("zlightgbm", FLAG_ZLIGHTGBM)
        mlflow.log_param("fix_aguinaldo", FLAG_AGUINALDO)
        mlflow.log_param("rankings", FLAG_RANKS)
        mlflow.log_param("q_lags", QLAGS)
        mlflow.log_param("trend_3m", FLAG_TREND_3M)
        mlflow.log_param("trend_6m", FLAG_TREND_6M)
        mlflow.log_param("zero_sd", FLAG_ZEROSD)
        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact("main.py")

        if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"{DF_FE}.parquet")):
            logger.info(f"✅ {DF_FE} encontrado")
            data_path = os.path.join(BUCKET_NAME, "datasets", f"{DF_FE}.parquet")
            if FLAG_GCP == 1:
                data_path = f'~/datasets/{DF_FE}.parquet'
            months_filter = list(set(MES_TRAIN + MES_VALIDACION + MES_TEST + FINAL_TRAIN + FINAL_PREDICT))
            df = cargar_datos(data_path, lazy=True, months=months_filter)
            gc.collect()

        else:
            ## Carga de Datos
            logger.info(f"❌ {DF_FE} no encontrado")
            os.makedirs(f'{BUCKET_NAME}/datasets', exist_ok=True)

            # Me fijo si existe el df con la clase_ternaria (target) y lo cargo
            if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"competencia_03.parquet")):
                data_path = os.path.join(BUCKET_NAME, "datasets", f"competencia_03.parquet")
                if FLAG_GCP == 1:
                    data_path = '~/datasets/competencia_03.parquet'
                df = cargar_datos(data_path, lazy=False)
            # Si no existe, lo creo
            else:
                # Me fijo si existe el df crudo de la competencia 03
                if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"competencia_02_03_crudo.parquet")):
                    df = cargar_datos(os.path.join(BUCKET_NAME, "datasets", f"competencia_02_03_crudo.parquet"), lazy=False)
                # Si no existe, cargo los dos df y los concateno
                else:
                    df_2 = cargar_datos_csv(os.path.join(BUCKET_NAME, "datasets", f"competencia_02_crudo.csv.gz"))
                    df_3 = cargar_datos_csv(os.path.join(BUCKET_NAME, "datasets", f"competencia_03_crudo.csv.gz"))
                    df = pl.concat([df_2, df_3])
                    data_path = os.path.join(BUCKET_NAME, "datasets", f"competencia_02_03_crudo.parquet")
                    df.write_parquet(data_path, compression="gzip")

                df = create_target(df, export=True)

            ## Feature Engineering
            atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'numero_de_cliente']]

            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos.html')
            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos_after_data_quality.html')
            if FLAG_AGUINALDO:
                df = fix_aguinaldo(df)
            if FLAG_ZEROSD:
                df = fix_zero_sd(df, columnas=atributos)

            df = create_features(df)
            atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'numero_de_cliente']]
            atributos_monetarios = [c for c in df.columns if any(c.startswith(p) for p in ['m', 'Visa_m', 'Master_m', 'tc_m'])]

            if FLAG_RANKS:
                df = feature_engineering_rank(df, columnas=atributos_monetarios) # pandas
            elif FLAG_IPC:
                df = feature_engineering_ipc(df, columnas=atributos_monetarios) # polars

            # generar_reporte_mensual_html(df, columna_target='target', nombre_archivo='reporte_atributos_final.html')

            if FLAG_TREND_3M:
                df = feature_engineering_trend(df, columnas=atributos, q=3)
            if FLAG_TREND_6M:
                df = feature_engineering_trend(df, columnas=atributos, q=6)

            if any([FLAG_MIN_6M, FLAG_MAX_6M, FLAG_AVG_6M]):
                df = feature_engineering_min_max_avg(df, columnas=atributos, window=6, f_min=FLAG_MIN_6M, f_max=FLAG_MAX_6M, f_avg=FLAG_AVG_6M)

            if FLAG_AVG_3M:
                df = feature_engineering_min_max_avg(df, columnas=atributos, window=3, f_min=False, f_max=False, f_avg=FLAG_AVG_3M)

            # atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'numero_de_cliente']]

            df = feature_engineering_lag(df, columnas=atributos, lags=QLAGS) # duckdb 
            df = feature_engineering_delta(df, columnas=atributos, lags=QLAGS) # polars

            ## Convertir clase ternaria a target binaria
            df = convertir_clase_ternaria_a_target(df)

            if FLAG_EMBEDDING:
                df = create_embedding_lgbm_rf(df)

            logging.info(f"==== Exporto el archivo {DF_FE}.parquet ====")
            data_path = os.path.join(BUCKET_NAME, "datasets", f"{DF_FE}.parquet")
            if FLAG_GCP == 1:
                data_path = f'~/datasets/{DF_FE}.parquet'
                source_file = os.path.expanduser(f'~/datasets/{DF_FE}.parquet') 
                destination_file = os.path.expanduser(f'~/buckets/b1/datasets/{DF_FE}.parquet')
                
                df.write_parquet(data_path, compression="gzip")

                try:
                    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
                    shutil.copyfile(source_file, destination_file)
                    print(f"Archivo copiado exitosamente de {source_file} a {destination_file}.")
                
                except FileNotFoundError:
                    print(f"Error: El archivo de origen no se encontró en {source_file}.")
                except Exception as e:
                    print(f"Ocurrió un error al copiar el archivo: {e}")
            else: 
                df.write_parquet(data_path, compression="gzip")
            

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

        resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df, mejores_params)
        guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)
        
        return
        
        entrenar_modelo_final(df, mejores_params)

        ## Generar predicciones
        if ENVIOS is not None:
            envios = ENVIOS
            logger.info(f"Envios: {envios}")
        else:
            envios = cargar_mejores_envios()

        predicciones = generar_predicciones_finales(df, envios)
        salida_kaggle = exportar_envios_bot(predicciones)

        ## Resumen final
        logger.info("=== RESUMEN FINAL ===")
        logger.info(f"✅ Entrenamiento final completado exitosamente")
        logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
        logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
        logger.info(f"🔮 Período de predicción: {FINAL_PREDICT}")
        logger.info(f"📁 Archivo de salida: {salida_kaggle}")
        logger.info(f"📝 Log detallado: log/{nombre_log}")

        logger.info(f'Ejecucion finalizada. Revisar log para mas detalles. {nombre_log}')

if __name__ == '__main__':
    main()

