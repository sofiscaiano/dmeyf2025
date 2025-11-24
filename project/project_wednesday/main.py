import pandas as pd
from datetime import datetime
import os
import logging
import argparse
import numpy as np
import gc
import polars as pl
import mlflow
import ast
from src.features import *
from src.loader import cargar_datos_csv, cargar_datos, convertir_clase_ternaria_a_target, load_dataset_undersampling_efficient
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
        mlflow.log_param("zlgbm_wl", ZLGBM_WEAKLEARNER)
        mlflow.log_param("fix_aguinaldo", FLAG_AGUINALDO)
        mlflow.log_param("rankings", FLAG_RANKS)
        mlflow.log_param("q_lags", QLAGS)
        mlflow.log_param("trends", FLAG_TRENDS)
        mlflow.log_param("zero_sd", FLAG_ZEROSD)
        mlflow.log_artifact("config.yaml")

        # Importo el df_fe si existe
        if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")):
            logger.info("✅ df_fe encontrado")
            data_path = os.path.join(BUCKET_NAME, "datasets", f"df_fe.parquet")
            if FLAG_GCP == 1:
                data_path = '~/datasets/df_fe.parquet'
            # Cargar df para train con undersampling
            full_months = [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202106, 202107]
            df_undersampled = load_dataset_undersampling_efficient(path=data_path, months=full_months, seed=SEMILLA[0], fraction=UNDERSAMPLING_FRACTION)
            df_val = load_dataset_undersampling_efficient(path=data_path, months=MES_VALIDACION, seed=SEMILLA[0], fraction=1)
            df_test = load_dataset_undersampling_efficient(path=data_path, months=MES_TEST, seed=SEMILLA[0], fraction=1)
            df_predict = load_dataset_undersampling_efficient(path=data_path, months=FINAL_PREDICT, seed=SEMILLA[0], fraction=1)

            gc.collect()

            if FLAG_CANARITOS_ASESINOS:
                # Uso los hiperparametros de la competencia 2 para hacer una reduccion de dimensionalidad con canaritos
                df = pl.concat([df_undersampled.filter(pl.col("foto_mes").is_in(MES_TRAIN_BO)), df_val])
                run_canaritos_asesinos(df, qcanaritos=50, ksemillerio=5, metric=50, params_path='lgb_optimization_competencia197')

            # Importo listado de features seleccionadas por los canaritos asesinos
            features_path = os.path.join(os.path.join(BUCKET_NAME, "resultados"), f"selected_features_{STUDY_NAME}.txt")
            with open(features_path, 'r') as f:
                features_str = f.read()
            selected_features = ast.literal_eval(features_str)
            logging.info(f"Features seleccionadas desde el archivo: {features_path}")
            logging.info(selected_features)
            logging.info(f'Shape after selected features: {df.shape}')

            # Si defini atributos para descartar los elimino ahora del selected_features
            logging.info("Atributos a eliminar:")
            to_drop = [c for c in df_undersampled.columns if any(c.startswith(p) for p in DROP)]
            selected_features_final = [
                elemento for elemento in selected_features
                if elemento not in to_drop
            ]

            fixed_features = ['target', 'target_train', 'target_test', 'w_train']
            df_undersampled = df_undersampled.select(selected_features_final + fixed_features)
            df_val = df_val.select(selected_features_final + fixed_features)
            df_test = df_test.select(selected_features_final + fixed_features)
            df_predict = df_predict.select(selected_features_final + fixed_features)


        # Si no existe el df_fe lo genero
        else:
            logger.info("❌ df_fe no encontrado")
            os.makedirs(f'{BUCKET_NAME}/datasets', exist_ok=True)
            # Me fijo si existe el df con la clase_ternaria (target) y lo cargo
            if os.path.exists(os.path.join(BUCKET_NAME, "datasets", f"competencia_03.parquet")):
                data_path = os.path.join(BUCKET_NAME, "datasets", f"competencia_03.parquet")
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

            ## Inicio de Feature Engineering
            atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'target_train', 'target_test', 'numero_de_cliente', 'w_train']]

            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos.html')
            # generar_reporte_mensual_html(df, columna_target= 'target', nombre_archivo= 'reporte_atributos_after_data_quality.html')
            if FLAG_AGUINALDO:
                df = fix_aguinaldo(df)
            if FLAG_ZEROSD:
                df = fix_zero_sd(df, columnas=atributos)

            df = create_features(df)
            atributos = [c for c in df.columns if c not in ['foto_mes', 'target', 'target_train', 'target_test', 'numero_de_cliente', 'w_train']]
            atributos_monetarios = [c for c in df.columns if any(c.startswith(p) for p in ['m', 'Visa_m', 'Master_m', 'tc_m'])]

            # RANKINGS
            if FLAG_RANKS:
                df = feature_engineering_percent_rank(df, columnas=atributos)
                df = feature_engineering_ntile(df, columnas=atributos, k=10)
                df = feature_engineering_rank_cero_fijo(df, columnas=atributos, prefijo='')
                df = feature_engineering_percent_rank_dense(df, columnas=atributos)

            if FLAG_MIN or FLAG_MAX:
                df = feature_engineering_min_max(df, columnas=atributos, window=6)
            if FLAG_RATIOAVG:
                df = feature_engineering_ratioavg(df, columnas=atributos, window=6)

            # generar_reporte_mensual_html(df, columna_target='target', nombre_archivo='reporte_atributos_final.html')

            # TENDENCIAS
            if FLAG_TRENDS:
                df = feature_engineering_trend(df, columnas=atributos, q=3)
                df = feature_engineering_trend(df, columnas=atributos, q=6)

            df = feature_engineering_lag(df, columnas=atributos, cant_lag=QLAGS)
            df = feature_engineering_delta(df, columnas=atributos, cant_lag=QLAGS)

            if FLAG_EMBEDDING:
                df = create_embedding_lgbm_rf(df)

            logging.info("==== Exporto el df_fe.parquet ====")
            data_path = os.path.join(BUCKET_NAME, "datasets", "df_fe.parquet")
            if FLAG_GCP == 1:
                data_path = '~/datasets/df_fe.parquet'
            df.write_parquet(data_path, compression="gzip")

            return


        mlflow.log_param("df_shape", df.shape)

        if FLAG_ZLIGHTGBM == 0 and STUDY_HP is None and ZEROSHOT == False:
            ## Ejecutar optimizacion de hiperparametros
            mlflow.log_param("undersampling_ratio_BO", UNDERSAMPLING_FRACTION)
            mlflow.log_param("n_trials_BO", args.n_trials)
            mlflow.log_param("ksemillerio_BO", KSEMILLERIO_BO)

            # df para BO:
            df = pl.concat([df_undersampled.filter(pl.col("foto_mes").is_in(MES_TRAIN_BO)), df_val])

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
            # df para ZLIGHTGBM:
            df = pl.concat([df_undersampled.filter(pl.col("foto_mes").is_in(MES_TRAIN)), df_test])
            df = create_canaritos(df, qcanaritos=PARAMETROS_ZLGB['qcanaritos'])
            gc.collect()

        if ZEROSHOT:
            logger.info("=== ANÁLISIS ZEROSHOT ===")
            # df para ZeroShot
            df = pl.concat([df_undersampled.filter(pl.col("foto_mes").is_in(MES_TRAIN)), df_test])
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

        # df para testing
        df = pl.concat([df_undersampled.filter(pl.col("foto_mes").is_in(MES_TRAIN)), df_test])
        resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df, mejores_params)
        guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)

        # df para training final
        df = df_undersampled.filter(pl.col("foto_mes").is_in(FINAL_TRAIN))
        entrenar_modelo_final(df, mejores_params)

        ## Generar predicciones
        if ENVIOS is not None:
            envios = ENVIOS
            logger.info(f"Envios: {envios}")
        else:
            envios = cargar_mejores_envios()

        predicciones = generar_predicciones_finales(df_predict, envios)
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

