import pandas as pd
from datetime import datetime
import os
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
import polars as pl
import mlflow
from src.loader import cargar_datos_csv
from src.test_evaluation import calcular_ganancias_acumuladas, crear_grafico_multiple_ganancia
from src.config import *
from src.final_training import generar_predicciones_finales
from src.output_manager import exportar_envios_bot


# config basico logging
path_logs = os.path.join(BUCKET_NAME, "log")
os.makedirs(path_logs, exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nombre_log = f"log_{STUDY_NAME}_final_ensemble_{fecha}.log"
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
logger.info(f"MODELOS A ENSAMBLAR: {MODELOS}")
logger.info(f"WEIGHTS: {WEIGHTS}")
logger.info(f"FINAL_PREDICT: {MES_TEST_ENSEMBLE}")
logger.info(f"CANTIDAD DE ENVIOS: {ENVIOS_ENSEMBLE}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

def main():

    print(">>> Inicio de ejecucion")

    # Configurar MLflow y ejecutar pipeline completo
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow configurado con servidor remoto: {MLFLOW_TRACKING_URI}")

    except Exception as e:
        logger.error(f"Error al configurar MLflow: {e}")
        raise

    with mlflow.start_run(run_name=f"final_ensemble-{STUDY_NAME}"):
        mlflow.set_tags(MLFLOW_TAGS)

        mlflow.log_param("models", MODELOS)
        mlflow.log_param("weights", WEIGHTS)
        mlflow.log_param("mes_test", MES_TEST_ENSEMBLE)
        mlflow.log_param("mes_predict", MES_FINAL_ENSEMBLE)
        mlflow.log_param("envios", ENVIOS_ENSEMBLE)
        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact("final_ensemble.py")

        from functools import reduce
        dfs = {}

        for modelo in MODELOS:
            # 1. Construir el nombre del archivo (path) completo
            # Usamos f-string para concatenar el nombre base con el sufijo
            path_archivo = os.path.join(BUCKET_NAME, "predict", f"{modelo}_predicciones.csv")

            # 2. Llamar a la función cargar_datos() con el path
            # La función debe devolver el DataFrame
            df = cargar_datos_csv(path_archivo)
            df = df.drop("Predicted")
            dfs[modelo] = df

        lista_dataframes = list(dfs.values())
        lista_dataframes_renombrados = []

        for modelo, df in zip(MODELOS, lista_dataframes):
            # Determinar qué columnas renombrar (todas excepto la clave)
            cols = {
                'probabilidad': f"probabilidad_{modelo}"
            }

            df_renombrado = df.clone().rename(cols)
            lista_dataframes_renombrados.append(df_renombrado)

        # Función que define la operación de join
        def merge_dataframes(left_df, right_df):
            """Realiza un join entre dos DFs usando 'numero_de_cliente' como clave."""
            return left_df.join(
                right_df,
                on="numero_de_cliente",
                how="inner"  # Usamos 'outer' para incluir a todos los clientes de todos los DFs
            )

        # Aplicar la función de merge a toda la lista de DataFrames renombrados
        df_final = reduce(merge_dataframes, lista_dataframes_renombrados)

        cols = df_final.select(
            pl.col("^probabilidad_.*$")  # Expresión regular para columnas que comienzan con "probabilidad_"
        ).columns

        # probabilidades ensambladas
        df_final = df_final.with_columns([
            pl.mean_horizontal(cols).alias("ensemble")
        ])

        print(df_final.head())

        print(len(df_final))

        df_final = df_final.drop(cols)

        mlflow.set_tag("ensemble_final", 1)

        if ENVIOS_ENSEMBLE is not None:
            envios = ENVIOS_ENSEMBLE
            logger.info(f"Envios: {envios}")

        resultados = df_final.to_pandas()
        resultados = resultados.sort_values(by="ensemble", ascending=False).reset_index(drop=True)

        # 4. Asignar etiquetas: 1 a los primeros envios, 0 al resto
        resultados["Predicted"] = 0
        resultados.loc[:envios -1, "Predicted"] = 1

        # Estadísticas de predicciones
        total_predicciones = len(resultados)
        predicciones_positivas = (resultados['Predicted'] == 1).sum()
        porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

        logger.info(f"Predicciones generadas:")
        logger.info(f"  Total clientes: {total_predicciones:,}")
        logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
        logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
        logger.info(f"  Cantidad de envios: {envios}")

        salida_kaggle = exportar_envios_bot(resultados)

        return


if __name__ == '__main__':
    main()

