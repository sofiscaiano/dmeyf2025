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
nombre_log = f"log_{STUDY_NAME}_ensemble_{fecha}.log"
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
logger.info(f"MES_TEST: {MES_TEST_ENSEMBLE}")
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

    with mlflow.start_run(run_name=f"ensemble-{STUDY_NAME}"):
        mlflow.set_tags(MLFLOW_TAGS)

        mlflow.log_param("models", MODELOS)
        mlflow.log_param("weights", WEIGHTS)
        mlflow.log_param("mes_test", MES_TEST_ENSEMBLE)
        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact("ensemble.py")

        from functools import reduce
        dfs = {}

        for modelo in MODELOS:
            # 1. Construir el nombre del archivo (path) completo
            # Usamos f-string para concatenar el nombre base con el sufijo
            path_archivo = os.path.join(BUCKET_NAME, "resultados", f"{modelo}_probabilidades.csv")

            # 2. Llamar a la función cargar_datos() con el path
            # La función debe devolver el DataFrame
            df = cargar_datos_csv(path_archivo)
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

        df_primero = lista_dataframes_renombrados[0]

        lista_sin_claves = [
            df.drop(["numero_de_cliente", "target"])
            for df in lista_dataframes_renombrados[1:]
        ]

        # 3. Concatenar horizontalmente
        df_final = pl.concat(
            [df_primero] + lista_sin_claves,
            how='horizontal'
        )

        cols = df_final.select(
            pl.col("^probabilidad_.*$")  # Expresión regular para columnas que comienzan con "probabilidad_"
        ).columns

        df_final = df_final.with_columns([
            pl.mean_horizontal(cols).alias("ensemble")
        ])

        print(df_final.head())

        print(len(df_final))

        # Calcular la ganancia de cada modelo individual
        ganancias_acumuladas = []
        y_test = df_final.get_column('target').to_list()

        mlflow.set_tag("ensemble", 1)

        for i, col in enumerate(cols):
            pred_modelo = df_final.get_column(col).to_list()
            ganancia = calcular_ganancias_acumuladas(y_test, pred_modelo)
            ganancias_acumuladas.append(ganancia)

            mlflow.log_metric(
                key="ganancia",
                value=np.max(ganancia),
                step=i
            )

        # Calculo la ganancia del ensamble
        pred_ensamble = df_final.get_column('ensemble').to_list()
        ganancia_ensamble = calcular_ganancias_acumuladas(y_test, pred_ensamble)
        ganancias_acumuladas.append(ganancia_ensamble)

        auc = roc_auc_score(y_test, pred_ensamble)

        mlflow.log_metric(
            key="auc_test",
            value=auc
        )

        mlflow.log_metric(
            key="ganancia_test",
            value=np.max(ganancia_ensamble)
        )

        ganancia_ensamble_meseta = (
            pd.Series(ganancia_ensamble)
            .rolling(window=1001, center=True, min_periods=1)
            .mean()
        ).max(skipna=True)

        mlflow.log_metric(
            key="ganancia_meseta_test",
            value=ganancia_ensamble_meseta
        )

        ruta_grafico_multiple = crear_grafico_multiple_ganancia(ganancias_acumuladas)
        mlflow.log_artifact(ruta_grafico_multiple)

        # resultados_test, y_pred, ganancias_acumuladas = evaluar_en_test(df, mejores_params)
        # guardar_resultados_test(resultados_test, archivo_base=STUDY_NAME)
        #
        # entrenar_modelo_final(df, mejores_params)

        return 
        
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

