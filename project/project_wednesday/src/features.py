import pandas as pd
import numpy as np
import duckdb
import logging
from .config import SEMILLA

logger = logging.getLogger(__name__)

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT CAST(STRFTIME(foto_mes::DATE, '%Y%m') AS INTEGER) as foto_mes, * EXCLUDE (foto_mes)"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
                # sql += f", {attr} - {attr}_lag_{i} as {attr}_delta_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += (" FROM df")
    sql += (" ORDER BY numero_de_cliente, foto_mes")

    # logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    for attr in columnas:
        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag_{i}"
            delta_col = f"{attr}_delta_lag_{i}"
            if lag_col in df.columns:
                # Usar .values para evitar la indexación de Pandas, que puede ser lenta
                df[delta_col] = df[attr].values - df[lag_col].values

    print(df.head())

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def fix_aguinaldo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige las columnas asociadas al pago de aguinaldo para el mes de junio 2021
    :param df: dataframe con los datos
    :return: dataframe corregido
    """

    logger.info(f"Inicia fix de variables por aguinaldo")
    sql = """
    SELECT a.* EXCLUDE(mpayroll, cpayroll_trx, flag_aguinaldo),
           case when flag_aguinaldo = 1 then mpayroll_lag_1 else mpayroll end as mpayroll,
           case when flag_aguinaldo = 1 and cpayroll_trx > 1 then cpayroll_trx - 1 else cpayroll_trx end as cpayroll_trx
    FROM (
    SELECT *,
           case when foto_mes = 202106 
               and mpayroll/mpayroll_lag_1 >= 1.3 
               and mpayroll/mpayroll_lag_2 >= 1.3 
                    then 1 
                else 0 
               end as flag_aguinaldo
    FROM df) as a
    """

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.execute("SET memory_limit='20GB';")
    con.execute("SET threads=6;")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()
    logger.info(f"Finaliza fix de variables por aguinaldo")

    return df


def undersample(df, sample_fraction):
    """
    Realiza un undersampling de la clase mayoritaria.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        sample_fraction (float): Fracción de la clase mayoritaria a conservar (entre 0 y 1).
    Returns:
        pd.DataFrame: El DataFrame resultante submuestreado.
    """

    # Función lambda para aplicar a cada grupo (clase)
    # Si el grupo es la clase mayoritaria, se aplica el muestreo con la fracción.
    # Para el resto de clases, se conserva el 100% de los datos (frac=1).
    df_undersampled = df.groupby('target', group_keys=False).apply(
        lambda x: x.sample(
            frac=sample_fraction,
            random_state=SEMILLA[1]
        ) if x.name == 0 else x
    ).reset_index(drop=True)

    prop_continua = (df_undersampled['target'] == 0).sum() / (df['target'] == 0).sum()
    prop_baja = (df_undersampled['target'] == 1).sum() / (df['target'] == 1).sum()

    # Imprimir estadísticas para verificar la reducción
    logging.info(f"Tamaño original del DataFrame: {len(df)}")
    logging.info(f"Tamaño final del DataFrame: {len(df_undersampled)}")
    logging.info(f"Proporcion final de clase mayoritaria: {prop_continua:.2f}")
    logging.info(f"Proporcion final de clase minoritaria: {prop_baja:.2f}")

    return df_undersampled


import pandas as pd
import plotly.express as px
from io import StringIO
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def generar_reporte_mensual_html(df, columna_fecha='foto_mes', nombre_archivo='reporte_mensual.html'):
    """
    Genera un archivo HTML con gráficos de líneas interactivos de Plotly,
    mostrando la evolución mensual del promedio de cada feature.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene la columna de fecha y features.
        columna_fecha (str): Nombre de la columna que contiene la fecha mensual (ej: 'foto_mes').
        nombre_archivo (str): Nombre del archivo HTML de salida.
    """

    print(f"Iniciando generación de reporte para {len(df.columns) - 1} features...")

    # --- 1. Preparación de datos: Calcular el promedio mensual para cada feature ---

    # Excluir la columna de fecha para agrupar las features numéricas
    features_numericas = df.drop(columns=[columna_fecha]).select_dtypes(include=np.number).columns

    if features_numericas.empty:
        print("Error: No se encontraron features numéricas para graficar.")
        return

    # Calcular el promedio mensual para todas las features numéricas
    df_agrupado = df.groupby(columna_fecha)[features_numericas].mean().reset_index()

    # Asegurarse de que la columna de fecha esté ordenada
    df_agrupado = df_agrupado.sort_values(columna_fecha)

    # --- 2. Generar el contenido HTML de cada gráfico ---

    graficos_html = []

    for feature in features_numericas:
        # Generar el gráfico de línea interactivo con Plotly Express
        fig = px.line(
            df_agrupado,
            x=columna_fecha,
            y=feature,
            title=f'Evolución Mensual de {feature} (Promedio)'
        )

        # Mejorar el layout y el formato de los ejes
        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            xaxis_title="Período Mensual",
            yaxis_title=f"Promedio de {feature}",
            title_x=0.5  # Centrar el título
        )

        # Exportar el gráfico como una cadena HTML
        # full_html=False asegura que solo se exporte el <div> que contiene el gráfico
        # y no un documento HTML completo
        html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')
        graficos_html.append(f"""
            <div style="width: 80%; margin: 40px auto; border: 1px solid #ddd; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                {html_string}
            </div>
            <hr>
        """)

    # --- 3. Unir los gráficos en un solo archivo HTML ---

    # Plantilla HTML básica
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Evolución Mensual</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; }}
            h1 {{ text-align: center; color: #333; padding: 20px; background-color: #fff; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Evolución Mensual de Features ({df_agrupado[columna_fecha].min()} a {df_agrupado[columna_fecha].max()})</h1>

        {''.join(graficos_html)}

    </body>
    </html>
    """

    # Escribir el contenido en el archivo
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"\n✅ Reporte HTML generado exitosamente: {nombre_archivo}")
    print("Abre el archivo en tu navegador web para ver los gráficos interactivos.")
