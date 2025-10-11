import pandas as pd
import polars as pl
import numpy as np
import duckdb
import logging
from .config import SEMILLA
import os

logger = logging.getLogger(__name__)


def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Para cada columna en `columnas`, calcula un ranking percentil por grupo (group_col).
    - Valores > 0: percentil en (0, 1]
    - Valores < 0: percentil en [-1, 0)
    - Valores == 0 -> 0
    NaNs se mantienen como NaN.

    Parameters
    ----------
    df : pd.DataFrame
    columnas : list[str]
    """

    logger.info(f"Realizando feature engineering para reducir data drift para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar rankings")
        return df

    columnas_inicial = df.columns.tolist()

    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"Advertencia: El atributo {attr} no existe en el DataFrame")
            continue

        def rank_signed(s: pd.Series) -> pd.Series:
            # resultado inicial con NaNs
            res = pd.Series(index=s.index, dtype=float)

            mask_neg = s < 0
            mask_pos = s > 0
            mask_zero = s == 0

            # Negativos
            if mask_neg.any():
                # rank por magnitud: mayor |valor| -> percentil más alto -> mapeo a -1
                abs_rank = s.loc[mask_neg].abs().rank(pct=True, method='average')
                res.loc[mask_neg] = -abs_rank

            # Positivos
            if mask_pos.any():
                pos_rank = s.loc[mask_pos].rank(pct=True, method='average', ascending=True)
                res.loc[mask_pos] = pos_rank

            # Ceros → 0 exactamente
            if mask_zero.any():
                res.loc[mask_zero] = 0.0

            return res

        # aplicamos por grupo; transform devuelve una serie alineada con el índice original
        df[attr] = df.groupby('foto_mes')[attr].transform(rank_signed)

    # Selecciono las columnas
    df = df[columnas_inicial]

    print(df.head())

    logger.info(f"Feature engineering rank finalizado. DataFrame resultante con {df.shape[1]} columnas")

    return df

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
    # sql = "SELECT *"
    sql = "SELECT CAST(STRFTIME(foto_mes::DATE, '%Y%m') AS INTEGER) as foto_mes, * EXCLUDE(foto_mes)"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def feature_engineering_trend(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de tendencia para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar tendencias. Si es None, no se generan tendencias.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de tendencia agregadas
    """

    logger.info(f"Realizando feature engineering de tendencia para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT CAST(STRFTIME(foto_mes::DATE, '%Y%m') AS INTEGER) as foto_mes, * EXCLUDE(foto_mes)"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f", regr_slope({attr}, cliente_antiguedad) over ventana as {attr}_trend_3m"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += " window ventana as (partition by numero_de_cliente order by foto_mes rows between 3 preceding and current row)"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


def feature_engineering_delta(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables delta (attr - attr_lag_i) usando Polars.
    """
    logger.info(f"Comienzo feature delta (Polars). df shape: {df.shape}")

    df = pl.from_pandas(df)

    exprs = []
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"No se encontró {attr}, se omite.")
            continue

        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag_{i}"
            if lag_col not in df.columns:
                logger.warning(f"No se encontró {lag_col}, se omite.")
                continue

            exprs.append((pl.col(attr) - pl.col(lag_col)).alias(f"{attr}_delta_{i}"))

    if exprs:
        df = df.with_columns(exprs)

    df = df.to_pandas()

    logger.info(df.head())

    logger.info(f"Ejecución delta finalizada. df shape: {df.shape}")
    return df


def fix_aguinaldo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige las columnas asociadas al pago de aguinaldo para el mes de junio 2021
    :param df: dataframe con los datos
    :return: dataframe corregido
    """

    columns = df.columns.tolist()

    logger.info(f"Inicia fix de variables por aguinaldo")
    sql = """
    with aguinaldo as (
    SELECT foto_mes, 
           numero_de_cliente, 
           mpayroll_delta_1,
           flag_aguinaldo,
           cpayroll_trx
    FROM (
    SELECT foto_mes, 
           numero_de_cliente,
           lag(mpayroll, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag_1,
           lag(mpayroll, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag_2,
           mpayroll - mpayroll_lag_1 as mpayroll_delta_1,
           case when foto_mes = '2021-06-30' 
               and mpayroll/mpayroll_lag_1  >= 1.3 
               and mpayroll/mpayroll_lag_2  >= 1.3 
                    then 1 
                else 0 
           end as flag_aguinaldo,
           case when flag_aguinaldo = 1 and cpayroll_trx > 1 then cpayroll_trx - 1 else cpayroll_trx end as cpayroll_trx
    FROM df) as a
    WHERE foto_mes = '2021-06-30')
        
    SELECT df.* REPLACE(
                case when aguinaldo.mpayroll_delta_1 is null then df.mpayroll when df.foto_mes = '2021-06-30' then df.mpayroll - aguinaldo.mpayroll_delta_1 + aguinaldo.mpayroll_delta_1/6 else df.mpayroll + aguinaldo.mpayroll_delta_1/6 end as mpayroll, 
                case when aguinaldo.mpayroll_delta_1 is null then df.cpayroll_trx when df.foto_mes = '2021-06-30' then aguinaldo.cpayroll_trx else df.cpayroll_trx end as cpayroll_trx
                )
    FROM df 
    LEFT JOIN aguinaldo
    ON df.numero_de_cliente = aguinaldo.numero_de_cliente
    ORDER BY df.numero_de_cliente, df.foto_mes
    """


    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()
    logger.info(f"Finaliza fix de variables por aguinaldo")

    df = df[columns]

    return df


def undersample(df, sample_fraction):
    """
    Realiza un undersampling de la clase mayoritaria.
    Versión optimizada usando boolean indexing en lugar de groupby.apply.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        sample_fraction (float): Fracción de la clase mayoritaria a conservar (entre 0 y 1).
    Returns:
        pd.DataFrame: El DataFrame resultante submuestreado.
    """

    # Separar las clases usando máscaras booleanas (más rápido que groupby)
    mask_mayoritaria = df['target'] == 0
    mask_minoritaria = df['target'] == 1

    df_mayoritaria = df[mask_mayoritaria]
    df_minoritaria = df[mask_minoritaria]

    # Sample de la clase mayoritaria
    df_mayoritaria_sampled = df_mayoritaria.sample(
        frac=sample_fraction,
        random_state=SEMILLA[1]
    )

    # Concatenar las clases (minoritaria completa + mayoritaria submuestreada)
    df_undersampled = pd.concat([df_mayoritaria_sampled, df_minoritaria], ignore_index=True)

    # Shuffle para mezclar las clases
    df_undersampled = df_undersampled.sample(frac=1, random_state=SEMILLA[1]).reset_index(drop=True)

    # Calcular proporciones
    prop_continua = len(df_mayoritaria_sampled) / len(df_mayoritaria)
    prop_baja = len(df_minoritaria) / len(df_minoritaria)

    # Imprimir estadísticas para verificar la reducción
    logging.info(f"Tamaño original del DataFrame: {len(df)}")
    logging.info(f"Tamaño final del DataFrame: {len(df_undersampled)}")
    logging.info(f"Proporcion final de clase mayoritaria: {prop_continua:.2f}")
    logging.info(f"Proporcion final de clase minoritaria: {prop_baja:.2f}")
    logging.info(f"Clase mayoritaria: {len(df_mayoritaria_sampled):,} / Clase minoritaria: {len(df_minoritaria):,}")

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
