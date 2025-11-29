import pandas as pd
import polars as pl
import numpy as np
import duckdb
import logging
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from .config import SEMILLA
from .basic_functions import train_test_split
import os
import gc
import plotly.express as px
from io import StringIO
import warnings
# import mlflow

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def feature_engineering_ipc(df: pl.DataFrame, columnas: list[str], mes_base: int = 201901) -> pl.DataFrame:
    """
    Actualizacion por IPC con base en 201901
    """

    logger.info(f"Realizando tratamiento por IPC para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para ajustar por IPC")
        return df

    dict_ipc = {
    202109: 3.5,
    202108: 2.5,
    202107: 3.0,
    202106: 3.2,
    202105: 3.3,
    202104: 4.1,
    202103: 4.8,
    202102: 3.6,
    202101: 4.0,
    202012: 4.0,
    202011: 3.2,
    202010: 3.8,
    202009: 2.8,
    202008: 2.7,
    202007: 1.9,
    202006: 2.2,
    202005: 1.5,
    202004: 1.5,
    202003: 3.3,
    202002: 2.0,
    202001: 2.3,
    201912: 3.7,
    201911: 4.3,
    201910: 3.3,
    201909: 5.9,
    201908: 4.0,
    201907: 2.2,
    201906: 2.7,
    201905: 3.1,
    201904: 3.4,
    201903: 4.7,
    201902: 3.8,
    201901: 2.9
    }

    df_ipc = pl.DataFrame({
        "foto_mes": list(dict_ipc.keys()),
        "inflacion": list(dict_ipc.values())
    }).sort("foto_mes")

    # 2. Calcular el Factor de Actualización 
    df_factores = df_ipc.with_columns(
        (1 / (pl.col("inflacion") / 100 + 1).cum_prod()).alias("factor_actualizacion")
    ).with_columns(
        pl.when(pl.col("foto_mes") == mes_base)
          .then(1.0)
          .otherwise(pl.col("factor_actualizacion").shift(1))
          .alias("factor_actualizacion_ajustado")
    ).select(
        pl.col("foto_mes"),
        pl.col("factor_actualizacion_ajustado").alias("factor_actualizacion")
    )

    df = df.join(df_factores, on="foto_mes", how="left")

    expresiones = [
        (pl.col(feature) * pl.col("factor_actualizacion")).alias(feature) 
        for feature in columnas
    ]

    # Aplicar las nuevas columnas (sobrescribe las originales)
    df = df.with_columns(expresiones)

    logger.info(f"Feature engineering [IPC] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df.drop("factor_actualizacion")

def feature_engineering_rank(df: pl.DataFrame, columnas: list[str], group_col: str = "foto_mes") -> pl.DataFrame:
    """
    - Para cada columna en `columnas`, calcula ranking percentil firmado por grupo (group_col).
    - Valores > 0: percentil en (0, 1] calculado solo entre los positivos del grupo.
    - Valores < 0: percentil en [-1, 0) calculado por magnitud solo entre los negativos del grupo.
    - Valores == 0 -> 0.0
    - Nulls se mantienen como Null (None).
    """

    logger.info(f"Realizando rankings para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar rankings")
        return df

    group = group_col

    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"Advertencia: El atributo {attr} no existe en el DataFrame")
            continue

        # contar positivos / negativos por grupo (True->1 sumado por ventana)
        pos_count = (pl.col(attr) > 0).cast(pl.Int64).sum().over(group)
        neg_count = (pl.col(attr) < 0).cast(pl.Int64).sum().over(group)

        # Rank entre positivos: rank de (valor si >0 else null) sobre grupo, dividido por pos_count
        pos_rank = (
            pl.when(pl.col(attr) > 0)
            .then(pl.col(attr))
            .otherwise(None)
            .rank(method="average")
            .over(group)
            / pos_count
        )

        # Rank entre negativos por magnitud: rank de (abs(valor) si <0 else null) sobre grupo,
        # dividido por neg_count, y luego negado.
        neg_rank = (
            pl.when(pl.col(attr) < 0)
            .then(pl.col(attr).abs())
            .otherwise(None)
            .rank(method="average")
            .over(group)
            / neg_count
        )
        neg_rank = -neg_rank

        # Combinamos: si original es null -> None; si ==0 -> 0.0; sino pos_rank o neg_rank
        expr = (
            pl.when(pl.col(attr).is_null()).then(None)
            .when(pl.col(attr) == 0).then(0.0)
            .otherwise(pl.coalesce([pos_rank, neg_rank]))
            .alias(attr)
        )

        df = df.with_columns(expr)
        gc.collect()

    # mantener mismo orden de columnas original
    df = df.select(df.columns)
    logger.info(f"Feature engineering [ranks] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    # mlflow.log_param("flag_rankings", True)

    return df


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], lags: list = [1, 2]) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    lags : list 
        Cantidad de lags a generar para cada atributo

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {lags} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"
    # sql = "SELECT CAST(STRFTIME(foto_mes::DATE, '%Y%m') AS INTEGER) as foto_mes, * EXCLUDE(foto_mes)"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in lags:
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
    df = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering [lags] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def feature_engineering_trend(df: pl.DataFrame, columnas: list[str], q=3) -> pl.DataFrame:
    """
    Genera variables de tendencia para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar tendencias. Si es None, no se generan tendencias.

    Returns:
    --------
    pl.DataFrame
        DataFrame con las variables de tendencia agregadas
    """

    logger.info(f"Realizando feature engineering de tendencia de {q} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f", regr_slope({attr}, cliente_antiguedad) over ventana as {attr}_trend_{q}m"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += f" window ventana as (partition by numero_de_cliente order by foto_mes rows between {q} preceding and current row)"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).pl()
    con.close()

    logging.info(df.head())

    logger.info(f"Feature engineering [trends] completado")
    logger.info(df.shape)

    # mlflow.log_param("flag_trend", True)

    return df


def feature_engineering_delta(df: pd.DataFrame, columnas: list[str], lags: list = [1, 2]) -> pd.DataFrame:
    """
    Genera variables delta (attr - attr_lag_i) usando Polars.
    """
    logger.info(f"Comienzo feature delta. df shape: {df.shape}")

    # df = pl.from_pandas(df)

    exprs = []
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"No se encontró {attr}, se omite.")
            continue

        for i in lags:
            lag_col = f"{attr}_lag_{i}"
            if lag_col not in df.columns:
                logger.warning(f"No se encontró {lag_col}, se omite.")
                continue

            exprs.append((pl.col(attr) - pl.col(lag_col)).alias(f"{attr}_delta_{i}"))

    if exprs:
        df = df.with_columns(exprs)

    # df = df.to_pandas()

    logger.info(f"Feature engineering [deltas] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df


def fix_aguinaldo(df: pl.DataFrame) -> pl.DataFrame:
    """
    Corrige las columnas asociadas al pago de aguinaldo para el mes de junio 2021
    :param df: dataframe con los datos
    :return: dataframe corregido
    """

    columns = df.columns

    logger.info(f"Inicia fix de variables por aguinaldo")
    sql = """
    with aguinaldo as (
    SELECT foto_mes,
           numero_de_cliente,
           mpayroll_delta_1,
           flag_aguinaldo,
           --cpayroll_trx
    FROM (
    SELECT foto_mes,
           numero_de_cliente,
           lag(mpayroll, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag_1,
           lag(mpayroll, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag_2,
           mpayroll - mpayroll_lag_1 as mpayroll_delta_1,
           case when foto_mes in (202106, 202012, 202006, 201912, 201906)
                and mpayroll/mpayroll_lag_1  >= 1.3
                and mpayroll/mpayroll_lag_2  >= 1.3
                    then 1
                else 0
           end as flag_aguinaldo,
           --case when flag_aguinaldo = 1 and cpayroll_trx > 1 then cpayroll_trx - 1 else cpayroll_trx end as cpayroll_trx
    FROM df) as a
    WHERE foto_mes in (202106, 202012, 202006, 201912, 201906))

    SELECT df.* REPLACE(
                case when aguinaldo.mpayroll_delta_1 is null or aguinaldo.flag_aguinaldo = 0 then df.mpayroll when df.foto_mes in (202106, 202012, 202006, 201912, 201906) then df.mpayroll - aguinaldo.mpayroll_delta_1 + aguinaldo.mpayroll_delta_1/6 else df.mpayroll + aguinaldo.mpayroll_delta_1/6 end as mpayroll
                --,case when aguinaldo.mpayroll_delta_1 is null then df.cpayroll_trx when df.foto_mes = 202106 then aguinaldo.cpayroll_trx else df.cpayroll_trx end as cpayroll_trx
                ), mpayroll as mpayroll_original
    FROM df
    LEFT JOIN aguinaldo
    ON df.numero_de_cliente = aguinaldo.numero_de_cliente
    AND aguinaldo.foto_mes =
    CASE
        WHEN MOD(df.foto_mes, 100) <= 6 
        THEN (TRUNC(df.foto_mes / 100) * 100) + 6
        ELSE (TRUNC(df.foto_mes / 100) * 100) + 12
    END
    ORDER BY df.numero_de_cliente, df.foto_mes
    """

    con = duckdb.connect(database=":memory:")
    df = con.execute(sql).pl()
    con.close()
    logger.info(f"Finaliza fix de variables por aguinaldo")

    df = df.select(columns)

    return df


def fix_zero_sd(df: pl.DataFrame, columnas: list) -> pl.DataFrame:
    """
    Identifica qué atributos (columnas) tienen una desviación estándar de 0
    para cada grupo de 'foto_mes' en un DataFrame de Polars.

    Args:
        df: Un DataFrame de Polars que debe contener la columna 'foto_mes'.

    Returns:
        Un DataFrame de Polars con las columnas ['foto_mes', 'atributo', 'std_dev'],
        mostrando solo las combinaciones donde std_dev es 0.
    """

    # Agrupar por 'foto_mes' y calcular la desviación estándar
    df_std = df.group_by("foto_mes").agg(
        pl.col(columnas).std()
    )

    # Reformatear
    df_long = df_std.unpivot(
        index="foto_mes",
        on=columnas,
        variable_name="atributo",
        value_name="std_dev"
    )

    # Filtrar solo los atributos con std_dev == 0
    columnas_zero_sd = df_long.filter(
        pl.col("std_dev") == 0
    )

    # Si no hay nada que nulificar, retornamos el DF original
    if columnas_zero_sd.height == 0:
        logging.info("No se encontraron atributos constantes.")
        return df

    # Agrupamos por 'atributo' para obtener una LISTA de 'foto_mes'
    # donde ese atributo debe ser nulificado.
    mapa_constantes = columnas_zero_sd.group_by("atributo").agg(
        pl.col("foto_mes")
    )

    # Convertimos a un diccionario de Python para fácil acceso.
    mapa = dict(
        zip(
            mapa_constantes.get_column("atributo").to_list(),
            mapa_constantes.get_column("foto_mes").to_list()
        )
    )

    logging.info(f"Mapa de atributos constantes: {mapa}")

    expresiones = []

    # Iteramos SOLO sobre las columnas que sabemos que tienen constantes
    for col_name in mapa.keys():
        # Obtenemos la lista de meses a nulificar para esta columna
        meses_a_nulificar = mapa[col_name]

        # Creamos la expresión
        expr = (
            pl.when(pl.col("foto_mes").is_in(meses_a_nulificar))
            .then(None)  # Si el mes está en la lista -> Null
            .otherwise(pl.col(col_name))  # Si no -> Mantener valor original
            .alias(col_name)  # Sobrescribir la columna
        )

        expresiones.append(expr)

    # mlflow.log_param("flag_data_quality", True)

    return df.with_columns(expresiones)




import polars as pl
import plotly.express as px
from datetime import date

def generar_reporte_mensual_html(
    df: pl.DataFrame,
    columna_fecha: str = 'foto_mes',
    columna_target: str | None = None,
    nombre_archivo: str = 'reporte_atributos.html'
):
    """
    Genera un archivo HTML con gráficos de líneas interactivos de Plotly,
    mostrando la evolución mensual del promedio de cada feature, usando Polars.
    """

    print(f"Iniciando generación de reporte (con Polars) para {len(df.columns) - 1} features...")

    # -------------------------------------------------------------------------
    # 0. Convertir foto_mes (YYYYMM) a fecha real YYYY-MM-01
    # -------------------------------------------------------------------------
    df = df.with_columns([
        pl.col(columna_fecha)
        .cast(pl.Utf8)                          # convertir a string
        .str.concat("01")                       # agregar día artificial
        .str.strptime(pl.Date, "%Y%m%d")        # parsear como fecha
        .alias(columna_fecha)
    ])

    # --- 1. Preparación de datos ------------------------------------------------

    exclusion_cols = [columna_fecha]
    if columna_target:
        exclusion_cols.append(columna_target)

    features_numericas = df.select(
        pl.selectors.numeric().exclude(exclusion_cols)
    ).columns

    if not features_numericas:
        print("Error: No se encontraron features numéricas para graficar.")
        return

    df_agrupado = df.group_by(columna_fecha).agg(
        pl.col(features_numericas).mean()
    ).sort(columna_fecha)

    # Preparación con target
    df_agrupado_con_target = None
    if columna_target and columna_target in df.columns:
        df_agrupado_con_target = df.group_by([columna_fecha, columna_target]).agg(
            pl.col(features_numericas).mean()
        ).sort(columna_fecha, columna_target)

    # --- 2. Generación de gráficos ------------------------------------------------

    graficos_html = []

    for feature in features_numericas:

        fig_general = px.line(
            df_agrupado,
            x=columna_fecha,
            y=feature,
            title=f'Evolución Mensual de {feature} (Promedio General)'
        )

        fig_general.update_traces(mode='lines+markers')
        fig_general.update_layout(
            xaxis_title="Fecha",
            yaxis_title=f"Promedio de {feature}",
            title_x=0.5
        )

        html_string_general = fig_general.to_html(full_html=False, include_plotlyjs='cdn')
        graficos_html.append(f"""
            <div style="width: 80%; margin: 40px auto; border: 1px solid #ddd; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                {html_string_general}
            </div>
        """)

        if df_agrupado_con_target is not None:
            fig_target = px.line(
                df_agrupado_con_target,
                x=columna_fecha,
                y=feature,
                color=columna_target,
                title=f'Evolución Mensual de {feature} (Promedio por {columna_target})'
            )
            fig_target.update_traces(mode='lines+markers')
            fig_target.update_layout(
                xaxis_title="Fecha",
                yaxis_title=f"Promedio de {feature}",
                title_x=0.5
            )

            html_string_target = fig_target.to_html(full_html=False, include_plotlyjs='cdn')
            graficos_html.append(f"""
                <div style="width: 80%; margin: 40px auto; border: 1px solid #ddd; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    {html_string_target}
                </div>
            """)

        graficos_html.append("<hr style='border: 1px solid #ccc; margin-top: 20px;'>")

    # --- 3. Template HTML final ---------------------------------------------------

    fecha_min = df_agrupado[columna_fecha].min()
    fecha_max = df_agrupado[columna_fecha].max()

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Evolución Mensual (Polars)</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; }}
            h1 {{ text-align: center; color: #333; padding: 20px; background-color: #fff; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Evolución Mensual de Features ({fecha_min} a {fecha_max})</h1>
        {''.join(graficos_html)}
    </body>
    </html>
    """

    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"\n✅ Reporte HTML generado exitosamente: {nombre_archivo}")

def create_features(df: pl.DataFrame) -> pl.DataFrame:

    logging.info("==== Creando nuevas features")
    df = (df.with_columns([
        (pl.col("foto_mes").cast(pl.Int64) % 100).alias("kmes"),
        (pl.col("Master_delinquency").fill_null(0) + pl.col("Visa_delinquency").fill_null(0)).alias("tc_delinquency"),
        (pl.col("Master_status").fill_null(0) + pl.col("Visa_status").fill_null(0)).alias("tc_status"),
        (pl.col("Master_mfinanciacion_limite").fill_null(0) + pl.col("Visa_mfinanciacion_limite").fill_null(0)).alias("tc_mfinanciacion_limite"),
        (pl.col("Master_msaldototal").fill_null(0) + pl.col("Visa_msaldototal").fill_null(0)).alias("tc_msaldototal"),
        (pl.col("Master_msaldopesos").fill_null(0) + pl.col("Visa_msaldopesos").fill_null(0)).alias("tc_msaldopesos"),
        (pl.col("Master_msaldodolares").fill_null(0) + pl.col("Visa_msaldodolares").fill_null(0)).alias("tc_msaldodolares"),
        (pl.col("Master_mconsumospesos").fill_null(0) + pl.col("Visa_mconsumospesos").fill_null(0)).alias("tc_mconsumospesos"),
        (pl.col("Master_mconsumosdolares").fill_null(0) + pl.col("Visa_mconsumosdolares").fill_null(0)).alias("tc_mconsumosdolares"),
        (pl.col("Master_mlimitecompra").fill_null(0) + pl.col("Visa_mlimitecompra").fill_null(0)).alias("tc_mlimitecompra"),
        (pl.col("Master_madelantopesos").fill_null(0) + pl.col("Visa_madelantopesos").fill_null(0)).alias("tc_madelantopesos"),
        (pl.col("Master_madelantodolares").fill_null(0) + pl.col("Visa_madelantodolares").fill_null(0)).alias(
            "Visa_madelantodolares"),
        (pl.col("Master_mpagado").fill_null(0) + pl.col("Visa_mpagado").fill_null(0)).alias(
            "tc_mpagado"),
        (pl.col("Master_mpagospesos").fill_null(0) + pl.col("Visa_mpagospesos").fill_null(0)).alias(
            "tc_mpagospesos"),
        (pl.col("Master_mpagosdolares").fill_null(0) + pl.col("Visa_mpagosdolares").fill_null(0)).alias(
            "tc_mpagosdolares"),
        (pl.col("Master_mconsumototal").fill_null(0) + pl.col("Visa_mconsumototal").fill_null(0)).alias(
            "tc_mconsumototal"),
        (pl.col("Master_cconsumos").fill_null(0) + pl.col("Visa_cconsumos").fill_null(0)).alias(
            "tc_cconsumos"),
        (pl.col("Master_cadelantosefectivo").fill_null(0) + pl.col("Visa_cadelantosefectivo").fill_null(0)).alias(
            "tc_cadelantosefectivo"),
        (pl.col("Master_mpagominimo").fill_null(0) + pl.col("Visa_mpagominimo").fill_null(0)).alias(
            "tc_mpagominimo"),
    ])
    .with_columns([
        (pl.col("tc_mpagado") / pl.col("tc_msaldopesos")).alias("tc_porc_pagos_saldo"), # agregar tratamientos de division por cero
        (pl.col("tc_msaldototal") / pl.col("tc_mlimitecompra")).alias("tc_porc_utilizacion"),
        (pl.col("tc_mpagado") / pl.col("tc_mpagominimo")).alias("tc_porc_pago_pagominimo"),
        (pl.col("tc_mpagominimo") / pl.col("tc_mlimitecompra")).alias("tc_porc_pagominimo_saldo"),
        (pl.when(pl.col("tc_mpagado") < pl.col("tc_mpagominimo"))
        .then(1)
        .otherwise(0)
        .alias("tc_flag_pago_pagominimo"))
    ]))

    logger.info(f"Feature engineering [create_features] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def create_canaritos(df: pl.DataFrame, qcanaritos: int = 100) -> pl.DataFrame:
    """
    Igual que tu versión original, pero generando una sola matriz numpy
    para evitar usar mucha memoria.
    """
    logging.info(f"==== Creando {qcanaritos} canaritos...")

    original_cols = df.columns
    num_filas = df.height

    # 👉 Generamos TODO en un solo array (mucho más eficiente)
    canary_matrix = np.random.rand(num_filas, qcanaritos)

    # 👉 Convertimos cada columna numpy → Polars
    canary_expressions = []
    for i in range(qcanaritos):
        name = f"canarito_{i+1}"
        col = pl.Series(name, canary_matrix[:, i])
        canary_expressions.append(col)

    # 👉 Añadimos las nuevas columnas
    df = df.hstack(canary_expressions)

    # 👉 Reordenamos: canarios primero, luego las originales
    df = df.select([f"canarito_{i+1}" for i in range(qcanaritos)] + original_cols)

    logger.info(f"Feature engineering [create_canaritos] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df.clone()

from sklearn.ensemble import RandomTreesEmbedding
def create_embedding_rf(df: pl.DataFrame) -> pl.DataFrame:

    X_train, y_train, X_test, y_test, feature_name = train_test_split(df=df, undersampling=False, mes_train=[202101, 202102, 202103], mes_test=[202104])
    X = df.select(pl.all().exclude(["target", "target_test"])).to_numpy().astype("float32")

    embedding_model = RandomTreesEmbedding(
        n_estimators=20,
        max_leaf_nodes=16,
        min_samples_leaf=100,
        random_state=SEMILLA[1]
    )

    embedding_model.fit(X_train)

    X_embedded_sparse = embedding_model.transform(X)

    n_cols = X_embedded_sparse.shape[1]
    columnas_nuevas = [f"rf_{i:06d}" for i in range(n_cols)]

    df_embedding = pd.DataFrame.sparse.from_spmatrix(
        X_embedded_sparse,
        columns=columnas_nuevas
    )

    df_embedding = pl.from_pandas(df_embedding)
    df_embedding = df_embedding.with_columns(
        [pl.col(col).cast(pl.Int8) for col in df_embedding.columns]
    )
    df = pl.concat([df, df_embedding], how="horizontal")

    return df, embedding_model


def create_embedding_lgbm_rf(df: pl.DataFrame):

    X_train, y_train, X_test, y_test, feature_name = train_test_split(
        df=df,
        undersampling=False,
        mes_train=[202101, 202102, 202103],
        mes_test=[202104]
    )

    X_all = df.select(pl.all().exclude(["target", "target_test"])).to_numpy().astype("float32")

    # --- Random Forest supervisado en LightGBM
    params = {
        "objective": "binary",
        "num_leaves": 16,
        "max_depth": -1,
        "feature_fraction": 1,
        "feature_fraction_bynode": 0.2,
        "bagging_fraction": ( 1.0 - 1.0/np.exp(1.0)),
        "bagging_freq": 1,
        "num_iterations": 20,
        "boosting": "rf",

        # genericos de LightGBM
        "max_bin": 31,
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,

        "pos_bagging_fraction": 1.0,
        "neg_bagging_fraction": 1.0,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,

        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,

        "extra_trees": False
    }

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True, feature_name=feature_name)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=params["num_iterations"]
    )

    leaf_matrix = model.predict(X_all, pred_leaf=True)
    encoder = OneHotEncoder(
        sparse_output=True,
        handle_unknown="ignore",
        dtype=np.int8
    )

    leaf_sparse = encoder.fit_transform(leaf_matrix)

    columnas_nuevas = [f"rf_{i:06d}" for i in range(leaf_sparse.shape[1])]

    df_embedding = sparse_to_polars(leaf_sparse, chunk_size=300)
    df_final = pl.concat([df, df_embedding], how="horizontal")

    logging.info(f"Feature engineering [embedding_rf] completado")
    logging.info(f"Filas: {df_final.height}, Columnas: {df_final.width}")
    logging.info(df_final.head())

    return df_final

def sparse_to_polars(df_sparse, chunk_size=500):
    """
    Convierte una matriz sparse CSR (scipy) a un DataFrame de Polars
    sin pasar por pandas. Usa bloques para no romper la RAM.
    """
    n_rows, n_cols = df_sparse.shape
    columns = []

    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)

        block = df_sparse[:, start:end].toarray().astype(np.uint8)
        # crear columnas polars para este chunk
        for j in range(block.shape[1]):
            col_name = f"rf_{start+j:06d}"
            columns.append(pl.Series(col_name, block[:, j]))

    return pl.DataFrame(columns)