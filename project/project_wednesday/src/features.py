import pandas as pd
import polars as pl
import numpy as np
import duckdb
import logging
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from .config import *
from .basic_functions import train_test_split
from .best_params import cargar_mejores_hiperparametros
from .test_evaluation import calcular_ganancias_acumuladas
from .basic_functions import generar_semillas
from datetime import datetime
import os
import gc
import plotly.express as px
from io import StringIO
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def feature_engineering_rank_cero_fijo(df: pl.DataFrame, columnas: list[str], group_col: str = "foto_mes") -> pl.DataFrame:
    """
    - Para cada columna en `columnas`, calcula ranking percentil firmado por grupo (group_col).
    - Valores > 0: percentil en (0, 1] calculado solo entre los positivos del grupo.
    - Valores < 0: percentil en [-1, 0) calculado por magnitud solo entre los negativos del grupo.
    - Valores == 0 -> 0.0
    - Nulls se mantienen como Null (None).
    """

    logger.info(f"🔄 Feature engineering [ranking con cero fijo] en proceso")

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
            .alias(f"rankcf_{attr}")
        )

        df = df.with_columns(expr)
        gc.collect()

    # mantener mismo orden de columnas original
    df = df.select(df.columns)
    logger.info(f"✅ Feature engineering [ranking con cero fijo] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def feature_engineering_percent_rank(
    df: pl.DataFrame,
    columnas: list[str],
    ) -> pl.DataFrame:
    """
    Replica percent_rank() de SQL usando Polars.

    percent_rank = (rank - 1) / (count - 1)

    Si count == 1 -> 0.0 (igual que SQL)
    Nulls se mantienen como Null.
    """

    if not columnas:
        logger.warning("No se especificaron columnas para percent_rank()")
        return df

    out = df

    logger.info(f"🔄 Feature engineering [percent rank] en proceso")

    for col in columnas:
        if col not in df.columns:
            logger.warning(f"[percent_rank] advertencia: columna {col} no existe")
            continue

        # rank dentro del grupo
        base_rank = pl.col(col).rank(method="average").over("foto_mes")

        # cantidad por grupo
        cnt = pl.count().over("foto_mes")

        # Fórmula SQL
        pr_expr = (
            pl.when(cnt == 1)
            .then(0.0)
            .otherwise((base_rank - 1) / (cnt - 1))
            .alias(f"rankp_{col}")
        )

        out = out.with_columns(pr_expr)

    logger.info(f"✅ Feature engineering [percent rank] completado")
    logger.info(f"Filas: {out.height}, Columnas: {out.width}")

    return out

def feature_engineering_ntile(
    df: pl.DataFrame,
    columnas: list[str],
    k: int = 10,
    ) -> pl.DataFrame:
    """
    Replica NTILE() de SQL usando Polars.

    ```
    Para cada columna en `columnas`, dentro de cada grupo:
        ntile = ceil( rank / count * buckets )

    Nulls quedan en Null.

    Parameters
    ----------
    df : pl.DataFrame
    columnas : list[str]
        Columnas numéricas a transformar.
    buckets : int
        Cantidad de tiles (ej: 10 = deciles).
    group_col : str
        Columna de agrupamiento (default: "foto_mes").
    """

    if not columnas:
        logger.warning("No se especificaron columnas para ntile()")
        return df

    logger.info(f"🔄 Feature engineering [ntiles] en proceso")

    out = df

    for col in columnas:
        if col not in df.columns:
            logger.warning(f"Advertencia: columna {col} no existe en el DataFrame")
            continue

        # rank dentro de grupo
        base_rank = pl.col(col).rank(method="average").over("foto_mes")

        # cantidad total por grupo
        cnt = pl.count().over("foto_mes")

        # fórmula tipo SQL:
        # ntile = ceil(rank / count * buckets)
        ntile_expr = (
            ((base_rank / cnt) * k)
            .ceil()
            .cast(pl.Int64)
            .alias(f"rankn_{col}")
        )

        out = out.with_columns(ntile_expr)

    logger.info(f"✅ Feature engineering [ntiles] completado")
    logger.info(f"Filas: {out.height}, Columnas: {out.width}")

    return out

def feature_engineering_percent_rank_dense(
    df: pl.DataFrame,
    columnas: list[str]
    ) -> pl.DataFrame:
    """
    Replica percent_rank() de SQL pero usando DENSE_RANK en vez de AVERAGE_RANK.

    - dense_rank no deja huecos cuando hay empates.
    - percent_rank_dense = (dense_rank - 1) / (unique_count - 1)
    - Si unique_count == 1 -> 0.0

    Nulls se mantienen como Null.
    """

    logger.info(f"🔄 Feature engineering [dense rank] en proceso")

    if not columnas:
        logger.warning("No se especificaron columnas para percent_rank_dense()")
        return df

    out = df

    for col in columnas:
        if col not in df.columns:
            logger.warning(f"[percent_rank_dense] advertencia: columna {col} no existe")
            continue

        # dense rank dentro del grupo
        dense_rank = pl.col(col).rank(method="dense").over("foto_mes")

        # cantidad de valores distintos por grupo
        uniq = pl.col(col).n_unique().over("foto_mes")

        # Fórmula tipo SQL
        pr_expr = (
            pl.when(uniq == 1)
            .then(0.0)
            .otherwise((dense_rank - 1) / (uniq - 1))
            .alias(f"rankpd_{col}")
        )

        out = out.with_columns(pr_expr)

    logger.info(f"✅ Feature engineering [dense rank] completado")
    logger.info(f"Filas: {out.height}, Columnas: {out.width}")

    return out


def feature_engineering_min_max(df: pl.DataFrame, columnas: list[str], window: int) -> pl.DataFrame:
    """
    Genera variables de minimos y/o maximos temporales para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    min :  indicador para calcular el valor minimo
    max :  indicador para calcular el valor maximo
    window :  cantidad de meses a considerar para el calculo

    columnas : list
        Lista de atributos para los cuales generar minimos y maximos. Si es None, no se generan atributos.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de minimos y maximos agregadas
    """

    logger.info(f"🔄 Feature engineering [min/max/avg {window}m] en proceso")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar atributos")
        return df

    # Construir la consulta SQL
    sql = "SELECT foto_mes, numero_de_cliente"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f', MIN({attr}) over ventana as {attr}_min{window}'
            sql += f', MAX({attr}) over ventana as {attr}_max{window}'
            sql += f', AVG({attr}) over ventana as {attr}_avg{window}'

        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += f" window ventana as (partition by numero_de_cliente order by foto_mes rows between {window-1} preceding and current row)"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df.select(["numero_de_cliente", "foto_mes"] + columnas))
    df_new = con.execute(sql).pl()
    con.close()

    # Merge al dataframe original
    df = df.join(
        df_new,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )

    logger.info(f"✅ Feature engineering [min/max/avg] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def feature_engineering_ratioavg(df: pl.DataFrame, columnas: list[str], window: int) -> pl.DataFrame:
    """
    Genera variables ratioavg para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    window :  cantidad de meses a considerar para el calculo

    columnas : list
        Lista de atributos para los cuales generar ratioavg. Si es None, no se generan atributos.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables ratioavg agregadas
    """

    logger.info(f"🔄 Feature engineering [ratioavg {window}m] en proceso")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar atributos")
        return df

    columnas_avg = [f"{c}_avg{window}" for c in columnas]

    # Construir la consulta SQL
    sql = "SELECT foto_mes, numero_de_cliente"

    for attr in columnas:
        if attr in df.columns:
            sql += f', {attr} / NULLIF({attr}_avg{window}, 0) AS {attr}_ratioavg{window}'

        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df.select(["numero_de_cliente", "foto_mes"] + columnas + columnas_avg))
    df_new = con.execute(sql).pl()
    con.close()

    df = df.join(
        df_new,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )

    logger.info(f"✅ Feature engineering [ratioavg] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def feature_engineering_lag(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
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

    logger.info(f"🔄 Feature engineering [lags] en proceso")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT foto_mes, numero_de_cliente"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df.select(["numero_de_cliente", "foto_mes"] + columnas))
    df_new = con.execute(sql).pl()
    con.close()

    # Merge al dataframe original
    df = df.join(
        df_new,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )


    logger.info(f"✅ Feature engineering [lags] completado")
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

    logger.info(f"🔄 Feature engineering [trend {q}m] en proceso")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT foto_mes, numero_de_cliente"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f", regr_slope({attr}, cliente_antiguedad) over ventana as {attr}_trend_{q}m"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += f" window ventana as (partition by numero_de_cliente order by foto_mes rows between {q-1} preceding and current row)"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df.select(["numero_de_cliente", "foto_mes", 'cliente_antiguedad'] + columnas))
    df_new = con.execute(sql).pl()
    con.close()

    # Merge al dataframe original
    df = df.join(
        df_new,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )

    logger.info(f"✅ Feature engineering [trend {q}m] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df


def feature_engineering_delta(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    """
    Genera variables delta (attr - attr_lagi) usando Polars.
    """
    logger.info(f"🔄 Feature engineering [deltalags] en proceso")

    exprs = []
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"No se encontró {attr}, se omite.")
            continue

        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag{i}"
            if lag_col not in df.columns:
                logger.warning(f"No se encontró {lag_col}, se omite.")
                continue

            exprs.append((pl.col(attr) - pl.col(lag_col)).alias(f"{attr}_delta{i}"))

    if exprs:
        df = df.with_columns(exprs)

    logger.info(f"✅ Feature engineering [deltalags] completado")
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
           lag(mpayroll, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag1,
           lag(mpayroll, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)  as mpayroll_lag2,
           mpayroll - mpayroll_lag1 as mpayroll_delta_1,
           case when foto_mes in (202106, 202012, 202006, 201912, 201906)
                and mpayroll/mpayroll_lag1  >= 1.3
                and mpayroll/mpayroll_lag2  >= 1.3
                    then 1
                else 0
           end as flag_aguinaldo,
           --case when flag_aguinaldo = 1 and cpayroll_trx > 1 then cpayroll_trx - 1 else cpayroll_trx end as cpayroll_trx
    FROM df) as a
    WHERE foto_mes in (202106, 202012, 202006, 201912, 201906))

    SELECT df.* REPLACE(
                case when aguinaldo.mpayroll_delta_1 is null or aguinaldo.flag_aguinaldo = 0 then df.mpayroll when df.foto_mes in (202106, 202012, 202006, 201912, 201906) then df.mpayroll - aguinaldo.mpayroll_delta_1 + aguinaldo.mpayroll_delta_1/6 else df.mpayroll + aguinaldo.mpayroll_delta_1/6 end as mpayroll
                --,case when aguinaldo.mpayroll_delta_1 is null then df.cpayroll_trx when df.foto_mes = '2021-06-30' then aguinaldo.cpayroll_trx else df.cpayroll_trx end as cpayroll_trx
                ), mpayroll as mpayroll_original
    FROM df
    LEFT JOIN aguinaldo
    ON df.numero_de_cliente = aguinaldo.numero_de_cliente
    AND aguinaldo.foto_mes =
    CASE
        -- El operador % (módulo) obtiene los últimos 2 dígitos (el mes)
        WHEN foto_mes % 100 <= 6 
        -- La división / 100 obtiene los primeros 4 dígitos (el año)
        THEN make_date((foto_mes / 100)::INT, 6, 30)
        ELSE make_date((foto_mes / 100)::INT, 12, 31)
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

    logger.info(f"✅ Data Cleaning [zero_sd] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

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

    logger.info(f"🔄 Feature engineering [create_features] en proceso")
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
        # ctrx_quarter_normalizado
        pl.when(pl.col("cliente_antiguedad") == 1)
        .then(pl.col("ctrx_quarter") * 5.0)
        .when(pl.col("cliente_antiguedad") == 2)
        .then(pl.col("ctrx_quarter") * 2.0)
        .when(pl.col("cliente_antiguedad") == 3)
        .then(pl.col("ctrx_quarter") * 1.2)
        .otherwise(pl.col("ctrx_quarter"))
        .alias("ctrx_quarter_normalizado"),

        (pl.col("mpayroll").fill_null(0) / pl.col("cliente_edad")).alias(
            "mpayroll_sobre_edad")
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

    logger.info(f"✅ Feature engineering [create_features] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df

def create_canaritos(df: pl.DataFrame, qcanaritos: int = 100) -> pl.DataFrame:
    """
    Crea variables aleatorias para usar en zLightGBM o reduccion de dimensionalidad
    """
    logging.info(f"==== Creando {qcanaritos} canaritos...")

    original_cols = df.columns
    num_filas = df.height

    canary_matrix = np.random.rand(num_filas, qcanaritos)

    canary_expressions = []
    for i in range(qcanaritos):
        name = f"canarito_{i+1}"
        col = pl.Series(name, canary_matrix[:, i])
        canary_expressions.append(col)

    df = df.hstack(canary_expressions)

    df = df.select([f"canarito_{i+1}" for i in range(qcanaritos)] + original_cols)

    logger.info(f"Feature engineering [create_canaritos] completado")
    logger.info(f"Filas: {df.height}, Columnas: {df.width}")

    return df


def create_embedding_lgbm_rf(df: pl.DataFrame):

    df = df.with_columns([
            pl.when(pl.col("target") == "CONTINUA").then(0).otherwise(1).alias("target_train"),
            pl.when(pl.col("target") == "BAJA+2").then(1).otherwise(0).alias("target_test"),
            pl.when(pl.col("target") == "CONTINUA").then(1)
            .when(pl.col("target") == "BAJA+1").then(1.00001)
            .when(pl.col("target") == "BAJA+2").then(1.00002)
            .otherwise(None)
            .alias("w_train")
        ])

    X_train, y_train, X_test, y_test, w_train = train_test_split(
        df=df,
        undersampling=False,
        mes_train=[202101, 202102, 202103, 202104],
        mes_test=[202105]
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

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

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

def run_canaritos_asesinos(df: pl.DataFrame, qcanaritos: int = 50, params_path: str = None, ksemillerio: int = 5, metric: float = 0.5) -> pl.DataFrame:

    logger.info("==== Iniciando Canaritos Asesinos ====")
    df_with_canaritos = create_canaritos(df, qcanaritos)
    features = [c for c in df_with_canaritos.columns if c not in ["target", "target_train", "target_test", "w_train"]]
    X_train, y_train, X_test, y_test, w_train = train_test_split(df=df_with_canaritos, undersampling=False, mes_train=MES_TRAIN, mes_test=MES_TEST)

    logging.info(X_train.shape)
    logging.info(X_test.shape)

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        feature_name=features,
        free_raw_data=True
    )

    mejores_params = cargar_mejores_hiperparametros(archivo_base = params_path)

    # Hiperparámetros fijos
    params = {
        'objective': 'binary',
        'metric': 'None',
        'verbose': -1,
        'verbosity': -1,
        'silent': 1,
        'boosting': 'gbdt',
        'n_threads': -1,
        'feature_pre_filter': PARAMETROS_LGB['feature_pre_filter'],
        'force_row_wise': PARAMETROS_LGB['force_row_wise'],  # para reducir warnings
        'max_bin': PARAMETROS_LGB['max_bin'],
        'seed': SEMILLA[0]
    }

    final_params = {**params, **mejores_params}
    logger.info(f"Parámetros del modelo: {final_params}")

    semillas = generar_semillas(SEMILLA[0], ksemillerio)
    all_importances = []
    # Inicializamos acumulador de predicciones para calcular promedio parcial
    pred_acumulada = np.zeros(len(y_test))

    for i, seed in enumerate(semillas):
        logging.info(f'Entrenando modelo base con seed = {seed} ({i+1}/{len(semillas)})')

        # Copia de parámetros con la semilla actual
        params_seed = final_params.copy()
        params_seed['seed'] = seed

        model = lgb.train(final_params, train_data)

        logging.info(f'Fin de entrenamiento del modelo base con seed = {seed} ({i + 1}/{len(semillas)})')

        y_pred_actual = model.predict(X_test)
        pred_acumulada += y_pred_actual

        # Importancia del modelo
        feature_imp = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type='gain')
        })

        logging.info(feature_imp.head())

        all_importances.append(feature_imp)

    y_pred = pred_acumulada / len(semillas)

    # Calculo la ganancia del modelo
    ganancias = calcular_ganancias_acumuladas(y_test, y_pred)
    ganancias_meseta = (
        pd.Series(ganancias)
        .rolling(window=2001, center=True, min_periods=1)
        .mean()
    )
    max_ganancia = ganancias_meseta.max(skipna=True)

    logging.info(f"Ganancia base obtenida en test: {max_ganancia}")

    # Calculo la importancia de las variables del modelo
    df_importances = (
        pd.concat(all_importances, axis=0)
        .groupby("feature", as_index=False)
        .mean()
        .sort_values("importance", ascending=False)
    )

    canaritos_in_top = []
    for i, row in df_importances.iterrows():
        feat = row['feature']
        imp = row['importance']

        if feat.startswith("canarito_"):
            canaritos_in_top.append((feat, i + 1, imp))

    logger.info(f"=== ANÁLISIS DE CANARITOS ===")
    logger.info(f"Total canaritos generados: {qcanaritos}")
    logger.info(f"Canaritos en top features:")
    for canarito, rank, importance in canaritos_in_top[:10]:
        logger.info(f"  {canarito}: rank #{rank}, gain={importance:.2f}")

    if canaritos_in_top:
        best_canarito_rank = min([rank for _, rank, _ in canaritos_in_top])
        logger.info(f"Mejor canarito en posición: #{best_canarito_rank}")

        # Obtengo las posiciones de los canaritos y calculo metricas
        ranks = [x[1] for x in canaritos_in_top]
        p50 = np.median(ranks)
        p25 = np.percentile(ranks, 25)
        p75 = np.percentile(ranks, 75)

        logger.info(f"Posiciones de los canaritos en el feature importance: P25: {p25} | Mediana: {p50} | P75: {p75}")
    else:
        logger.info(f"Ningún canarito entre las top features - todas las variables reales son útiles")

    # Guardar TXT con lista ordenada (solo nombres)
    txt_path = os.path.join(os.path.join(BUCKET_NAME, "log"), f"features_ordered_by_gain_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.txt")
    with open(txt_path, 'w') as f:
        f.write(str(df_importances['feature'].tolist()))

    logger.info(f"==== Canaritos Asesinos completado ====")