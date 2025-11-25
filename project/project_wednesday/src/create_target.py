import duckdb
import pandas as pd
import logging
import os
from .config import BUCKET_NAME
import polars as pl

from .loader import cargar_datos_csv

logger = logging.getLogger(__name__)


def create_target(df: pl.DataFrame, export=True) -> pl.DataFrame:

    sql = """
    with cte as (
    select
        * REPLACE(last_day(make_date(
        CAST(SUBSTR(CAST(foto_mes AS VARCHAR), 1, 4) AS INTEGER),
        CAST(SUBSTR(CAST(foto_mes AS VARCHAR), 5, 2) AS INTEGER),
        1
      )) as foto_mes)
    from df)
    
    select t0.* REPLACE((YEAR(t0.foto_mes) * 100 + MONTH(t0.foto_mes))::INT as foto_mes),
           case when t0.foto_mes = '2021-09-30' then NULL -- todavia no tengo datos para conocer las bajas 
                when t2.foto_mes is null and t0.foto_mes = '2021-08-31' then NULL -- no tengo todavia el dato para agosto
                when t1.foto_mes is null then 'BAJA+1' 
                when t2.foto_mes is null then 'BAJA+2'
                else 'CONTINUA' end as target
    from cte as t0
    left join cte as t1 -- mes siguiente
    on t0.numero_de_cliente = t1.numero_de_cliente
    and last_day(date_add(t0.foto_mes, INTERVAL 1 MONTH)) = t1.foto_mes
    left join cte as t2 -- dos meses despues
    on t0.numero_de_cliente = t2.numero_de_cliente
    and last_day(date_add(t0.foto_mes, INTERVAL 2 MONTH)) = t2.foto_mes
    ORDER BY t0.numero_de_cliente, t0.foto_mes
    """

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")

    df = con.execute(sql).pl()

    # df = df.with_columns([
    #     pl.col("tmobile_app").cast(pl.Int64),
    #     pl.col("cmobile_app_trx").cast(pl.Int64)
    # ])

    if export:
        export_path = os.path.join(BUCKET_NAME, "datasets/competencia_03.parquet")
        df.write_parquet(export_path, compression="gzip")
        logger.info(">>> Creacion de target finalizada -> {export_path}")

    con.close()

    return df