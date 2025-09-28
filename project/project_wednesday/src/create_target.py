import duckdb
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_target():
    df = pd.read_csv('data/competencia_01_crudo.csv')

    sql = """
    with cte as (
    select
        * REPLACE(last_day(make_date(
        CAST(SUBSTR(CAST(foto_mes AS VARCHAR), 1, 4) AS INTEGER),
        CAST(SUBSTR(CAST(foto_mes AS VARCHAR), 5, 2) AS INTEGER),
        1
      )) as foto_mes)
    from df)
    
    select t0.*,
           --t0.numero_de_cliente, 
           --t0.foto_mes, 
           case when t0.foto_mes = '2021-06-30' then NULL -- todavia no tengo datos para conocer las bajas 
                when t2.foto_mes is null and t0.foto_mes = '2021-05-31' then NULL -- no tengo todavia el dato para mayo
                when t1.foto_mes is null then 'BAJA+1' 
                when t2.foto_mes is null then 'BAJA+2'
                else 'CONTINUA' end as target
    from cte as t0
    left join cte as t1
    on t0.numero_de_cliente = t1.numero_de_cliente
    and last_day(date_add(t0.foto_mes, INTERVAL 1 MONTH)) = t1.foto_mes
    left join cte as t2
    on t0.numero_de_cliente = t2.numero_de_cliente
    and last_day(date_add(t0.foto_mes, INTERVAL 2 MONTH)) = t2.foto_mes
    --where t0.foto_mes <= '2021-04-30'
    """

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()

    print(df.shape)
    print(df['target'].value_counts(dropna=False))
    df.to_csv("data/competencia_01.csv", index=False)

    con.close()