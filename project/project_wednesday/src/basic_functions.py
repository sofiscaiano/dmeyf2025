
import logging
import polars as pl
from sympy import primerange
import random
from .config import *
import numpy as np

logger = logging.getLogger(__name__)


def generar_semillas(semilla_primigenia: int, cantidad: int,
                     rango_min: int = 100000, rango_max: int = 1000000) -> list[int]:
    """
    Genera una lista de 'cantidad' números primos seleccionados aleatoriamente
    dentro de un rango determinado, utilizando una semilla para reproducibilidad.

    Parámetros
    ----------
    semilla_primigenia : int
        Valor entero usado para inicializar el generador aleatorio (reproducibilidad).
    cantidad : int
        Número de primos aleatorios que se desean obtener.
    rango_min : int, opcional
        Límite inferior del rango de búsqueda de primos (por defecto 100_000).
    rango_max : int, opcional
        Límite superior del rango de búsqueda de primos (por defecto 1_000_000).

    Retorna
    -------
    list[int]
        Lista de primos seleccionados aleatoriamente.

    """
    if cantidad <= 0:
        raise ValueError("La cantidad de primos a generar debe ser mayor que cero.")
    if rango_min >= rango_max:
        raise ValueError("rango_min debe ser menor que rango_max.")

    # Fijar semilla para reproducibilidad
    random.seed(semilla_primigenia)

    # Generar los primos solo una vez (uso de generador para eficiencia)
    primos = list(primerange(rango_min, rango_max))

    if cantidad > len(primos):
        raise ValueError(f"No hay suficientes primos en el rango: {len(primos)} disponibles.")

    # Seleccionar los primos al azar
    return random.sample(primos, cantidad)

def train_test_split(df: pl.DataFrame, undersampling: bool, mes_train: list, mes_test: list) -> tuple:

    df_train = df.filter(pl.col("foto_mes").is_in(mes_train))
    if undersampling:
        df_train = undersample(df_train, sample_fraction=UNDERSAMPLING_FRACTION)

    df_test = df.filter(pl.col("foto_mes").is_in(mes_test))

    X_train = df_train.select(pl.all().exclude(["target_train", "target_test"])).to_numpy().astype("float32")
    y_train = df_train["target_train"].to_numpy().astype("float32")

    X_test = df_test.select(pl.all().exclude(["target_train", "target_test"])).to_numpy().astype("float32")
    y_test = df_test["target_test"].to_numpy().astype("float32")

    return X_train, y_train, X_test, y_test

def undersample(df: pl.DataFrame, sample_fraction: float) -> pl.DataFrame:
    """
    Realiza un undersampling de la clase mayoritaria (target_train == 0) en Polars.

    Args:
        df (pl.DataFrame): DataFrame de entrada (con columnas 'target_train' y 'numero_de_cliente').
        sample_fraction (float): Fracción de la clase mayoritaria a conservar (0 < frac ≤ 1).
        semilla (int): Semilla aleatoria para reproducibilidad.

    Returns:
        pl.DataFrame: DataFrame resultante submuestreado y mezclado.
    """

    # Separar clases
    df_mayoritaria = df.filter(pl.col("target_train") == 0)
    df_minoritaria = df.filter(pl.col("target_train") == 1)

    # Obtener clientes únicos de la clase mayoritaria
    clientes_unicos = df_mayoritaria.select("numero_de_cliente").unique().sort(pl.col("numero_de_cliente"))
    clientes_unicos = clientes_unicos.rechunk()

    # Muestrear fracción de clientes únicos
    clientes_sampled = clientes_unicos.sample(
        fraction=sample_fraction,
        with_replacement=False,
        seed=SEMILLA[1]
    )

    # Filtrar los registros de esos clientes
    df_mayoritaria_sampled = df_mayoritaria.join(
        clientes_sampled,
        on="numero_de_cliente",
        how="inner"
    )

    # Concatenar ambas clases
    df_undersampled = pl.concat([df_mayoritaria_sampled, df_minoritaria])

    return df_undersampled