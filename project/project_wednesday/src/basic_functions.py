
import logging
import polars as pl
from sympy import primerange
import random
from .config import *

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

    X_train = df_train.select(pl.all().exclude(["target", "target_test"])).to_numpy().astype("float32")
    y_train = df_train["target"].to_numpy().astype("float32")

    X_test = df_test.select(pl.all().exclude(["target", "target_test"])).to_numpy().astype("float32")
    y_test = df_test["target_test"].to_numpy().astype("float32")

    return X_train, y_train, X_test, y_test

def undersample(df: pl.DataFrame, sample_fraction: float) -> pl.DataFrame:
    logging.info(f"=== Undersampling al {sample_fraction}")

    # Obtener clientes 0-sampleados
    clientes_sampled = (
        df.filter(pl.col("target") == 0)
          .select("numero_de_cliente")
          .unique()
          .sample(
              fraction=sample_fraction,
              with_replacement=False,
              seed=SEMILLA[1]
          )
    )

    # Filtrar en una única pasada sin copiar todo
    df_out = df.filter(
        (pl.col("target") == 1) |
        (pl.col("numero_de_cliente").is_in(clientes_sampled["numero_de_cliente"]))
    )

    # Mezclar
    df_out = df_out.sample(fraction=1.0, seed=SEMILLA[1])

    return df_out
