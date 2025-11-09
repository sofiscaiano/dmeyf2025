
import logging
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
