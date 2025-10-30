import sympy
from .config import *
import random

def best_model(q_seeds, model1, model2):

    prime_list = list(sympy.primerange(100000, 1000000))
    random.seed(SEMILLA[0])
    seeds = random.sample(prime_list, q_seeds)

    pvalue = 1
    isem = 1
    vgan1 = []
    vgan2 = []

    while isem <= q_seeds and pvalue > 0.05:



