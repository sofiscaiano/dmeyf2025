# Proyecto DMEyF 2025 - Script de Procesamiento Principal

Este repositorio contiene el script principal (`main.py`) para el proyecto de la materia en la carpeta `project/project_wednesday/`.

## ⚙️ Requisitos

Asegurate de que el archivo `competencia_02_crudo.csv.gz` se encuentre en `project/project_wednesday/datasets/` antes de ejecutar el script.

---

## 🚀 Uso

El script se ejecuta desde la terminal en el directorio `project/project_wednesday/` de forma local:

```bash
GCP=0 Z=0 python3 main.py --n_trials 50
```

o de forma remota en GCP:

```bash
GCP=1 Z=0 python3 main.py --n_trials 50
```

**Resultado:** Se generará el archivo `predict/lgb_optimization_competencia197_YYYYMMDD_HHMMSS.csv` listo para ser enviado.


