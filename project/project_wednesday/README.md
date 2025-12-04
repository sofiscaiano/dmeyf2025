# 🚀 Entrega Competencia 3

> Proyecto de prediccion de Churn para materia Data Mining en Economia y Finanzas 2025

---

## 🧐 Resumen <a name="acerca-del-proyecto"></a>

Se entrenaron tres modelos con distintas semillas primigenias y 50 semillerio en cada uno. Luego fueron ensambladas sus predicciones para obtener una prediccion final.

 ## 💻 Guia de reproducibilidad <a name="reproducibilidad"></a>

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/sofiscaiano/dmeyf2025.git](https://github.com/sofiscaiano/dmeyf2025.git)
    cd dmeyf2025/project/project_wednesday
    ```
    
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    
    cd
    rm -rf  LightGBM
    git clone --recursive  https://github.com/dmecoyfin/LightGBM
    
    source  ~/.venv/bin/activate
    pip install sympy
    pip uninstall --yes lightgbm
    
    # instalacion Python
    cd  ~/LightGBM
    sh ./build-python.sh  install
    
    cd ~/dmeyf2025/project/project_wednesday
    
    ```
    
4.  **Entrenar modelo 1**
    ```bash
    GCP=1 Z=1 CONFIG='config1.yaml' python3 main.py
    ```
    
5.  **Entrenar modelo 1**
    ```bash
    GCP=1 Z=1 CONFIG='config2.yaml' python3 main.py
    ```
    
6.  **Entrenar modelo 3**
    ```bash
    GCP=1 Z=1 CONFIG='config3.yaml' python3 main.py
    ```
    
7.  **Ensamblar los tres modelos**
      ```bash
      GCP=1 Z=1 CONFIG='config_ensemble.yaml' python3 final_ensemble.py
      ```
          