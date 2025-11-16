import yaml
import os
import logging

logger = logging.getLogger(__name__)

PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

try:
    FLAG_GCP = int(os.getenv('GCP', 1))
    FLAG_ZLIGHTGBM = int(os.getenv('Z', 0))
    with open(PATH_CONFIG, 'r') as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral['competencia02']

        PARAMETROS_LGB = _cfgGeneral['parametros_lgb']
        PARAMETROS_LGB_ADHOC = _cfgGeneral['parametros_adhoc']
        PARAMETROS_ZLGB = _cfgGeneral['parametros_zlgb']
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", 'Default')
        STUDY_HP = _cfg.get("STUDY_HP", None)
        ZEROSHOT = _cfg.get("ZEROSHOT", False)
        DATA_PATH = _cfg.get('DATA_PATH', "../data/competencia.csv")
        SEMILLA = _cfg.get('SEMILLA', [42])
        MES_TRAIN = _cfg.get('MES_TRAIN', [202102])
        MES_TRAIN_BO = _cfg.get('MES_TRAIN_BO', [202102])
        MES_VALIDACION = _cfg.get('MES_VALIDACION_BO', None)
        KSEMILLERIO = _cfg.get('KSEMILLERIO', 5)
        KSEMILLERIO_BO = _cfg.get('KSEMILLERIO_BO', 5)
        MES_TEST = _cfg.get('MES_TEST', [202104])
        GANANCIA_ACIERTO = _cfg.get('GANANCIA_ACIERTO', None)
        COSTO_ESTIMULO = _cfg.get('COSTO_ESTIMULO', None)
        FINAL_TRAIN = _cfg.get('FINAL_TRAIN', [])
        FINAL_PREDICT = _cfg.get('FINAL_PREDICT', "")
        UNDERSAMPLING_FRACTION = _cfg.get('UNDERSAMPLING_FRACTION', 1.0)
        UNDERSAMPLING_FINAL_TRAINING = _cfg.get('UNDERSAMPLING_FINAL_TRAINING', False)
        DROP = _cfg.get('DROP', [])
        ADHOC = _cfg.get('ADHOC', False)
        ENVIOS = _cfg.get('ENVIOS', None)
        DESCRIPCION = _cfg.get('DESCRIPCION', '')
        FLAG_AGUINALDO = _cfg.get('FLAG_AGUINALDO', False)
        FLAG_RANKS = _cfg.get('FLAG_RANKS', False)
        FLAG_TREND_3M = _cfg.get('FLAG_TREND_3M', False)
        FLAG_TREND_6M = _cfg.get('FLAG_TREND_6M', False)
        FLAG_ZEROSD = _cfg.get('FLAG_ZEROSD', False)
        FLAG_EMBEDDING = _cfg.get('FLAG_EMBEDDING', False)
        QLAGS = _cfg.get('QLAGS', 2)

        if FLAG_GCP == 1:
            BUCKET_NAME = os.path.expanduser(_cfgGeneral.get("BUCKET_NAME", '~/buckets/'))
        else:
            BUCKET_NAME = os.path.dirname(os.path.dirname(__file__))#'/Users/sofi/Documents/dmeyf2025/project/project_wednesday/'

except Exception as e:
    logger.error(f'Error al cargar el archivo de configuracion: {e}')
    raise

# =============================================================================
# Configuración de MLflow
# =============================================================================
try:
    # Cargar configuración de MLflow desde config.yaml
    MLFLOW_CFG = _cfgGeneral.get("mlflow", {})

    _bucket_root = BUCKET_NAME #or os.path.dirname(os.path.dirname(__file__))
    _default_artifact_dir = os.path.abspath(
        MLFLOW_CFG.get("ARTIFACT_PATH", os.path.join(_bucket_root, "mlruns"))
    )

    # Configuración básica
    MLFLOW_TRACKING_URI = MLFLOW_CFG.get("TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        MLFLOW_TRACKING_URI = f"file://{_default_artifact_dir}"

    # Configuración directa sin plantillas
    #MLFLOW_EXPERIMENT_NAME = f"DMEyF-{STUDY_NAME}"
    MLFLOW_EXPERIMENT_NAME = "DMEyF-Scaiano"
    MLFLOW_ARTIFACT_PATH = _default_artifact_dir
    MLFLOW_REGISTERED_MODEL_NAME = f"dmeyf-{STUDY_NAME}"

    # Tags fijos
    MLFLOW_TAGS = {
        "proyecto": "DMEyF-Competencia02",
        "equipo": "python",
        "user_name": "sscaiano",
        "comision": "Jueves",
        "version_dataset": "1.0"
    }

    # Crear directorio de artefactos si no existe
    os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

    logger.info(f"Configuración de MLflow cargada - URI: {MLFLOW_TRACKING_URI}")

except Exception as e:
    logger.error(f"Error al cargar configuración de MLflow: {e}")
    raise