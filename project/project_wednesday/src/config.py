import yaml
import os
import logging

logger = logging.getLogger(__name__)

PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

try:
    FLAG_GCP = int(os.getenv('GCP', 1))
    with open(PATH_CONFIG, 'r') as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral['competencia01']

        PARAMETROS_LGB = _cfgGeneral['parametros_lgb']
        PARAMETROS_LGB_ADHOC = _cfgGeneral['parametros_adhoc']
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", 'Default')
        DATA_PATH = _cfg.get('DATA_PATH', "../data/competencia.csv")
        SEMILLA = _cfg.get('SEMILLA', [42])
        MES_TRAIN = _cfg.get('MES_TRAIN', [202102])
        MES_VALIDACION = _cfg.get('MES_VALIDACION', None)
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

        if FLAG_GCP == 1:
            BUCKET_NAME = os.path.expanduser(_cfgGeneral.get("BUCKET_NAME", '~/buckets/'))
        else:
            BUCKET_NAME = '/Users/sofiascaiano/Documents/maestria/Data Mining EyF/DMEYF/project/project_wednesday/'

except Exception as e:
    logger.error(f'Error al cargar el archivo de configuracion: {e}')
    raise
