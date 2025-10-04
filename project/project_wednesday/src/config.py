import yaml
import os
import logging

logger = logging.getLogger(__name__)

PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

try:
    with open(PATH_CONFIG, 'r') as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral['competencia01']

        PARAMETROS_LGB = _cfgGeneral['parametros_lgb']
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", 'Wednesday')
        DATA_PATH = _cfg.get('DATA_PATH', "../data/competencia.csv")
        SEMILLA = _cfg.get('SEMILLA', [42])
        MES_TRAIN = _cfg.get('MES_TRAIN', ['202102'])
        MES_VALIDACION = _cfg.get('MES_VALIDACION', ['202103'])
        MES_TEST = _cfg.get('MES_TEST', ['202104'])
        GANANCIA_ACIERTO = _cfg.get('GANANCIA_ACIERTO', None)
        COSTO_ESTIMULO = _cfg.get('COSTO_ESTIMULO', None)
        FINAL_TRAIN = _cfg.get('FINAL_TRAIN', [])
        FINAL_PREDICT = _cfg.get('FINAL_PREDICT', "")

except Exception as e:
    logger.error(f'Error al cargar el archivo de configuracion: {e}')
    raise
