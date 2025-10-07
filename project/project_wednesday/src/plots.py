import pandas as pd
from matplotlib import pyplot as plt
from .config import *

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def plot_mean_importance(all_importances, importance_type, type):

    # 3. Combinar y promediar las importancias
    df_importances = pd.concat(all_importances)
    df_mean_importance = df_importances.groupby('feature')['importance'].mean().reset_index()

    # Filtrar aquellas con importancia cero (opcional, pero buena práctica)
    df_filtered_importance = df_mean_importance[df_mean_importance['importance'] > 0]

    # Ordenar por importancia de forma descendente
    df_sorted_importance = df_filtered_importance.sort_values(by='importance', ascending=False)

    df_top_25 = df_sorted_importance.head(25)

    # Ajustamos la altura de la figura para que sea legible (25 * 0.4 es un buen punto de partida)
    figsize_height = max(8, len(df_top_25) * 0.4)

    plt.figure(figsize=(10, figsize_height))
    plt.barh(df_top_25['feature'], df_top_25['importance'])
    plt.xlabel(f"Importancia Promediada ({importance_type})")
    plt.title(f"Top {len(df_top_25)} Variables Más Importantes (Ensamble LightGBM)")
    plt.gca().invert_yaxis()  # Invertir el eje Y para que la más importante quede arriba

    # Guardar la figura
    plt.savefig(f'resultados/{STUDY_NAME}_importancia_ensamble_{type}.png', bbox_inches='tight', dpi=300)