# 1) Definición del problema
El presente proyecto toma como contexto el mercado global de la música digital, un entorno sumamente competitivo donde miles de canciones compiten constantemente por captar oyentes. El problema radica en descubrir qué rasgos sonoros son los que definen un éxito comercial. Para resolverlo, se busca entrenar un algoritmo que logre proyectar la popularidad de una pista basándose estrictamente en su perfil acústico, tales como su intensidad (`energy`), índice de bailabilidad (`danceability`), positividad musical (`valence`) y su velocidad (`tempo`).

# 2) Plan de acción
## 2.1) Descripción del DataSet
Se usará el DataSet "30000 Spotify Songs". Este DataSet tiene casi 30 mil canciones de la API oficial de Spotify y cuenta con un total de 23 variables. En él, cada registro es una canción de la aplicación y tiene información como:
* track_id: id único de cada canción.
* track_name: nombre de la canción.
* track_artist: quien canta la canción.
* energy: ​La energía (de 0.0 a 1.0) indica qué tan intensa y activa se percibe una canción basándose en factores acústicos como el volumen, la velocidad y el timbre.
* danceability: ​La bailabilidad (de 0.0 a 1.0) mide qué tan buena es una canción para el baile.
* acousticness: ​La acústica (de 0.0 a 1.0) indica la probabilidad de que una canción esté hecha con instrumentos acústicos en lugar de electrónicos o sintetizados.

Además de las ya mencionadas, el conjunto de datos incluye otras variables, tales como el track-album_id, track_album_name, popularity entre otras características de audio.

## 2.2) Modelo(s) seleccionado(s)
Se seleccionó un modelo de Random Forest Regressor para estimar el índice de popularidad de las pistas. Este algoritmo fue seleccionado por su capacidad de predicción de variables continuas y la captación de relaciones no lineales entre las características acústicas de las canciones. Además, permitirá identificar cuáles son los atributos sonoros específicos (como la bailabilidad o la energía) que presentan mayor influencia en el éxito de una pista.

## 2.3) Estrategia de evaluación
El desempeño del algoritmo se evaluará mediante validación cruzada K-Fold con k=5. Además, la precisión de las predicciones se medirá utilizando las métricas estadísticas R², MAE y RMSE, las cuales indicarán el margen de error entre la popularidad predicha y la real.

# 3) Justificación del modelo
