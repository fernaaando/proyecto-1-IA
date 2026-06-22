# Proyecto 1: Predicción de la Popularidad Musical en Spotify mediante Random Forest Regressor

# Definición del problema
El presente proyecto toma como contexto el mercado global de la música digital, un entorno sumamente competitivo donde miles de canciones compiten constantemente por captar oyentes. El problema radica en descubrir qué rasgos sonoros son los que definen un éxito comercial. Para resolverlo, se busca entrenar un algoritmo que logre proyectar la popularidad de una pista basándose estrictamente en su perfil acústico, tales como su intensidad (`energy`), índice de bailabilidad (`danceability`), positividad musical (`valence`) y su velocidad (`tempo`).

## Plan de acción
### Descripción del DataSet
Se usará el DataSet "30000 Spotify Songs". Este DataSet tiene casi 30 mil canciones de la API oficial de Spotify y cuenta con un total de 23 variables. En él, cada registro es una canción de la aplicación y tiene información como:
* track_id: id único de cada canción.
* track_name: nombre de la canción.
* track_artist: quien canta la canción.
* energy: ​La energía (de 0.0 a 1.0) indica qué tan intensa y activa se percibe una canción basándose en factores acústicos como el volumen, la velocidad y el timbre.
* danceability: ​La bailabilidad (de 0.0 a 1.0) mide qué tan buena es una canción para el baile.
* acousticness: ​La acústica (de 0.0 a 1.0) indica la probabilidad de que una canción esté hecha con instrumentos acústicos en lugar de electrónicos o sintetizados.

Además de las ya mencionadas, el conjunto de datos incluye otras variables, tales como el track-album_id, track_album_name, popularity entre otras características de audio.

### Modelo(s) seleccionado(s)
Se seleccionó un modelo de Random Forest Regressor para estimar el índice de popularidad de las pistas. Este algoritmo fue seleccionado por su capacidad de predicción de variables continuas y la captación de relaciones no lineales entre las características acústicas de las canciones. Además, permitirá identificar cuáles son los atributos sonoros específicos (como la bailabilidad o la energía) que presentan mayor influencia en el éxito de una pista.

### Estrategia de evaluación
El desempeño del algoritmo se evaluará mediante validación cruzada K-Fold con k=5. Además, la precisión de las predicciones se medirá utilizando las métricas estadísticas R², MAE y RMSE, las cuales indicarán el margen de error entre la popularidad predicha y la real.

## Justificación del modelo
La elección del predictor Random Forest se basa principalmente en su alta viabilidad para predecir en base a variables que no tienen la misma naturaleza o que son independientes entre sí, como las que tiene el dataset a trabajar.
  * Ventajas: Es altamente eficaz en datos no relacionados entre sí y no lineales, sin la necesidad de hacer transformaciones previas a los datos. Además, por cómo funciona el algoritmo, este detectará de forma automática cuáles son las variables más características del dataset para poder predecir.
  * Limitaciones: La principal desventaja es el alto costo computacional al entrenar el algoritmo con aproximadamente 30 mil datos en múltiples árboles de decisión que se generan al mismo tiempo.
  * Pertinencia: La elección de este modelo tiene toda la pertinencia, ya que para nuestro problema a resolver, el cual es tratar de predecir la popularidad de una canción, no se hace con variables lineales, sino que, esto se hace usando solo variables y características de la pista que no tienen relación directa entre ellas, escenario para el cual Random Forest es lo mejor.



## Resultados
La idea es ver si se puede predecir la popularidad de las canciones en Spotify en base a sus características sonoras. Para eso se implementó un modelo basado en Random Forest y una validación cruzada con K-Fold de 5 particiones, con una búsqueda óptima de los mejores hiperparámetros para asegurar que el modelo fuera capaz de predecir de forma correcta datos desconocidos.

### Eleccion del modelo a usar
La decisión de utilizar Random Forest para predecir y no otro modelo se basa principalmente en la naturaleza de los datos del dataset. De partida se descartaron modelos como Regresión Logística o Naive Bayes, ya que estos dos son para separar categorías y en cambio, acá queremos predecir la popularidad exacta en una escala de 0 a 100. Además, las características sonoras de cada canción no funcionan en línea recta; por ejemplo, tener más volumen o más tempo no significa que la canción será más o menos popular. Para características de este tipo, los árboles de decisión son los más aptos, ya que captan estas relaciones no lineales mucho mejor que una regresión lineal sin necesidad de hacer transformaciones matemáticas previas.

### Hiperparámetros
Para llegar al modelo final se realizaron varias pruebas modificando los hiperparámetros y así poder lograr un buen equilibrio y evitar el overfitting y el underfitting. A continuación se detallan algunas pruebas que se hicieron con sus resultados:
En la primera prueba entrenamos al modelo sin restricciones, se entrenó solo utilizando 300 árboles con una profundidad de 20, sin configurar mínimo por hoja ni para partición. Dando los siguientes resultados: 

  * Resultados en conjunto de entrenamiento:
    $R^2$ promedio: 0.84
    MAE promedio: 7.45
    RMSE promedio: 9.12

  * Resultados en conjunto de prueba
    $R^2$ promedio: 0.15
    MAE promedio: 19.50
    RMSE promedio: 23.10

Al analizar los resultados se puede ver que el modelo se aprendió los datos de entrenamiento de memoria. El rendimiento en entrenamiento fue muy alto, llegando a un R^2 de 0.84 y errores muy bajos, pero al evaluar con el conjunto de validación el rendimiento bajó enormemente a un valor de 0.15 y el error se fue muy alto. Esto es claro de un modelo sobreajustado que no sirve para predecir datos nuevos.
Para tratar de amortiguar este sobreajuste se evaluó el modelo de forma contraria, evaluándolo con 1000 árboles y una profundidad de 10. Al finalizar el entreno se obtuvieron los siguientes resultados: 

  * Resultados en conjunto de entrenamiento:
    $R^2$ promedio: 0.33
    MAE promedio: 17.06
    RMSE promedio: 20.52

  * Resultados en conjunto de prueba
    $R^2$ promedio: 0.17
    MAE promedio: 18.99
    RMSE promedio: 22.79

Los resultados acá muestran que el modelo se asfixió y no es capaz de predecir totalmente. Aunque se volvió un poco más estable que el anterior ya que los números de entrenamiento y validación son muy parecidos, perdió la capacidad de predicción ya que tiene un valor de R^2 de 0.17.
Finalmente, después de iterar con distintas combinaciones se llegó a la más óptima. Esta configuración se entrenó ajustando el modelo con 300 árboles, un máximo de profundidad de cada árbol de 15 y estabilizadores de sobreajuste como un mínimo de 10 canciones para particionar una hoja del árbol y que cada hoja tenga 5 canciones como mínimo. Estos hiperparámetros permitieron los siguientes resultados: 

  * Resultados en conjunto de entrenamiento:
    $R^2$ promedio: 0.58
    MAE promedio: 13.24
    RMSE promedio: 16.15

  * Resultados en conjunto de prueba
    $R^2$ promedio: 0.24
    MAE promedio: 18.01
    RMSE promedio: 21.82

Decidimos dejar esta configuración porque así le damos al modelo el suficiente espacio y profundidad para que pueda predecir de forma correcta, pero con las restricciones necesarias para bajar el sobreajuste en los datos, cayendo así de un 0.84 a un 0.58. Conseguimos de esta forma la mejor predicción de datos desconocidos para el modelo, logrando el R^2 más alto posible en el conjunto de prueba, R^2=0.24, y reduciendo el error MAE a su punto más bajo de tan solo 18.01.

### Análisis final
Luego de entrenar el modelo con la configuración más efectiva, se logró controlar el sobreajuste para que el predictor no memorice los datos del conjunto de entrenamiento, obteniendo así un R^2 de 0.58 en los datos de entrenamiento y un R^2 de 0.24 en el conjunto de validación. La razón de que el R^2 en el conjunto de validación sea de solo un 24% indica que si bien el modelo es capaz de predecir la popularidad de una canción en base a sus atributos, no se puede predecir con total certeza solo fijándose en esos datos, ya que el éxito está basado casi en su totalidad en agentes externos como el marketing, la fama del artista o si se hace viral en alguna red social. En cuanto a su estabilidad, las métricas arrojaron un margen de error MAE de 18.01 y un RMSE de 21.82 puntos, lo que confirma que el modelo no comete grandes fallos con frecuencia.
Finalmente, al analizar el gráfico de importancia de características, se puede ver que el modelo al predecir se basa principalmente en sus 3 variables más importantes, que son la duración de la canción, qué tan instrumental es y el volumen de la misma, indicando que la estructura básica y qué tan fuerte suena son la clave para la popularidad de una canción. Luego, atributos como el ritmo o la energía sí aportan a la predicción, pero no son tan importantes como las 3 anteriores. Por último, características como la nota principal o la escala casi no le importan al modelo al momento de predecir, lo que confirma que a la gente no le importa en qué nota esté compuesta la canción al momento de hacerse popular.