 
 
Pagina 1  
PROYECTO FINAL DE CURSO  
Sistema de Prediccion e Inteligencia Deportiva - FI FA World Cup 2026  
Curso: Deep Learning aplicado a la Ingenieria  
Docente: Ing. Jairo Pinedo Taquia, M.Sc. 
Modalidad: Grupal - 3 a 4 integrantes 
Puntaje total: 20 puntos 
Entrega: Semana 15 
 
Descripcion general 
La FIFA World Cup 2026 se disputara en Estados Unid os, Canada y Mexico con 48 selecciones 
organizadas en 12 grupos. Un consorcio de medios de portivos contrata al grupo para construir 
un sistema de inteligencia artificial interactivo q ue permita a analistas y periodistas deportivos 
simular escenarios del torneo, predecir resultados de partidos y explorar como cambia el 
campeon probable cuando se modifican los grupos. 
El sistema debe combinar tecnicas de Deep Learning vistas a lo largo del curso con una 
interfaz funcional que cualquier usuario no tecnico  pueda operar. 
Datasets obligatorios 
Todos son publicos y gratuitos.  
Dataset  Plataforma  Enlace  Uso en el 
proyecto  
International 
football results 
1872 -2024  Kaggle  kaggle.com/datasets/martj42/international-
football-results-from-1872-to-2017  Historial de 
partidos y goles  
FIFA World 
Rankings  Kaggle  kaggle.com/datasets/cashncarry/fifaworldranking  Ranking FIFA 
por seleccion  
FIFA Players 
15-24  Kaggle  kaggle.com/datasets/stefanoleone992/fifa-21-
complete-player-dataset  Atributos de 
jugadores por 
seleccion  
World Cup 
historical stats  Kaggle  kaggle.com/datasets/piterfm/fifa-football-world-
cup  Estadisticas por 
edicion de 
mundial  
 
 
 

---

 
 
Pagina 2  
Componentes del sistema 
El proyecto se divide en dos componentes principale s que deben estar integrados en un unico 
sistema interactivo. 
Componente 1 - Modelo predictivo de partidos 
Construir un modelo de Deep Learning que prediga el  resultado de un partido entre dos 
selecciones cualesquiera. El modelo debe recibir co mo entrada: 
- Ranking FIFA actual de ambas selecciones 
- Promedio de goles anotados y recibidos en los ult imos 10 partidos de cada seleccion 
- Historial de enfrentamientos directos (victorias,  empates, derrotas y diferencia de goles) 
- Atributos agregados de los jugadores FIFA de cada  seleccion: promedio de overall, 
pace, shooting, defending y physical de los 23 conv ocados 
Y producir como salida: 
- Probabilidad de victoria del equipo A 
- Probabilidad de empate 
- Probabilidad de victoria del equipo B 
- Marcador mas probable (goles A - goles B) 
Requisitos tecnicos del modelo: el grupo debe imple mentar y comparar obligatoriamente 
dos arquitecturas.  
Arquitectura 1 - Red neuronal densa (MLP) 
Cubre Semanas 1 a 4 del curso. Debe incluir: 
- Al menos 4 capas ocultas con funciones de activac ion ReLU 
- Regularizacion L2 en al menos dos capas 
- Dropout entre capas densas 
- Entrenamiento con Adam y comparacion contra SGD c on momentum 
- Graficos de curvas de loss y accuracy para train y validation 
Arquitectura 2 - Red recurrente sobre trayectoria h istorica (LSTM) 
Cubre Semanas 9 y 10 del curso. La idea es que la s ecuencia de los ultimos k partidos de cada 
seleccion se procesa como una serie temporal. Debe incluir: 
- Construccion de ventanas temporales con los ultim os 10 partidos como lookback 
- Arquitectura LSTM de dos capas con Dropout 
- Comparacion con GRU en terminos de precision y ve locidad 
- Documentacion del problema de vanishing gradient con la ecuacion de BPTT y 
evidencia empirica en el notebook 
El grupo elige cual de los dos modelos usa en la si mulacion del torneo y justifica la decision con 
metricas concretas (F1-score sobre partidos del tes t set, que corresponde a la fase de grupos y 
eliminatorias del World Cup 2022 reservada como con junto de prueba). 
 
 

---

 
 
Pagina 3  
Componente 2 - Simulador del torneo 
Construir un simulador completo del Mundial 2026 qu e use el modelo entrenado para calcular 
probabilidades en cada fase del torneo, desde la fa se de grupos hasta la final. 
Funcionalidades obligatorias:  
Vista de fase de grupos  
- Mostrar los 12 grupos del Mundial 2026 con sus 4 selecciones cada uno como 
configuracion por defecto 
- Permitir al usuario reasignar selecciones entre g rupos arrastrando o usando selectores 
- Al modificar un grupo, recalcular automaticamente  las probabilidades de clasificacion de 
las 4 selecciones y actualizar la tabla en tiempo r eal 
- Mostrar la tabla de cada grupo con puntos proyect ados, goles esperados y porcentaje 
de clasificacion a octavos 
Vista de cuadro de eliminacion directa  
- Simular automaticamente octavos, cuartos, semifin ales y final usando las probabilidades 
del modelo 
- Mostrar el bracket completo del torneo con el equ ipo favorito en cada cruce 
- Permitir al usuario forzar un resultado especific o (ejemplo: hacer que Peru gane un 
partido) y ver como cambia el cuadro desde ese punt o 
- Mostrar en cada partido las tres probabilidades: victoria A, empate, victoria B 
Vista de campeones probables  
- Grafico de barras horizontal con las 10 seleccion es con mayor probabilidad de ganar el 
torneo 
- Las probabilidades se recalculan en tiempo real c uando el usuario modifica los grupos 
- Historico: mostrar cuantas veces cada seleccion h a ganado el Mundial como contexto 
Entregables 
Entregable  Descripcion  Peso  
Notebook Colab  Ejecutado de inicio a fin, con 
comentarios en cada linea de 
codigo, teoria en celdas 
Markdown y graficos de 
evaluacion de los modelos  10 pts  
Dashboard interactivo  Funcional, en espanol, con los 
dos componentes integrados. 
Puede ser Streamlit, Dash o un 
artifact HTML/React. Si es 
Streamlit, incluir requirements.txt 
y video de 2 minutos mostrando 
el dashboard en funcionamiento  8 pts  
Defensa oral  10 minutos por grupo. Cualquier 
integrante puede ser preguntado 
sobre cualquier parte del 
sistema  2 pts  

---

 
 
Pagina 4  
 
Rubrica detallada 
Criterio  Descripcion  Pts  
Preprocesamiento y features  Datos limpios, features de 
ranking, historial y atributos 
FIFA correctamente construidos 
y normalizados  1.5  
Modelo MLP  Arquitectura correcta, 
regularizacion, comparacion de 
optimizadores, curvas 
documentadas  2.0  
Modelo LSTM/GRU  Ventanas temporales sin data 
leakage, comparacion LSTM vs 
GRU, evidencia de vanishing 
gradient  2.5  
Evaluacion y seleccion  Justificacion con metricas del 
modelo elegido para la 
simulacion, test set con datos 
del Mundial 2022  1.0  
Simulador de grupos  Grupos modificables, tabla de 
clasificacion con probabilidades 
en tiempo real  1.5  
Simulador de eliminacion  Bracket completo, resultados 
forzables, probabilidades por 
partido  1.5  
Campeones probables  Top 10 actualizado 
dinamicamente con contexto 
historico  1.0  
Calidad del notebook  Comentarios en cada linea, 
teoria correcta en Markdown, 
graficos etiquetados  1.5  
Calidad del dashboard  Interfaz en espanol, sin jerga 
tecnica, usable por un no 
tecnico  1.5  
Defensa oral  Dominio del sistema, capacidad 
de explicar decisiones de diseno  2.0  
Total   16.5  
El puntaje se normaliza a 20. Un grupo que entrega todo correcto obtiene 20.  
 
 
 
 
 

---

 
 
Pagina 5  
Distribucion de roles sugerida 
Rol  Integrante  Responsabilidad principal  
Ingeniero de datos  1 EDA, limpieza, construccion de 
features, dataset de 
entrenamiento  
Ingeniero ML - redes densas  2 Modelo MLP, comparacion de 
optimizadores, regularizacion  
Ingeniero ML - secuencial  3 Modelo LSTM/GRU, ventanas 
temporales, evaluacion final  
Ingeniero de producto  4 Dashboard completo, 
integracion de modelos, UX  
En grupos de 3, el rol 4 se distribuye entre los tr es integrantes. El notebook debe tener una 
celda inicial indicando que secciones elaboro cada integrante. 
 
 
Nota importante sobre el uso de IA generativa 
El grupo puede usar herramientas de IA (ChatGPT, Cl aude, Copilot) para apoyo en la escritura 
de codigo. Sin embargo, cada linea de codigo del no tebook debe tener un comentario que 
explique que hace y por que. Durante la defensa ora l se verificara que el grupo comprende el 
codigo entregado. Entregar codigo no comprendido eq uivale a nota cero en la defensa. 