# Borrador — Marco de referencia (OVCD)

Documento de trabajo para redactar el marco teórico y el estado del arte del plan.
Aquí están, uno por uno, los conceptos que hay que definir, un borrador de cada
definición con sus citas, y al final la tabla de fuentes con la idea concreta que se
extrae de cada una. Las claves de cita entre `[[ ]]` corresponden a las entradas de
`referencias.bib`; en LaTeX se escriben como `\cite{clave}`.

> Estilo: sin paréntesis en la prosa. Para siglas y aclaraciones se usan comas o
> guiones largos.

---

## 1. Conceptos que hay que definir

Estos son todos los conceptos que el lector necesita para entender el trabajo. Van
ordenados de lo general a lo particular, que es como debe leerse el marco teórico.

1. **Imagen hiperespectral y cubo de datos** — qué es, la firma espectral, cómo se adquiere, en qué se diferencia de una imagen RGB o multiespectral.
2. **Detección de cambios bitemporal** — comparar dos imágenes de la misma escena en momentos distintos para localizar lo que cambió.
3. **Segmentación de imágenes** — delimitar regiones a nivel de píxel.
4. **Transformers y Vision Transformers** — la arquitectura base de los modelos usados.
5. **Modelos fundacionales** — modelos grandes preentrenados y reutilizables.
6. **Aprendizaje zero-shot** — resolver una tarea sin ejemplos de entrenamiento de esa tarea.
7. **Vocabulario abierto** — etiquetar sin un conjunto cerrado de clases predefinidas.
8. **Modelos de visión-lenguaje y descripción de imágenes** — alinear imagen y texto, y generar descripciones en lenguaje natural.
9. **Representaciones vectoriales y similitud coseno** — los *embeddings* y cómo se comparan.
10. **Métricas de evaluación** — precisión, exhaustividad, puntaje F1 e intersección sobre unión.
11. **Modelos y datos del sistema** — HyperSIGMA, Segment Anything, CLIP, BLIP y el conjunto HyperCap.

---

## 2. Definiciones con citas (borrador)

### 2.1 Imagen hiperespectral y cubo de datos
Una imagen hiperespectral es una representación de una escena que registra, para cada
píxel, la intensidad de la luz en decenas o cientos de bandas espectrales contiguas y
estrechas, en lugar de las tres bandas de una imagen a color convencional. El resultado
es una estructura tridimensional conocida como cubo hiperespectral, con dos dimensiones
espaciales y una dimensión espectral. La secuencia de valores de un mismo píxel a lo
largo de todas las bandas se denomina firma espectral y actúa como una huella que
permite identificar materiales y coberturas con un detalle que el color no alcanza
[[bhargava2024hyperspectral]]. La adquisición suele realizarse por barrido, ya sea
línea a línea con sensores de empuje o punto a punto, y existe siempre un compromiso
entre la resolución espacial, la resolución espectral y la relación señal a ruido
[[bhargava2024hyperspectral]]. Frente a una imagen RGB, que solo captura tres bandas
amplias, la imagen hiperespectral aporta una riqueza espectral que habilita tareas de
clasificación de materiales, agricultura de precisión, monitoreo ambiental y
teledetección [[bhargava2024hyperspectral]].

### 2.2 Detección de cambios bitemporal
La detección de cambios consiste en comparar dos o más imágenes de la misma escena
geográfica capturadas en momentos distintos para identificar las regiones donde se
produjo una modificación. En teledetección es una tarea central para el monitoreo del
territorio, la evaluación de desastres y el seguimiento del uso del suelo. Las técnicas
clásicas operaban sobre diferencias de píxeles o índices espectrales, mientras que los
enfoques actuales basados en aprendizaje profundo aprenden representaciones que capturan
información semántica y son más robustos frente a variaciones de iluminación y registro
[[khelifi2020deep]]. En este trabajo la detección es bitemporal, es decir, se comparan
exactamente dos capturas, una en el instante inicial y otra en el instante final.

### 2.3 Segmentación de imágenes
La segmentación es la tarea de asignar a cada píxel de una imagen una etiqueta que lo
agrupa en una región coherente, delimitando objetos o zonas con precisión a nivel de
píxel. A diferencia de la clasificación, que asigna una única categoría a toda la imagen,
la segmentación produce máscaras que separan cada elemento de la escena.

En cuanto a los modelos que existen, el enfoque tradicional se apoyó en redes
convolucionales de tipo codificador y decodificador, como U-Net, Mask R-CNN o DeepLab,
que se entrenan de forma supervisada para un conjunto fijo de clases y un dominio
concreto. El paradigma actual son los modelos fundacionales de segmentación promptable,
cuyo referente es Segment Anything, entrenado sobre un conjunto de datos masivo y capaz de
generar máscaras precisas a partir de indicaciones espaciales como puntos o recuadros, sin
necesidad de reentrenarse para cada dominio [[kirillov2023sam]]. Modelos posteriores como
SAM 2 extienden esta idea a video, pero comparten el mismo principio de segmentación
guiada por indicaciones.

La aplicación de estos modelos a una imagen hiperespectral no es directa. Segment Anything
está diseñado para imágenes naturales de tres canales y no puede recibir como entrada un
cubo hiperespectral de cientos de bandas. En nuestro sistema se salva esta distancia
proyectando el cubo alineado a una representación de tres canales, ya sea seleccionando las
bandas más cercanas a las longitudes de onda del rojo, el verde y el azul, en torno a 660,
550 y 470 nanómetros, o tomando las tres primeras componentes principales del cubo. Esa
imagen de tres canales, que corresponde al estado final de la escena, es la que recibe
Segment Anything. La segmentación no se ejecuta sobre toda la imagen de forma ciega, sino
que se guía con indicaciones espaciales derivadas del mapa de cambio, y cada máscara
resultante se interseca con la región de cambio para confinarla a lo que efectivamente se
modificó [[kirillov2023sam]]. De este modo la precisión a nivel de píxel de un modelo
entrenado en imágenes RGB se traslada al dominio hiperespectral sin reentrenarlo.

### 2.4 Transformers y Vision Transformers
El transformador es una arquitectura de red neuronal basada por completo en el mecanismo
de atención, que en lugar de procesar la información de forma local, como hace una
convolución, pondera de manera explícita la relación entre todos los elementos de una
secuencia y aprende cuáles son relevantes entre sí [[vaswani2017attention]]. El Vision
Transformer traslada esta idea a las imágenes: divide la imagen en parches de tamaño fijo,
proyecta cada parche a un vector, los trata como una secuencia de fichas y aplica atención
sobre ellos. Su ventaja frente a las redes convolucionales se manifiesta al preentrenarse
con grandes volúmenes de datos, régimen en el que aprende representaciones muy
transferibles, y por eso se ha convertido en la arquitectura base de los modelos
fundacionales de visión [[dosovitskiy2021vit]].

### 2.5 Modelos fundacionales
Un modelo fundacional es un modelo de gran escala preentrenado sobre cantidades masivas
de datos, diseñado para servir como base reutilizable en múltiples tareas posteriores sin
necesidad de entrenarlo desde cero. En el dominio hiperespectral, HyperSIGMA es el primer
modelo fundacional con más de mil millones de parámetros, preentrenado sobre un conjunto
extenso de imágenes hiperespectrales, capaz de extraer características espaciales y
espectrales transferibles a tareas como clasificación, detección de cambios y
segmentación [[wang2024hypersigma]].

En los últimos años, los modelos fundacionales se han convertido en una herramienta central
de la investigación en visión por computadora y teledetección. Su capacidad de transferir el
conocimiento adquirido en el preentrenamiento a tareas nuevas, con pocos o ningún ejemplo
etiquetado, ha impulsado su adopción en problemas donde los datos anotados son escasos o
costosos, como ocurre con las imágenes hiperespectrales. Esta tendencia se refleja en la
aparición de modelos fundacionales propios del dominio [[wang2024hypersigma]], en enfoques de
clasificación sin entrenamiento específico [[pang2025special]] y en la detección de cambios
semánticos apoyada en estos modelos [[zhu2025semanticcd]].

### 2.6 Aprendizaje zero-shot
El aprendizaje zero-shot se refiere a la capacidad de un modelo para resolver una tarea o
reconocer categorías que no aparecieron durante su entrenamiento, apoyándose en el
conocimiento general adquirido en el preentrenamiento. Modelos como CLIP demuestran este
comportamiento al clasificar imágenes en categorías descritas únicamente mediante texto,
sin haber visto ejemplos etiquetados de esas categorías [[radford2021clip]]. En el ámbito
hiperespectral, trabajos recientes llevan esta idea a la clasificación de coberturas sin
entrenamiento específico [[pang2025special]]. El sistema propuesto es zero-shot en tanto
no requiere entrenar sobre pares hiperespectrales etiquetados para detectar y nombrar los
cambios.

### 2.7 Vocabulario abierto
El enfoque de vocabulario abierto permite etiquetar regiones de una imagen con cualquier
concepto expresable en lenguaje natural, en lugar de restringirse a un conjunto cerrado y
predefinido de clases.

Para entender su ventaja conviene contrastarlo con el enfoque tradicional. Un clasificador
clásico se entrena sobre un número fijo de categorías, por ejemplo agua, vegetación o
edificación, y solo puede responder con una de esas etiquetas aprendidas. Si aparece un
concepto nuevo que no estaba en ese conjunto, el modelo no puede nombrarlo, y para
incorporarlo hay que anotar datos nuevos y volver a entrenar la red por completo. El
vocabulario queda congelado en el momento del entrenamiento.

El vocabulario abierto cambia este funcionamiento. En lugar de una capa final con un número
fijo de salidas, se emplean modelos que proyectan imágenes y texto a un mismo espacio de
representación, de modo que la etiqueta se decide comparando la región con la descripción
textual de cada categoría candidata [[radford2021clip]]. Las categorías ya no están fijadas
en la red, sino que se entregan como texto en el momento de la consulta, y pueden cambiarse o
ampliarse sin reentrenar nada [[kombol2025trainingfree]].

La diferencia se entiende bien con una analogía. El enfoque tradicional es como un examen de
opción múltiple, donde las respuestas posibles están impresas de antemano y solo se puede
marcar una de ellas. El vocabulario abierto, en cambio, funciona como un prompt: uno mismo
escribe en lenguaje libre las opciones que quiere buscar y el modelo decide cuál encaja mejor
con la imagen. Igual que a un modelo de lenguaje se le escribe cualquier pregunta en vez de
elegir de un menú de botones, aquí se le entrega al sistema cualquier lista de etiquetas en
texto. En nuestro caso esto es literal: el vocabulario es una lista de indicaciones textuales
que recibe CLIP, y ampliarla o cambiarla es tan sencillo como editar esa lista, sin tocar los
pesos del modelo.

En teledetección, esta idea se ha comenzado a aplicar a la detección de cambios semánticos,
superando la limitación de las clases fijas [[zhu2025semanticcd]].

### 2.8 Modelos de visión-lenguaje y descripción de imágenes
Los modelos de visión-lenguaje aprenden una representación conjunta de imágenes y texto a
partir de grandes colecciones de pares imagen y descripción, con el fin de resolver tareas
visuales guiadas por lenguaje sin entrenar un modelo distinto para cada una
[[zhang2024vlm]]. CLIP es el exponente que se emplea en este trabajo: entrena dos
codificadores, uno visual y uno textual, para proyectar ambas modalidades a un espacio común
donde la correspondencia se mide por similitud, lo que habilita la clasificación por texto
libre y el etiquetado de vocabulario abierto [[radford2021clip]].

La descripción de imágenes, o generación de subtítulos, va un paso más allá de la
clasificación y produce frases en lenguaje natural que resumen el contenido visual. Los
primeros modelos neuronales de esta tarea combinaban un codificador convolucional con un
decodificador recurrente que generaba la frase palabra por palabra; en palabras de sus
autores, un modelo que "can be used to generate natural sentences describing an image"
[[vinyals2015show]]. La línea actual la dominan los modelos de visión-lenguaje preentrenados
a gran escala, como BLIP, que unifican en una sola arquitectura la comprensión y la
generación de lenguaje a partir de imágenes [[li2022blip]].

En teledetección, la descripción de imágenes tiene una trayectoria propia. La construcción
de conjuntos de datos específicos, como RSICD, con cerca de once mil imágenes de teledetección
y varias descripciones por imagen, permitió entrenar y evaluar modelos de descripción
adaptados a las escenas aéreas y satelitales [[lu2017rsicd]]. Sin embargo, ese avance se
concentró en imágenes ópticas de tres canales. El traslado al dominio hiperespectral es mucho
más reciente: HyperCap es el primer conjunto de datos y punto de referencia orientado a la
descripción de coberturas terrestres a partir de imágenes hiperespectrales, y abre la puerta
a aplicar modelos de visión-lenguaje sobre el cubo espectral completo [[das2025hypercap]].

En nuestro sistema, estos modelos ocupan las dos últimas etapas semánticas del pipeline. CLIP
asigna a cada región de cambio una etiqueta de vocabulario abierto por similitud entre imagen
y texto [[radford2021clip]], y un modelo de descripción de tipo BLIP, adaptado sobre HyperCap,
genera una frase en lenguaje natural que caracteriza la región con más riqueza que una sola
etiqueta [[li2022blip]] [[das2025hypercap]].

### 2.9 Representaciones vectoriales y similitud coseno
Una idea central del aprendizaje profundo es que una buena representación de los datos, una
que capture los factores de variación relevantes, es lo que hace posible resolver tareas
posteriores con facilidad [[bengio2013representation]]. Bajo este principio, los modelos
representan cada imagen o región mediante un vector de características, conocido como
*embedding*, que condensa su contenido semántico en un espacio de alta dimensión, de modo
que dos entradas parecidas quedan cerca en ese espacio y dos distintas quedan lejos.

La comparación entre dos representaciones se realiza con la similitud coseno, que mide el
coseno del ángulo entre ambos vectores y arroja un valor cercano a uno cuando apuntan en la
misma dirección y cercano a cero cuando son ortogonales. Se prefiere sobre la distancia
euclidiana porque atiende a la orientación del vector, asociada al contenido semántico, y no
a su magnitud, lo que la hace robusta frente a diferencias de escala o iluminación.

En este trabajo, la similitud coseno es el mecanismo que convierte las representaciones en un
mapa de cambio. HyperSIGMA extrae un vector por posición espacial en la imagen inicial y en
la final, y para cada posición se calcula la similitud coseno entre ambos vectores: donde la
similitud es baja, el contenido cambió, y donde es alta, se mantuvo [[wang2024hypersigma]].
Esa misma operación reaparece en el etiquetado, cuando se compara la representación de una
región con la de cada descripción textual candidata.

### 2.10 Métricas de evaluación
La evaluación de la detección de cambios se plantea como una comparación píxel a píxel entre
el resultado del sistema y una verdad de terreno, de la que surgen cuatro casos: aciertos
positivos cuando se detecta un cambio real, aciertos negativos cuando se identifica
correctamente una zona sin cambio, falsos positivos cuando se señala un cambio inexistente y
falsos negativos cuando se pasa por alto un cambio real. Sobre estos conteos se definen las
métricas de desempeño.

La precisión mide qué proporción de los cambios detectados son correctos y penaliza los
falsos positivos, mientras que la exhaustividad mide qué proporción de los cambios reales se
detectaron y penaliza los falsos negativos. Como suele existir una tensión entre ambas, el
puntaje F1 las combina en un único valor mediante su media armónica, que solo es alta cuando
las dos lo son. La intersección sobre unión complementa lo anterior al evaluar el solapamiento
espacial entre la región detectada y la real, dividiendo el área común entre el área total que
cubren juntas. Estas métricas son el estándar en la revisión de métodos de detección de
cambios basados en aprendizaje profundo [[khelifi2020deep]].

En este trabajo, estas métricas son la base de la validación cuantitativa del sistema. La
máscara de cambio producida por el pipeline se contrasta con la verdad de terreno de un
conjunto hiperespectral bitemporal de referencia, y los valores de precisión, exhaustividad y
puntaje F1 permiten situar el desempeño frente a métodos existentes.

### 2.11 Modelos y datos del sistema
Las tres piezas centrales del pipeline comparten el mismo esqueleto arquitectónico, el
Vision Transformer. En la extracción de características, HyperSIGMA procesa cada imagen con
un codificador de tipo Vision Transformer espacial, en su variante base, que divide el cubo
en parches y produce mapas de características densos a partir de los cuales se calcula el
mapa de cambio [[wang2024hypersigma]]. En la segmentación, Segment Anything delimita las
regiones con un codificador de imagen Vision Transformer de gran tamaño [[kirillov2023sam]].
Y en el etiquetado, CLIP asigna las etiquetas de vocabulario abierto mediante un codificador
visual Vision Transformer alineado con el texto [[radford2021clip]]. Así, el mismo principio
de atención sobre parches recorre la detección de características, la segmentación y el
etiquetado, lo que da coherencia arquitectónica a todo el sistema.

A continuación se reúnen, ya como tecnología concreta, las piezas del sistema:

- **HyperSIGMA** — modelo fundacional hiperespectral basado en Vision Transformer, escalable a más de mil millones de parámetros y preentrenado sobre un conjunto masivo de imágenes hiperespectrales. Extrae las representaciones profundas de cada imagen, sobre las que se calcula el mapa de cambio [[wang2024hypersigma]].
- **Segment Anything** — modelo fundacional de segmentación promptable con codificador Vision Transformer de gran tamaño. Convierte las regiones de cambio en máscaras precisas a nivel de píxel a partir de indicaciones espaciales [[kirillov2023sam]].
- **CLIP** — modelo de visión-lenguaje que alinea imagen y texto en un espacio común. Asigna a cada región una etiqueta de vocabulario abierto por similitud entre la imagen y las descripciones textuales candidatas [[radford2021clip]].
- **BLIP** — modelo de visión-lenguaje que unifica comprensión y generación. Genera la descripción en lenguaje natural de cada región, adaptado sobre datos hiperespectrales [[li2022blip]].
- **HyperCap** — primer conjunto de datos y punto de referencia para la descripción de imágenes hiperespectrales, con anotaciones textuales a nivel de píxel. Sirve de base de entrenamiento y evaluación del módulo de descripción [[das2025hypercap]].

---

## 3. Estado del arte (borrador de ejes)

El estado del arte no es una lista, es un argumento que termina en el vacío que llena este
trabajo. Se propone organizarlo en tres ejes y un cierre:

1. **Detección de cambios en teledetección** — de las técnicas clásicas basadas en píxeles a los métodos de aprendizaje profundo [[khelifi2020deep]], con la frontera actual en la detección semántica y de vocabulario abierto [[zhu2025semanticcd]].
2. **Segmentación y reconocimiento de vocabulario abierto** — el auge de los modelos fundacionales de segmentación [[kirillov2023sam]] y de visión-lenguaje [[radford2021clip]], y su combinación para segmentar sin clases fijas [[kombol2025trainingfree]].
3. **Modelos fundacionales y zero-shot en imágenes hiperespectrales** — la llegada de un modelo fundacional propio del dominio [[wang2024hypersigma]] y los primeros enfoques zero-shot [[pang2025special]] y de descripción [[das2025hypercap]].
4. **Vacío de investigación** — el vocabulario abierto y el zero-shot están resueltos en imágenes RGB y empiezan en teledetección, pero nadie los ha llevado a la detección de cambios sobre imágenes hiperespectrales bitemporales. Ese es el aporte de este trabajo.

### 3.5 Los modelos empleados en el sistema
El sistema no propone modelos nuevos, sino que combina modelos fundacionales de referencia,
cada uno líder en su tarea. A continuación se describe qué hace cada uno, para qué se usa hoy,
quién lo desarrolla y qué papel cumple en el pipeline.

**HyperSIGMA.** Es un modelo fundacional para imágenes hiperespectrales, basado en Vision
Transformer y escalable a más de mil millones de parámetros. Lo desarrolla un equipo de
investigación encabezado por Wang y colaboradores de la Universidad de Wuhan junto con otras
instituciones, y se preentrena sobre un conjunto masivo de imágenes hiperespectrales. Se usa
como base común para tareas de interpretación hiperespectral, como clasificación de coberturas,
detección de cambios y segmentación, que se resuelven reutilizando sus representaciones. En el
pipeline extrae las características de cada imagen, a partir de las cuales se construye el mapa
de cambio [[wang2024hypersigma]].

**Segment Anything.** Es un modelo fundacional de segmentación promptable, capaz de delimitar
cualquier objeto de una imagen a partir de indicaciones espaciales como puntos o recuadros. Lo
desarrolla Meta AI. Se usa como bloque de segmentación de propósito general en edición de
imágenes, anotación de datos y numerosos sistemas de visión, y cada vez más como componente en
teledetección. En el pipeline recibe la región de cambio y produce su máscara precisa a nivel
de píxel [[kirillov2023sam]].

**CLIP.** Es un modelo de visión-lenguaje que proyecta imágenes y texto a un espacio común
donde la correspondencia se mide por similitud. Lo desarrolla OpenAI. Se usa para clasificación
zero-shot, recuperación entre imagen y texto y como base de sistemas de vocabulario abierto y
de generación de imágenes. En el pipeline asigna a cada región de cambio una etiqueta de
vocabulario abierto por similitud entre la imagen y las descripciones candidatas
[[radford2021clip]].

**BLIP.** Es un modelo de visión-lenguaje que unifica la comprensión y la generación de
lenguaje a partir de imágenes. Lo desarrolla Salesforce Research. Se usa para descripción de
imágenes, respuesta a preguntas visuales y recuperación multimodal. En el pipeline genera la
descripción en lenguaje natural de cada región de cambio, adaptado sobre datos hiperespectrales
[[li2022blip]].

**HyperCap.** No es un modelo sino un conjunto de datos, el primero orientado a la descripción
de imágenes hiperespectrales, con anotaciones textuales a nivel de píxel sobre varios conjuntos
de referencia. Lo desarrolla un equipo encabezado por Das y colaboradores, con participación de
grupos de investigación en teledetección. Se usa para entrenar y evaluar modelos de
visión-lenguaje en el dominio hiperespectral. En el pipeline es la base de entrenamiento del
módulo de descripción [[das2025hypercap]].

---

## 4. Tabla de fuentes

Cada fila es una fuente verificada y accesible, con la idea concreta que se extrae para el
marco de referencia.

| Clave de cita | Referencia | Enlace accesible | Idea que se extrae |
|---|---|---|---|
| `bhargava2024hyperspectral` | Bhargava et al., *Hyperspectral imaging and its applications: A review*, Heliyon 10(12):e33208, 2024 | doi.org/10.1016/j.heliyon.2024.e33208 | Definición de imagen hiperespectral y cubo de datos, adquisición, firma espectral, ventajas frente a RGB y aplicaciones. |
| `khelifi2020deep` | Khelifi & Mignotte, *Deep Learning for Change Detection in Remote Sensing Images: Comprehensive Review and Meta-Analysis*, IEEE Access 8:126385–126400, 2020 | arxiv.org/abs/2006.05612 | Definición y evolución de la detección de cambios; paso de técnicas clásicas a aprendizaje profundo; métricas de evaluación. |
| `kirillov2023sam` | Kirillov et al., *Segment Anything*, ICCV 2023 | arxiv.org/abs/2304.02643 | Segmentación promptable a nivel de píxel a partir de indicaciones espaciales, sin reentrenamiento. |
| `vaswani2017attention` | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 | arxiv.org/abs/1706.03762 | Definición de la arquitectura transformador y del mecanismo de atención. |
| `dosovitskiy2021vit` | Dosovitskiy et al., *An Image Is Worth 16x16 Words*, ICLR 2021 | arxiv.org/abs/2010.11929 | Vision Transformer: tratar la imagen como secuencia de parches; arquitectura base de los modelos del sistema. |
| `zhang2024vlm` | Zhang et al., *Vision-Language Models for Vision Tasks: A Survey*, IEEE TPAMI 46(8):5625–5644, 2024 | doi.org/10.1109/TPAMI.2024.3369699 | Definición de modelos de visión-lenguaje; representación conjunta imagen-texto para tareas guiadas por lenguaje. |
| `vinyals2015show` | Vinyals et al., *Show and Tell: A Neural Image Caption Generator*, CVPR 2015 | cv-foundation.org — Vinyals_Show_and_Tell_2015_CVPR | Origen neuronal de la descripción de imágenes: codificador convolucional más decodificador recurrente. |
| `lu2017rsicd` | Lu et al., *Exploring Models and Data for Remote Sensing Image Caption Generation*, IEEE TGRS 56(4):2183–2195, 2018 | arxiv.org/abs/1712.07835 | Descripción de imágenes en teledetección; conjunto RSICD; estado del arte previo al dominio hiperespectral. |
| `bengio2013representation` | Bengio, Courville & Vincent, *Representation Learning: A Review and New Perspectives*, IEEE TPAMI 35(8):1798–1828, 2013 | doi.org/10.1109/TPAMI.2013.50 | Fundamento del aprendizaje de representaciones; los *embeddings* como espacio semántico de las tareas posteriores. |
| `wang2024hypersigma` | Wang et al., *HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model*, IEEE TPAMI 2025 | arxiv.org/abs/2406.11519 | Definición de modelo fundacional hiperespectral; extracción de características; base del mapa de cambio por similitud. |
| `radford2021clip` | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021 | arxiv.org/abs/2103.00020 | Modelo de visión-lenguaje; espacio común imagen-texto; clasificación zero-shot y etiquetado por texto libre. |
| `pang2025special` | Pang et al., *SPECIAL: Zero-shot Hyperspectral Image Classification With CLIP*, arXiv 2025 | arxiv.org/abs/2501.16222 | Aprendizaje zero-shot aplicado a la clasificación de coberturas hiperespectrales. |
| `kombol2025trainingfree` | Kombol et al., *A Survey on Training-free Open-Vocabulary Semantic Segmentation*, arXiv 2025 | arxiv.org/abs/2505.22209 | Definición de segmentación de vocabulario abierto; papel de CLIP y de los modelos fundacionales de visión. |
| `zhu2025semanticcd` | Zhu et al., *Semantic-CD: Remote Sensing Image Semantic Change Detection towards Open-vocabulary Setting*, arXiv 2025 | arxiv.org/abs/2501.06808 | Vocabulario abierto aplicado a la detección de cambios semánticos en teledetección; frontera actual. |
| `li2022blip` | Li et al., *BLIP: Bootstrapping Language-Image Pre-training*, ICML 2022 | arxiv.org/abs/2201.12086 | Descripción de imágenes; modelo que unifica comprensión y generación de lenguaje a partir de imágenes. |
| `das2025hypercap` | Das et al., *HyperCap: Hyperspectral Land Cover Captioning Dataset for Vision Language Models*, IEEE GRSM 2026 | arxiv.org/abs/2505.12217 | Conjunto de datos y punto de referencia para descripción de imágenes hiperespectrales; base del captioning. |

---

## 5. Cómo seguir

1. Revisar y corregir cada definición de la sección 2.
2. Una vez aprobadas, pasar la prosa al esqueleto ya insertado en `main.tex`, sección
   *Marco de referencia*.
3. Redactar el estado del arte de la sección 3 como argumento hilado, no como lista.
