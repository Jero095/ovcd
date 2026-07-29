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
de atención, que pondera la relación entre todos los elementos de una secuencia sin
recurrir a convoluciones ni a recurrencia [[vaswani2017attention]]. El Vision Transformer
traslada esta idea a las imágenes: divide la imagen en parches, los trata como una
secuencia de fichas y aplica atención sobre ellos, alcanzando un desempeño competitivo en
reconocimiento visual cuando se preentrena con grandes volúmenes de datos
[[dosovitskiy2021vit]]. Esta arquitectura es la base común de los modelos empleados en el
sistema, tanto para el análisis hiperespectral como para la segmentación y la alineación
entre imagen y texto.

### 2.5 Modelos fundacionales
Un modelo fundacional es un modelo de gran escala preentrenado sobre cantidades masivas
de datos, diseñado para servir como base reutilizable en múltiples tareas posteriores sin
necesidad de entrenarlo desde cero. En el dominio hiperespectral, HyperSIGMA es el primer
modelo fundacional con más de mil millones de parámetros, preentrenado sobre un conjunto
extenso de imágenes hiperespectrales, capaz de extraer características espaciales y
espectrales transferibles a tareas como clasificación, detección de cambios y
segmentación [[wang2024hypersigma]].

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
predefinido de clases. Esto se logra gracias a modelos que alinean el espacio visual con
el espacio del lenguaje, de modo que la etiqueta se decide por similitud entre la región y
una descripción textual arbitraria [[kombol2025trainingfree]]. En teledetección, esta idea
se ha comenzado a aplicar a la detección de cambios semánticos, superando la limitación de
las clases fijas [[zhu2025semanticcd]].

### 2.8 Modelos de visión-lenguaje y descripción de imágenes
Los modelos de visión-lenguaje aprenden una representación conjunta de imágenes y texto a
partir de grandes colecciones de pares imagen y descripción. CLIP entrena dos codificadores,
uno visual y uno textual, para proyectar ambas modalidades a un espacio común donde la
correspondencia se mide por similitud, lo que habilita la clasificación por texto libre y
el etiquetado de vocabulario abierto [[radford2021clip]]. La descripción de imágenes, o
generación de subtítulos, va un paso más allá y produce frases en lenguaje natural que
resumen el contenido visual; BLIP es un modelo que unifica comprensión y generación para
esta tarea [[li2022blip]]. Para el dominio hiperespectral existe HyperCap, un conjunto de
datos y punto de referencia orientado a la descripción de coberturas terrestres a partir
de imágenes hiperespectrales [[das2025hypercap]].

### 2.9 Representaciones vectoriales y similitud coseno
Los modelos profundos representan cada imagen o región mediante un vector de características,
conocido como *embedding*, que condensa su contenido semántico en un espacio de alta
dimensión. La comparación entre dos representaciones se realiza con la similitud coseno,
que mide el ángulo entre ambos vectores y arroja un valor alto cuando apuntan en la misma
dirección. En este trabajo, el mapa de cambio se construye calculando la similitud coseno
entre las representaciones de la imagen inicial y la final extraídas por el modelo
fundacional: donde la similitud es baja, hay cambio [[wang2024hypersigma]].

### 2.10 Métricas de evaluación
Para valorar el desempeño de la detección de cambios se emplean métricas derivadas de la
comparación con una verdad de terreno. La precisión mide qué proporción de los cambios
detectados son correctos, la exhaustividad mide qué proporción de los cambios reales fueron
detectados, y el puntaje F1 combina ambas en un único valor mediante su media armónica. La
intersección sobre unión evalúa el solapamiento entre la región detectada y la real. Estas
métricas son el estándar en la revisión de métodos de detección de cambios basados en
aprendizaje profundo [[khelifi2020deep]].

### 2.11 Modelos y datos del sistema
Esta subsección reúne, ya como tecnología concreta, las piezas del sistema:

- **HyperSIGMA** — modelo fundacional hiperespectral que extrae las representaciones profundas de cada imagen [[wang2024hypersigma]].
- **Segment Anything** — modelo de segmentación promptable que convierte las regiones de cambio en máscaras precisas a partir de indicaciones espaciales [[kirillov2023sam]].
- **CLIP** — modelo de visión-lenguaje que asigna etiquetas de vocabulario abierto por similitud entre imagen y texto [[radford2021clip]].
- **BLIP** — modelo de descripción de imágenes que genera frases en lenguaje natural [[li2022blip]].
- **HyperCap** — conjunto de datos y punto de referencia para la descripción de imágenes hiperespectrales, base del módulo de captioning [[das2025hypercap]].

---

## 3. Estado del arte (borrador de ejes)

El estado del arte no es una lista, es un argumento que termina en el vacío que llena este
trabajo. Se propone organizarlo en tres ejes y un cierre:

1. **Detección de cambios en teledetección** — de las técnicas clásicas basadas en píxeles a los métodos de aprendizaje profundo [[khelifi2020deep]], con la frontera actual en la detección semántica y de vocabulario abierto [[zhu2025semanticcd]].
2. **Segmentación y reconocimiento de vocabulario abierto** — el auge de los modelos fundacionales de segmentación [[kirillov2023sam]] y de visión-lenguaje [[radford2021clip]], y su combinación para segmentar sin clases fijas [[kombol2025trainingfree]].
3. **Modelos fundacionales y zero-shot en imágenes hiperespectrales** — la llegada de un modelo fundacional propio del dominio [[wang2024hypersigma]] y los primeros enfoques zero-shot [[pang2025special]] y de descripción [[das2025hypercap]].
4. **Vacío de investigación** — el vocabulario abierto y el zero-shot están resueltos en imágenes RGB y empiezan en teledetección, pero nadie los ha llevado a la detección de cambios sobre imágenes hiperespectrales bitemporales. Ese es el aporte de este trabajo.

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
