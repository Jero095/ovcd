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

Formalmente, la atención de producto punto escalado calcula, a partir de las matrices de
consultas $Q$, claves $K$ y valores $V$,

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,$$

donde $d_k$ es la dimensión de las claves y el factor $\sqrt{d_k}$ evita que el producto
crezca demasiado y sature la función softmax. Cada elemento de la secuencia queda así
representado como una combinación ponderada de los demás, con pesos que expresan su
relevancia mutua [[vaswani2017attention]].

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

#### 2.5.1 Arquitectura de HyperSIGMA: SpatViT y atención por muestreo disperso
La extracción de características de este trabajo recae en SpatViT, la subred espacial de
HyperSIGMA. Es un Vision Transformer que divide la imagen hiperespectral en parches no
solapados y los procesa con atención; en la variante base que se emplea aquí, produce un mapa
de características denso de dimensión de *embedding* 768 a un dieciseisavo de la resolución
espacial de entrada [[wang2024hypersigma]].

El problema es que la autoatención completa tiene un costo cuadrático en el número de fichas,
y las imágenes hiperespectrales presentan una fuerte redundancia espacial y espectral, por lo
que atender de forma uniforme a todas las posiciones es ineficiente. SpatViT lo resuelve con
la atención por muestreo disperso, o *sparse sampling attention*. En lugar de comparar cada
consulta con todas las claves, cada consulta $\mathbf{q}$ predice $N_p$ desplazamientos de
muestreo mediante una proyección lineal aprendida $W_p$, y las claves y los valores se
muestrean únicamente en esas posiciones desplazadas mediante interpolación bilineal, formando
los conjuntos reducidos $K'$ y $V'$. La atención se calcula entonces solo sobre ese conjunto
disperso:

$$\operatorname{SSA}(Q,K',V')=\operatorname{softmax}\!\left(\frac{Q\,K'^{\top}}{\sqrt{D'}}\right)V',$$

donde $D'$ es la dimensión de las claves. Así, cada consulta aprende qué regiones son
relevantes en vez de recorrerlas todas, lo que captura contexto diverso de largo alcance con
un costo mucho menor que la autoatención completa [[wang2024hypersigma]]. En el pipeline,
SpatViT con esta atención es el codificador que genera las representaciones de la imagen
inicial y la final, sobre las que se calcula el mapa de cambio por similitud coseno.

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

La descripción de imágenes, también llamada generación de subtítulos, va un paso más allá de
la clasificación: en lugar de asignar una etiqueta, genera una frase en lenguaje natural que
resume el contenido visual. Los primeros modelos neuronales de esta tarea acoplan dos piezas,
un codificador convolucional que condensa la imagen en un vector y un decodificador recurrente
que, a partir de ese vector, produce la frase palabra por palabra; sus autores lo describen
como un modelo que "can be used to generate natural sentences describing an image"
[[vinyals2015show]]. Hoy la línea la dominan los modelos de visión-lenguaje preentrenados a gran
escala, como BLIP, que reúnen en una sola arquitectura la comprensión de la imagen y la
generación del texto que la describe [[li2022blip]].

En teledetección, la descripción de imágenes se apoya en conjuntos de datos propios. RSICD,
por ejemplo, reúne cerca de once mil imágenes de teledetección con varias descripciones cada
una, y permite entrenar y evaluar modelos adaptados a las escenas aéreas y satelitales
[[lu2017rsicd]]. Ese avance, sin embargo, se dio sobre imágenes ópticas de tres canales. La
extensión al dominio hiperespectral es mucho más reciente: HyperCap es el primer conjunto de
datos y punto de referencia para describir coberturas a partir de imágenes hiperespectrales, y
hace posible aplicar modelos de visión-lenguaje sobre el cubo espectral completo
[[das2025hypercap]].

En nuestro sistema, estos modelos ocupan las dos últimas etapas semánticas del pipeline. CLIP
asigna a cada región de cambio una etiqueta de vocabulario abierto por similitud entre imagen
y texto [[radford2021clip]], y un modelo de descripción de tipo BLIP, adaptado sobre HyperCap,
genera una frase en lenguaje natural que caracteriza la región con más riqueza que una sola
etiqueta [[li2022blip]] [[das2025hypercap]].

### 2.9 Representaciones vectoriales y similitud coseno
El aprendizaje profundo parte de que el desempeño en una tarea depende sobre todo de la
calidad de la representación de los datos [[bengio2013representation]]. Un modelo profundo
transforma cada imagen o región en un vector de características o *embedding*
$\mathbf{v}\in\mathbb{R}^{d}$, con $d$ del orden de cientos, en un espacio donde la proximidad
entre vectores corresponde a similitud semántica.

La comparación entre dos *embeddings* $\mathbf{u}$ y $\mathbf{v}$ usa la similitud coseno:

$$\operatorname{sim}(\mathbf{u},\mathbf{v})=\frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}=\frac{\sum_{i=1}^{d}u_i v_i}{\sqrt{\sum_{i=1}^{d}u_i^{2}}\,\sqrt{\sum_{i=1}^{d}v_i^{2}}}\in[-1,1].$$

La normalización por las magnitudes $\lVert\mathbf{u}\rVert$ y $\lVert\mathbf{v}\rVert$ elimina
la escala y conserva solo la orientación, por lo que la medida es insensible a diferencias de
brillo o ganancia entre las dos capturas.

HyperSIGMA entrega un *embedding* por posición del mapa de características
[[wang2024hypersigma]]. Para cada posición $p$, con vectores $\mathbf{e}_1(p)$ y
$\mathbf{e}_2(p)$ de las imágenes inicial y final, el mapa de cambio es

$$C(p)=1-\operatorname{sim}\!\big(\mathbf{e}_1(p),\mathbf{e}_2(p)\big),$$

donde un valor alto indica cambio y uno bajo permanencia. En la implementación $C$ se
interpola de forma bilineal al tamaño original, se renormaliza a $[0,1]$ y se binariza con el
método de Otsu para delimitar las regiones de cambio. La misma similitud coseno rige el
etiquetado, al comparar el *embedding* de una región con el de cada descripción textual
candidata.

### 2.10 Métricas de evaluación
La evaluación de la detección de cambios se plantea como una comparación píxel a píxel entre
el resultado del sistema y una verdad de terreno, de la que surgen cuatro conteos: verdaderos
positivos $VP$ cuando se detecta un cambio real, verdaderos negativos $VN$ cuando se identifica
correctamente una zona sin cambio, falsos positivos $FP$ cuando se señala un cambio inexistente
y falsos negativos $FN$ cuando se pasa por alto un cambio real. Sobre estos conteos se definen
las métricas de desempeño.

La precisión $P$ mide qué proporción de los cambios detectados son correctos y penaliza los
falsos positivos, mientras que la exhaustividad $R$ mide qué proporción de los cambios reales
se detectaron y penaliza los falsos negativos:

$$P=\frac{VP}{VP+FP},\qquad R=\frac{VP}{VP+FN}.$$

Como suele existir una tensión entre ambas, el puntaje F1 las combina en un único valor
mediante su media armónica, que solo es alta cuando las dos lo son:

$$F_1=\frac{2\,P\,R}{P+R}=\frac{2\,VP}{2\,VP+FP+FN}.$$

La intersección sobre unión evalúa el solapamiento espacial entre la región detectada y la
real, dividiendo el área común entre el área total que cubren juntas:

$$\mathrm{IoU}=\frac{VP}{VP+FP+FN}.$$

Estas métricas son el estándar en la revisión de métodos de detección de cambios basados en
aprendizaje profundo [[khelifi2020deep]].

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

## 3. Estado del arte

El estado del arte se organiza siguiendo el recorrido del pipeline. Para cada etapa se revisa
cómo se resuelve hoy el problema, qué alternativas existen, y se justifica la herramienta
elegida; al mencionar cada modelo se indica quién lo desarrolla. El sistema no propone modelos
nuevos, sino que integra modelos fundacionales de referencia, cada uno líder en su tarea.

### 3.1 Imágenes hiperespectrales y su adquisición
La imagen hiperespectral nace de la espectrometría de imagen, formulada por Goetz en 1985 como la adquisición simultánea de imágenes en muchas bandas espectrales
contiguas para obtener, en cada píxel, un espectro completo con el que identificar los
materiales de la superficie terrestre [[goetz1985imaging]]. Desde los primeros sensores
aerotransportados, como AVIRIS del Jet Propulsion Laboratory de la NASA, la tecnología
evoluciona hacia sensores satelitales como Hyperion del satélite EO-1, que amplían la cobertura
y consolidan la disponibilidad de datos con cientos de bandas [[bhargava2024hyperspectral]].

En cuanto al método de toma, la adquisición se realiza mayoritariamente por barrido: los
sensores de empuje, o pushbroom, registran una línea espacial completa en todas las bandas a la
vez, mientras que los de barrido por punto, o whiskbroom, recorren la escena punto a punto;
existen además cámaras de instantánea que capturan todo el cubo de una vez a costa de
resolución. En todos los casos hay un compromiso entre resolución espacial, resolución
espectral y relación señal a ruido [[bhargava2024hyperspectral]]. En este trabajo los datos
entran como pares bitemporales en formatos ENVI, MATLAB o GeoTIFF, y se reinterpolan a un
espacio espectral de referencia, el del sensor Hyperion, que es el mismo con el que se
preentrena el modelo fundacional que extrae las características.

### 3.2 Extracción de características: transformers y modelos fundacionales hiperespectrales
El núcleo del sistema es la representación de cada imagen. La arquitectura dominante para
aprender representaciones visuales es el transformador, propuesto por Vaswani, de Google, en 2017, que sustituye la convolución y la recurrencia por el mecanismo de atención
[[vaswani2017attention]], y su adaptación a imágenes, el Vision Transformer de Dosovitskiy, de Google, en 2021, que trata la imagen como una secuencia de parches [[dosovitskiy2021vit]].
Sobre esta base surgen los modelos fundacionales, preentrenados a gran escala y reutilizables
sin entrenar desde cero. Esta arquitectura es transversal al proyecto: el transformador es el
esqueleto común de los tres modelos fundacionales que el sistema integra, HyperSIGMA para la
extracción de características, Segment Anything para la segmentación [[kirillov2023sam]] y CLIP
para el etiquetado [[radford2021clip]]. El proyecto no construye un transformador propio, sino
que reutiliza modelos ya preentrenados sobre él, lo que confirma que se trata de un concepto
central del sistema aunque su implementación provenga de terceros.

Existen dos maneras de obtener esta representación. La tradicional es calcularla a mano
mediante índices espectrales, fórmulas que combinan bandas concretas para resaltar un material,
como los índices de vegetación; son simples pero rígidas, pues cada índice captura una sola
propiedad y exige saber de antemano qué se busca. La alternativa moderna, que adopta este
trabajo, es aprenderla con un modelo fundacional. HyperSIGMA representa la imagen dividiendo el
cubo en parches y procesándolos con un Vision Transformer espacial que, mediante su atención por
muestreo disperso, produce un mapa denso de vectores de características, uno por posición, que
codifican de forma conjunta la información espacial y espectral de la escena. Es el primer modelo
fundacional hiperespectral con más de mil millones de parámetros, desarrollado por Wang, de la
Universidad de Wuhan, y preentrenado sobre el conjunto HyperGlobal-450K con cerca de 450.000
imágenes [[wang2024hypersigma]]. La elección se justifica porque estas representaciones son mucho
más ricas que un índice manual y, al haberse aprendido sobre datos hiperespectrales, resultan
transferibles a tareas nuevas sin reentrenar, como demuestra su desempeño superior en dieciséis
conjuntos de datos y siete tareas [[wang2024hypersigma]]. Su código es público en el repositorio
oficial del grupo WHU-Sigma [[whu2024hypersigma_repo]].

### 3.3 Detección de cambios y el emparejamiento en el espacio latente
La detección de cambios compara imágenes de una misma escena en instantes distintos para
localizar lo que cambió, y sirve al monitoreo del uso del suelo, la evaluación de desastres y el
seguimiento urbano y agrícola. Las técnicas se agrupan en cuatro familias. Las de álgebra de imágenes operan directamente sobre
los valores de los píxeles, como la diferencia, el cociente o el análisis de vectores de cambio.
Las de transformación proyectan los datos a otro espacio antes de compararlos, como el análisis
de componentes principales. Las de comparación post-clasificación clasifican cada imagen por
separado y luego contrastan las clases. Y, de forma dominante hoy, las de aprendizaje profundo
aprenden a detectar el cambio con redes siamesas y de atención [[khelifi2020deep]]. En cuanto a en qué
se aplica, la detección de cambios opera sobre imágenes multiespectrales, de radar de apertura
sintética y, cada vez más, hiperespectrales, donde la alta resolución espectral permite detectar
cambios finos. Un referente es GETNET, una red convolucional de dos dimensiones de extremo a
extremo propuesta por Wang en 2019 para detección de cambios hiperespectral [[wang2019getnet]].
Para manejar la alta dimensionalidad del cubo, GETNET no compara solo las bandas, sino que
incorpora información de abundancia a nivel de subpíxel obtenida por desmezclado espectral y la
combina en una matriz de afinidad mixta que fusiona varias fuentes de información; sobre esa
representación, la red aprende a clasificar cada píxel como cambiado o no cambiado. Junto con el
método, los autores publican un conjunto de datos de referencia, conocido como River, creado
expresamente para comparar de forma objetiva los distintos métodos sobre los mismos datos, algo
que antes escaseaba en este dominio. La frontera actual es la detección de cambios semántica y de
vocabulario abierto [[zhu2025semanticcd]].

La mayoría de estos métodos requiere entrenar con pares etiquetados, escasos en el dominio
hiperespectral. La alternativa que adopta este trabajo es prescindir del entrenamiento: se
comparan las representaciones que HyperSIGMA extrae de las dos capturas mediante la similitud
coseno, la misma métrica con la que los modelos de aprendizaje contrastivo, como CLIP, emparejan
vectores en un espacio latente común [[radford2021clip]]. Donde la similitud entre las
representaciones inicial y final es baja, hay cambio. Se justifica por ser un enfoque zero-shot,
que no depende de datos etiquetados de cambio y aprovecha directamente la calidad de las
representaciones del modelo fundacional.

### 3.4 Segmentación
La segmentación delimita las regiones a nivel de píxel. Su estado del arte pasa de las redes
convolucionales de codificador y decodificador, como U-Net de Ronneberger en
2015 [[ronneberger2015unet]], Mask R-CNN de He, de Facebook AI Research, en 2017
[[he2017maskrcnn]] y la familia DeepLab, todas entrenadas para un conjunto fijo de clases, al
paradigma de los modelos fundacionales de segmentación promptable. El referente es Segment
Anything, desarrollado por Meta AI en 2023, entrenado sobre el conjunto SA-1B con once millones
de imágenes y mil millones de máscaras, capaz de segmentar cualquier objeto a partir de
indicaciones espaciales sin reentrenarse [[kirillov2023sam]]; su sucesor, SAM 2, extiende la
capacidad a video [[ravi2024sam2]]. El modelo y sus pesos están disponibles en el sitio oficial
de Meta [[meta2023sam_repo]].

Las alternativas entrenadas por clase exigirían anotar datos de las coberturas de interés, lo
que choca con el carácter abierto del problema. Este trabajo usa Segment Anything porque produce
máscaras precisas sin depender de clases predefinidas ni de reentrenamiento, y se integra de
forma natural recibiendo como indicaciones los puntos y recuadros derivados del mapa de cambio.
Como Segment Anything opera sobre imágenes de tres canales, se le entrega una composición RGB
del cubo hiperespectral, y cada máscara se confina a la región de cambio.

### 3.5 Etiquetado de vocabulario abierto
Para nombrar cada región sin un conjunto cerrado de clases, el estado del arte se apoya en los
modelos de visión-lenguaje, que alinean imagen y texto en un espacio común [[zhang2024vlm]]. El
referente es CLIP, desarrollado por OpenAI en 2021, entrenado por contraste sobre cerca de
cuatrocientos millones de pares imagen-texto, que clasifica en modo zero-shot comparando la
imagen con descripciones textuales arbitrarias [[radford2021clip]]; su implementación oficial es
pública [[openai2021clip_repo]]. En teledetección esta idea ya se traslada a la segmentación y la
detección de cambios de vocabulario abierto [[kombol2025trainingfree]] [[zhu2025semanticcd]], y
en el dominio hiperespectral a la clasificación zero-shot [[pang2025special]].

La alternativa tradicional, un clasificador de clases fijas, obligaría a definir de antemano todo
lo que puede aparecer y a reentrenar para añadir una categoría. Este trabajo usa CLIP porque
convierte el etiquetado en una comparación por similitud contra un vocabulario de texto libre,
editable sin reentrenar, que es la esencia del objetivo de vocabulario abierto. En el pipeline,
CLIP asigna a cada máscara la etiqueta más afín de ese vocabulario.

### 3.6 Descripción en lenguaje natural
La última etapa enriquece la etiqueta con una frase. La descripción de imágenes nace con modelos
de codificador convolucional y decodificador recurrente, como Show and Tell de Vinyals, de Google, en 2015 [[vinyals2015show]], y hoy la dominan los modelos de
visión-lenguaje preentrenados como BLIP, desarrollado por Salesforce Research en 2022, que
unifica comprensión y generación [[li2022blip]]; su código es público [[salesforce2022blip_repo]].
En teledetección la descripción cuenta con conjuntos propios como RSICD [[lu2017rsicd]], pero
limitados a imágenes ópticas de tres canales. El salto al dominio hiperespectral es muy reciente
y lo habilita HyperCap, el primer conjunto de descripción hiperespectral con anotaciones a nivel
de píxel, desarrollado por Das, del Vellore Institute of Technology [[das2025hypercap]].

La alternativa de aplicar un modelo de descripción de imágenes naturales ignora la información
espectral del cubo. Por eso este trabajo, en su etapa opcional de descripción, adapta un modelo
de tipo BLIP cuyo codificador visual se sustituye por uno que consume directamente el cubo
hiperespectral, entrenado sobre HyperCap. Se justifica porque genera una caracterización más
rica que una sola etiqueta y aprovecha la dimensión espectral completa.

### 3.7 Aprendizaje zero-shot
El aprendizaje zero-shot designa la capacidad de un modelo para resolver una tarea o reconocer
categorías que no aparecieron en su entrenamiento, apoyándose en conocimiento previo que
relaciona lo conocido con lo nuevo. El término se acuña con Palatucci en 2009,
que predicen clases nunca vistas mediante códigos de salida semánticos [[palatucci2009zeroshot]],
en paralelo con los enfoques basados en atributos de Lampert, que transfieren
descripciones entre clases [[lampert2009attribute]].

Con la llegada de los modelos preentrenados a gran escala, el zero-shot se generaliza. CLIP
clasifica imágenes en categorías descritas solo con texto, sin ejemplos etiquetados
[[radford2021clip]], y la idea se extiende al reconocimiento de vocabulario abierto, la
segmentación [[kombol2025trainingfree]] y, en teledetección, a la clasificación hiperespectral
[[pang2025special]] y la detección de cambios semánticos [[zhu2025semanticcd]]. Su validez
descansa en que un modelo fundacional, tras preentrenarse sobre volúmenes masivos de datos,
captura conocimiento general y transferible, que permite generalizar a tareas y categorías
nuevas sin verlas, siempre que se expresen en un espacio semántico común, por ejemplo el del
lenguaje [[wang2024hypersigma]] [[radford2021clip]].

Las alternativas son el aprendizaje supervisado, que exige un conjunto etiquetado para cada
tarea y clase; el ajuste fino de un modelo preentrenado, que necesita algunos ejemplos anotados;
y el aprendizaje con pocos ejemplos, o few-shot. Todas requieren datos etiquetados del problema
concreto. Este trabajo adopta el zero-shot porque los pares hiperespectrales bitemporales con
verdad de terreno son escasos y costosos de anotar, y porque el objetivo mismo, detectar y
nombrar cambios con vocabulario libre, es incompatible con un conjunto cerrado de clases
entrenadas. El sistema encadena modelos fundacionales sin entrenar sobre la tarea: compara
representaciones por similitud coseno para el cambio y compara con texto para la etiqueta, de
modo que todo el proceso opera sin ejemplos etiquetados de detección de cambios.

### 3.8 Vacío de investigación
La revisión anterior muestra que cada pieza está resuelta por separado: hay modelos fundacionales
hiperespectrales [[wang2024hypersigma]], segmentación promptable [[kirillov2023sam]], etiquetado
de vocabulario abierto [[radford2021clip]] y descripción hiperespectral [[das2025hypercap]]. El
vocabulario abierto y el zero-shot están consolidados en imágenes RGB y empiezan a aparecer en
teledetección [[zhu2025semanticcd]] [[kombol2025trainingfree]]. Sin embargo, ninguna propuesta
los integra para detectar y describir cambios, con vocabulario libre y sin entrenamiento, sobre
imágenes hiperespectrales bitemporales. Ese vacío es el que aborda este trabajo.

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
| `goetz1985imaging` | Goetz, Vane, Solomon & Rock, *Imaging Spectrometry for Earth Remote Sensing*, Science 228(4704):1147–1153, 1985 | doi.org/10.1126/science.228.4704.1147 | Origen de la espectrometría de imagen; nacimiento histórico de la imagen hiperespectral. |
| `ronneberger2015unet` | Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015 | arxiv.org/abs/1505.04597 | Alternativa de segmentación: red convolucional codificador-decodificador entrenada por clase. |
| `he2017maskrcnn` | He, Gkioxari, Dollár & Girshick, *Mask R-CNN*, ICCV 2017 | arxiv.org/abs/1703.06870 | Alternativa de segmentación de instancias entrenada por clase, de Facebook AI Research. |
| `ravi2024sam2` | Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv 2024, Meta AI | arxiv.org/abs/2408.00714 | Sucesor de SAM; segmentación promptable extendida a video. |
| `wang2019getnet` | Wang, Yuan, Du & Li, *GETNET: A General End-to-End 2-D CNN Framework for Hyperspectral Image Change Detection*, IEEE TGRS 57(1):3–13, 2019 | arxiv.org/abs/1905.01662 | Detección de cambios hiperespectral con CNN; conjunto de referencia River. |
| `whu2024hypersigma_repo` | WHU-Sigma, *HyperSIGMA — repositorio oficial* | github.com/WHU-Sigma/HyperSIGMA | Fuente del desarrollador de HyperSIGMA; código y pesos. |
| `meta2023sam_repo` | Meta AI, *Segment Anything — sitio oficial* | segment-anything.com | Fuente del desarrollador de SAM; documentación, demo y pesos. |
| `openai2021clip_repo` | OpenAI, *CLIP — repositorio oficial* | github.com/openai/CLIP | Fuente del desarrollador de CLIP; código e implementación. |
| `salesforce2022blip_repo` | Salesforce Research, *BLIP — repositorio oficial* | github.com/salesforce/BLIP | Fuente del desarrollador de BLIP; código e implementación. |
| `palatucci2009zeroshot` | Palatucci, Pomerleau, Hinton & Mitchell, *Zero-shot Learning with Semantic Output Codes*, NIPS 2009 | papers.nips.cc/paper/3650 | Origen del término aprendizaje zero-shot; predicción de clases no vistas. |
| `lampert2009attribute` | Lampert, Nickisch & Harmeling, *Learning to Detect Unseen Object Classes by Between-Class Attribute Transfer*, CVPR 2009 | doi.org/10.1109/CVPR.2009.5206594 | Zero-shot basado en atributos; transferencia de descripciones entre clases. |
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

## 6. Figuras

La lista completa y autoritativa de las figuras del plan, con su fuente y su ubicación, vive en
un archivo aparte: `figuras-plan.md`. Se mantiene ahí para no duplicarla ni desincronizarla.
