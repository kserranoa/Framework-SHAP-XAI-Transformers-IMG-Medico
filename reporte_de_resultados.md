# Reporte de Resultados:
### Framework SHAP + Transformers para Imágenes Médicas
**Karina Serrano Avendaño — Universidad de Costa Rica, 2026**

---

## Imagina esto primero...

Imagina que le preguntas al profesor: "¿Por qué le dio esa nota al estudiante?" y él te dice: "Porque no presento la tarea y en el examen obtuvo la mita de los puntos". Eso es una **explicación**. Sin esa respuesta, el estudiante no confiaría en el sistema educativo y *por qué* obtuvo esa nota.

Con las computadoras que leen radiografías pasa lo mismo. La computadora puede decir "este paciente tiene neumonía", pero si no explica *por qué zonas de la imagen llegó a esa conclusión*, los médicos no pueden confiar en ella. 

Según la propuesta de investigación, el problema central es que no existe un marco de trabajo estandarizado para que entidades, profesionales y autoridades puedan aplicar la técnica que mejor les sirva para entender cómo las inteligencias artificiales toman las decisiones en el campo médico.

---

## Parte 1: ¿Qué es SHAP? (La lupa mágica)

Imagina que tienes una foto de una radiografía de tórax. La computadora la mira y dice "este pulmón está enfermo". SHAP es como una **lupa mágica** que le pregunta a la computadora:

> "¿Cuáles partes de la imagen te hicieron pensar que el pulmón está enfermo?"

SHAP (que viene de *SHapley Additive exPlanations*) pinta cada píxel de la imagen con un color:

- **Rojo** = "Esta zona hace pensar que hay enfermedad"
- **Azul** = "Esta zona hace pensar que todo está bien"

El resultado es un mapa de colores encima de la radiografía, como si la computadora estuviera subrayando con marcadores de colores las partes importantes.

Según Pezzini (2024, p. 16), SHAP es una de las técnicas de explicabilidad más utilizadas porque es la única que da una garantía matemática de que el "valor" asignado a cada píxel es justo entre todos los píxeles de la imagen.

---

## Parte 2: ¿Qué son los Transformers? (Los robots que aprenden a mirar)

Los **Transformers** son un tipo especial de inteligencia artificial que, en vez de mirar la imagen píxel por píxel como lo haría un humano, divide la imagen en pequeños cuadraditos (parches) y analiza cómo se relacionan todos esos cuadraditos entre sí al mismo tiempo.

Es como si en vez de leer una página de un libro letra por letra, pudieras leer todas las palabras simultáneamente y entender cuáles son más importantes para el significado de la oración.

En este proyecto se usaron **cinco tipos** de Transformers, cada uno especializado en un tipo diferente de imagen médica:

| Transformer | Tipo de imagen | Tarea |
|---|---|---|
| **ViT** | Radiografías de tórax | ¿Hay neumonía? |
| **Swin** | Imágenes de tejidos (histopatología) | ¿El tumor es benigno o maligno? |
| **BEiT** | Imágenes de piel (dermatoscopía) | ¿Es melanoma? |
| **TransUNet** | Cortes de tomografía (2D) | Encontrar órganos en la imagen |
| **Swin UNETR** | Resonancias magnéticas (3D) | Encontrar tumores en 3D |

---

## Parte 3: ¿Cómo se sabe si la computadora hizo bien su trabajo?

Para saber si la computadora es buena en su trabajo, se usan tres tipos de "calificaciones":

### 3.1 La calificación de aciertos (métricas clínicas)

**AUC** (Área bajo la curva): Imagina que tienes 100 radiografías: 50 de pacientes enfermos y 50 de sanos. AUC mide cuántas veces la computadora pone a un enfermo por encima de un sano cuando las ordena por "probabilidad de estar enfermo". Un AUC de 1.0 = perfecta. Un AUC de 0.5 = igual que tirar una moneda al aire.

**F1**: Es como la nota final que combina dos cosas: cuántos enfermos detectó correctamente y cuántos sanos no confundió con enfermos. Va de 0 a 1. Más alto = mejor.

**Sensibilidad y Especificidad**: La sensibilidad mide "¿cuántos enfermos encontró?" y la especificidad mide "¿cuántos sanos no confundió?". Las dos juntas son el balance ideal de un buen médico.

### 3.2 La calificación de la explicación SHAP

- **FaithCorr (Fidelidad)**: ¿La lupa SHAP señala las partes que REALMENTE importan? Valor de 0 a 1. Más alto = mejor.
- **Sparsity (Concentración)**: ¿La lupa está enfocada en pocas zonas o se dispersa por toda la imagen? Valor cercano a 1 = muy dispersa. Valor cercano a 0 = muy concentrada (como una lupa de precisión).
- **SensitivityN (Sensibilidad de la explicación)**: ¿Si cambias un poquito la imagen, cambia la explicación? Si sí cambia, es buena señal porque significa que la explicación responde al contenido real.

### 3.3 La calificación de segmentación

**Dice Score**: Imagina que la computadora dibuja un contorno alrededor de un tumor y el médico también dibuja su propio contorno. El Dice Score mide cuánto se solapan los dos contornos. 1.0 = solapamiento perfecto. 0.0 = no se solapan en nada.

---

## Parte 4: Resultados de Clasificación (encontrar enfermedades)

### 4.1 ViT en Radiografías de Tórax

El **ViT** (Vision Transformer) analizó radiografías de tórax para detectar neumonía. Es como el médico más concentrado del grupo.

**Resultados:**
- AUC: **1.0** (¡Perfecto! Separó enfermos de sanos sin errores)
- F1: **0.899** (Muy bueno)
- Sensibilidad: 0.90
- Especificidad: 0.90

**Matriz de confusión:**

| | Predicho Sano | Predicho Enfermo |
|---|---|---|
| Real Sano | **5** ✓ | 0 |
| Real Enfermo | 1 | **4** ✓ |

De 10 casos, acertó 9. Solo confundió un paciente enfermo con sano. En términos clínicos, esto es alentador, aunque el conjunto de evaluación es muy pequeño para sacar conclusiones definitivas.

**¿Qué vio SHAP en las radiografías del ViT?**

A continuación se muestran los paneles SHAP del ViT. El mapa rojo/azul muestra exactamente qué partes de la radiografía usó la computadora para tomar su decisión:

![Panel SHAP ViT - Imagen 0](resultados_shap/ViT/shap_panel_img_000.png)
*Imagen 1: Panel SHAP del ViT para la primera radiografía. Las zonas rojas son las que más influyeron en la decisión del modelo.*

![Panel SHAP ViT - Imagen 1](resultados_shap/ViT/shap_panel_img_001.png)
*Imagen 2: Panel SHAP del ViT para la segunda radiografía.*

![Panel SHAP ViT - Imagen 2](resultados_shap/ViT/shap_panel_img_002.png)
*Imagen 3: Panel SHAP del ViT para la tercera radiografía.*

![Panel SHAP ViT - Imagen 3](resultados_shap/ViT/shap_panel_img_003.png)
*Imagen 4: Panel SHAP del ViT para la cuarta radiografía.*

**¿Dónde miró más el ViT?** Los "top patches" son los cuadraditos de la imagen donde la computadora concentró más su atención:

![Top Patches ViT - Imagen 0](resultados_shap/ViT/shap_top_patches_img_000.png)
*Imagen 5: Los parches más importantes según SHAP para el ViT (imagen 0).*

![Top Patches ViT - Imagen 1](resultados_shap/ViT/shap_top_patches_img_001.png)
*Imagen 6: Los parches más importantes según SHAP para el ViT (imagen 1).*

**El mapa de calor promedio del ViT:**

![Resumen Global SHAP ViT](resultados_shap/ViT/shap_resumen_global.png)
*Imagen 7: Mapa de calor promedio del ViT. Permite ver si el modelo tiene un sesgo sistemático (por ejemplo, si siempre mira las esquinas en lugar de los pulmones).*

**¿Qué pasa cuando tapamos partes de la imagen?**

![Curvas de perturbación ViT](resultados_shap/ViT/curvas_perturbacion_shap.png)
*Imagen 8: Curvas Deletion/Insertion del ViT. Cuando eliminamos los píxeles más importantes (línea de Deletion), la probabilidad debería bajar. Cuando los añadimos de vuelta (Insertion), debería subir.*

**Los umbrales de importancia:**

![Umbrales SHAP ViT - Imagen 0](resultados_shap/ViT/shap_umbrales_img_000.png)
*Imagen 9: Mapa de umbrales del ViT para la imagen 0. Muestra qué porcentaje de la imagen concentra el 50%, 80% y 95% de la importancia total.*

---

### 4.2 Swin Transformer en Histopatología

El **Swin** analizó imágenes de tejido tomadas con microscopio para detectar si un tumor es benigno o maligno. Su truco especial es dividir la imagen en "ventanas" que se mueven, lo que le ayuda a detectar patrones de textura muy finos, igual que los que diferencian células cancerosas de células sanas.

**Resultados:**
- AUC: **1.0**
- F1: **0.792** (Sólido para un prototipo entrenado en solo 2 épocas)
- Sensibilidad: 0.80
- Especificidad: 0.80

**Matriz de confusión:**

| | Predicho Benigno | Predicho Maligno |
|---|---|---|
| Real Benigno | **3** ✓ | 2 ❌ |
| Real Maligno | 0 | **5** ✓ |

Encontró todos los tumores malignos, pero marcó como malignos a 2 casos que eran benignos. En clínica, es preferible ser "más cuidadoso" (falsos positivos) que dejar pasar un cáncer real.

**¿Qué vio SHAP en los tejidos del Swin?**

![Panel SHAP Swin - Imagen 0](resultados_shap/Swin/shap_panel_img_000.png)
*Imagen 10: Panel SHAP del Swin para el primer parche de tejido histopatológico.*

![Panel SHAP Swin - Imagen 1](resultados_shap/Swin/shap_panel_img_001.png)
*Imagen 11: Panel SHAP del Swin para el segundo parche de tejido.*

![Panel SHAP Swin - Imagen 2](resultados_shap/Swin/shap_panel_img_002.png)
*Imagen 12: Panel SHAP del Swin para el tercer parche de tejido.*

![Panel SHAP Swin - Imagen 3](resultados_shap/Swin/shap_panel_img_003.png)
*Imagen 13: Panel SHAP del Swin para el cuarto parche de tejido.*

**Los parches más importantes para el Swin:**

![Top Patches Swin - Imagen 0](resultados_shap/Swin/shap_top_patches_img_000.png)
*Imagen 14: Top patches del Swin (imagen 0).*

![Top Patches Swin - Imagen 2](resultados_shap/Swin/shap_top_patches_img_002.png)
*Imagen 15: Top patches del Swin (imagen 2).*

**El mapa de calor promedio del Swin:**

![Resumen Global SHAP Swin](resultados_shap/Swin/shap_resumen_global.png)
*Imagen 16: Mapa de calor promedio del Swin sobre las imágenes de histopatología.*

**Curvas de perturbación del Swin:**

![Curvas de perturbación Swin](resultados_shap/Swin/curvas_perturbacion_shap.png)
*Imagen 17: Curvas de perturbación del Swin. La probabilidad baja de aprox. 0.62 a 0.34 cuando se eliminan los píxeles importantes.*

**Curvas de concentración de importancia:**

![Curva importancia Swin - img 0](resultados_shap/Swin/shap_curva_importancia_img_000.png)
*Imagen 18: Curva de concentración SHAP del Swin (imagen 0). Una curva que sube rápido al inicio indica que pocos píxeles concentran mucha importancia.*

---

### 4.3 BEiT en Dermatoscopía

El **BEiT** analizó imágenes de lesiones de piel para diferenciar melanoma de lunares benignos. Fue pre-entrenado aprendiendo a "reconstruir" partes de imágenes que se le ocultaban, algo parecido a cómo los humanos aprendemos a rellenar lo que no vemos.

**Resultados:**
- AUC: **NaN** ⚠️ (resultado no válido — explicación abajo)
- F1: **1.0** (aparentemente perfecto)
- Sensibilidad: 0.50
- Especificidad: 0.50

**¿Por qué el AUC es NaN?** Aquí hay un problema importante: en el conjunto de evaluación de BEiT solo había casos de UNA clase (todos eran positivos). Es como si le pidieras a un árbitro que calificara un partido donde solo juega un equipo. El F1 de 1.0 es engañoso. La sensibilidad y especificidad de 0.5 confirman que el modelo no puede discriminar entre clases con este conjunto de datos.

**Matriz de confusión:**

| | Predicho clase 0 | Predicho clase 1 |
|---|---|---|
| Real clase 0 | 0 | 0 |
| Real clase 1 | 0 | **5** ✓ |

Solo hay 5 casos y todos son de la misma clase. No se puede concluir que BEiT discrimina melanoma de nevos benignos con estos datos.

**¿Qué vio SHAP en las imágenes de piel del BEiT?**

![Panel SHAP BEiT - Imagen 0](resultados_shap/BEiT/shap_panel_img_000.png)
*Imagen 19: Panel SHAP del BEiT para la primera imagen dermatoscópica.*

![Panel SHAP BEiT - Imagen 1](resultados_shap/BEiT/shap_panel_img_001.png)
*Imagen 20: Panel SHAP del BEiT para la segunda imagen.*

![Panel SHAP BEiT - Imagen 2](resultados_shap/BEiT/shap_panel_img_002.png)
*Imagen 21: Panel SHAP del BEiT para la tercera imagen.*

![Panel SHAP BEiT - Imagen 3](resultados_shap/BEiT/shap_panel_img_003.png)
*Imagen 22: Panel SHAP del BEiT para la cuarta imagen.*

**Los parches más importantes para BEiT:**

![Top Patches BEiT - Imagen 0](resultados_shap/BEiT/shap_top_patches_img_000.png)
*Imagen 23: Top patches del BEiT (imagen 0). El 33.4% de la importancia está concentrada en solo el 10% de los píxeles.*

![Top Patches BEiT - Imagen 1](resultados_shap/BEiT/shap_top_patches_img_001.png)
*Imagen 24: Top patches del BEiT (imagen 1).*

**Mapa de calor promedio del BEiT:**

![Resumen Global SHAP BEiT](resultados_shap/BEiT/shap_resumen_global.png)
*Imagen 25: Mapa de calor promedio del BEiT. Permite ver si las zonas de interés están en la lesión o en bordes/pelo periférico.*

**Curvas de perturbación del BEiT:**

![Curvas de perturbación BEiT](resultados_shap/BEiT/curvas_perturbacion_shap.png)
*Imagen 26: Curvas de perturbación del BEiT. Muestra la caída más pronunciada de todos los modelos: de ~0.98 a ~0.35 cuando se eliminan píxeles importantes.*

**Umbrales de importancia del BEiT:**

![Umbrales SHAP BEiT - Imagen 0](resultados_shap/BEiT/shap_umbrales_img_000.png)
*Imagen 27: Mapa de umbrales del BEiT (imagen 0).*

![Curva importancia BEiT - img 0](resultados_shap/BEiT/shap_curva_importancia_img_000.png)
*Imagen 28: Curva de concentración del BEiT (imagen 0).*

---

## Parte 5: Resultados de Segmentación (dibujar contornos)

### 5.1 TransUNet en segmentación de órganos

El **TransUNet** tiene una tarea diferente y más difícil: en vez de decir "hay tumor" o "no hay tumor", tiene que **dibujar exactamente dónde están los órganos** en una tomografía 2D. Es como la diferencia entre decir "hay un gato en la foto" versus "aquí están los contornos exactos del gato".

**Resultados:**
- Dice Score: **0.0084** (prácticamente 0 — muy mala segmentación)
- IoU: **0.0044** (casi sin solapamiento)
- HD95: **148.84** (el contorno predicho está a 148 píxeles del contorno real)

**¿Por qué tan bajo?** La razón es técnica e importante:

1. El dataset que se usó fue **sintético** (generado artificialmente para demostración del código)
2. Solo se entrenó **2 épocas** (es como pedirle a un estudiante que aprenda cirugía viendo solo dos videos)
3. El conjunto de evaluación es el mismo que el de entrenamiento

Para que un modelo de segmentación sea clínicamente útil, necesita entre 20 y 30 épocas de entrenamiento con datos reales.

**¿Qué vio SHAP en TransUNet?**

![Panel SHAP TransUNet](resultados_shap/TransUNet/shap_panel_img_000.png)
*Imagen 29: Panel SHAP del TransUNet. Explica la clase 6 (un órgano específico en la imagen sintética). Aunque la segmentación es mala, el pipeline de SHAP funcionó y generó el mapa.*

![Resumen Global SHAP TransUNet](resultados_shap/TransUNet/shap_resumen_global.png)
*Imagen 30: Mapa de calor promedio del TransUNet. La concentración (top10_share de solo 15.8%) es la más baja de todos los modelos, lo que confirma que las explicaciones son muy difusas.*

![Top Patches TransUNet](resultados_shap/TransUNet/shap_top_patches_img_000.png)
*Imagen 31: Top patches del TransUNet (imagen 0).*

![Curva importancia TransUNet](resultados_shap/TransUNet/shap_curva_importancia_img_000.png)
*Imagen 32: Curva de concentración del TransUNet (imagen 0). Sube más lentamente que los clasificadores, indicando mayor dispersión.*

![Umbrales SHAP TransUNet](resultados_shap/TransUNet/shap_umbrales_img_000.png)
*Imagen 33: Mapa de umbrales del TransUNet (imagen 0).*

---

## Parte 6: La calidad de las explicaciones SHAP

### 6.1 Fidelidad: ¿La lupa apunta a lo correcto?

La **FaithCorr** (Faithfulness Correlation) responde a: "cuando SHAP dice que un píxel es importante, ¿de verdad cambia algo si lo tapamos?"

| Modelo | FaithCorr | Interpretación |
|---|---|---|
| ViT | **0.165** | El más fiel. Hay correlación real. |
| Swin | **0.110** | Correlación moderada. |
| BEiT | **0.057** | La más baja. Las atribuciones son menos predictivas. |

Todos los valores son bajos. Esto significa que las explicaciones SHAP tienen fidelidad limitada en este prototipo. Para mejorar esto se necesitarían más datos y más épocas de entrenamiento.

Según Doshi-Velez y Kim (2021), una buena explicación debe ser fiel al modelo: cuando se eliminan las zonas que SHAP marca como importantes, la predicción del modelo debe cambiar significativamente.

### 6.2 Concentración: ¿La lupa está enfocada?

La **Sparsity** (entropía normalizada) mide qué tan dispersa o concentrada es la explicación. En este código, valores altos (cerca de 1.0) indican que SHAP reparte importancia por muchas zonas, no que haya explicaciones limpias y localizadas.

| Modelo | Sparsity/Entropía | Concentración top-10% |
|---|---|---|
| ViT | 0.968 | 28.3% |
| Swin | 0.966 | 28.2% |
| BEiT | 0.963 | **33.4%** (más concentrado) |
| TransUNet | 0.992 | 15.8% (más disperso) |

Los valores de entropía son altos en todos los modelos, indicando que SHAP reparte la importancia entre muchas zonas de la imagen. El BEiT es el más concentrado (33.4% de la importancia en el 10% de los píxeles), mientras que el TransUNet es el más difuso.

### 6.3 Robustez: ¿La explicación cambia si la imagen cambia un poquito?

El **SensitivityN** mide si las explicaciones responden a cambios reales en la imagen o son genéricas.

| Modelo | SensitivityN | Significado |
|---|---|---|
| ViT | **-0.014** | Cercano a 0 (explicaciones poco sensibles a cambios locales) |
| Swin | **+0.056** | Ligeramente positivo (más sensible) |
| BEiT | **-0.013** | Cercano a 0 |

Valores cercanos a cero debilitan la afirmación de que las explicaciones son robustas. Se necesitan más iteraciones de evaluación para obtener medidas estables.

---

## Parte 7: Las curvas de perturbación (el "juego de esconder y revelar")

Imagina un juego: primero tapas las partes más importantes de la imagen (las que SHAP marcó como rojas). Si la computadora baja su confianza rápidamente, significa que SHAP identificó bien las zonas clave. Luego haces lo contrario: partes de una imagen en blanco y vas revelando las zonas importantes. Si la confianza sube rápido, ¡significa que SHAP encontró las piezas correctas del rompecabezas!

Este juego se llama **Deletion/Insertion AUC**:

| Modelo | Deletion AUC | Insertion AUC |
|---|---|---|
| ViT | 0.492 | 0.496 |
| Swin | 0.525 | 0.554 |
| BEiT | **0.817** | **0.922** |

El BEiT muestra la caída más pronunciada al eliminar píxeles importantes (de 0.98 a 0.35), lo que sugiere que esos píxeles realmente importan para su decisión. Pero hay que recordar que su resultado clínico era inválido por tener solo casos de una clase.

---

## Parte 8: El resumen global — ¿Qué funcionó y qué no?

| Modelo | Resultado clínico | Métrica principal | ¿Confiable? |
|---|---|---|---|
| **ViT** | AUC 1.0, F1 0.899 | Mejor clasificador del experimento | Parcialmente (solo 10 casos) |
| **Swin** | AUC 1.0, F1 0.792 | Bueno, 2 falsos positivos de 10 | Parcialmente (prototipo) |
| **BEiT** | F1 1.0, AUC NaN | Resultado inválido — solo 1 clase | No (datos desbalanceados) |
| **TransUNet** | Dice 0.008, HD95 148.8 | Segmentación prácticamente fallida | No (datos sintéticos, 2 épocas) |

---

## Parte 9: ¿Qué se cumplió y qué falta?

### Lo que sí funcionó

- El código del framework funciona de principio a fin
- SHAP generó mapas de explicación para todos los modelos
- Las métricas de evaluación clínica y XAI se calcularon correctamente
- Los archivos de resultados (JSON, CSV, PNG) se exportaron automáticamente

### Lo que necesita mejorarse antes de usarlo en clínica

Según el análisis de resultados, el objetivo técnico de prototipo se cumplió **parcialmente**, pero el objetivo científico/clínico **no queda demostrado todavía** por cuatro razones:

1. **Muy pocas imágenes**: Cada modelo se evaluó con solo 10 casos. Para un estudio clínico real se necesitan cientos o miles.
2. **Sin separación train/test real**: Los modelos se evaluaron con las mismas imágenes con que aprendieron (como hacer trampa en un examen).
3. **Faltan 2 modelos**: El Swin UNETR (segmentación 3D) y el Swin-Unet no completaron su evaluación en este prototipo.
4. **Datos sintéticos en segmentación**: TransUNet necesita el dataset Synapse real y entrenarse al menos 20-30 épocas.

Como señala la propuesta de investigación, la validación del framework requiere aplicar SHAP y explicaciones contrafactuales sobre arquitecturas Transformer a imágenes médicas reales (Serrano, 2026, p. 4).

---

## Parte 10: Lo que un médico vería en la carpeta de resultados

Cada subcarpeta de resultados contiene los siguientes archivos visuales que un médico podría revisar:

| Archivo | ¿Qué muestra? | Utilidad clínica |
|---|---|---|
| `shap_panel_img_XXX.png` | Mapa rojo/azul sobre la imagen | ¿Dónde "miró" la computadora? |
| `shap_curva_importancia_XXX.png` | Concentración de evidencia | ¿Cuántos píxeles explican la decisión? |
| `shap_top_patches.csv` | Coordenadas exactas de zonas importantes | Automatizar "bounding boxes" para el radiólogo |
| `shap_resumen_global.png` | Mapa de calor promedio | ¿Tiene sesgos el modelo? |
| `curvas_perturbacion_shap.png` | Curvas Deletion/Insertion | ¿Qué tan fiel es la explicación? |
| `resultado_modelo.json` | Todas las métricas numéricas | Comparar modelos entre sí |
| `matriz_confusion.csv` | Aciertos y errores del modelo | Diagnóstico clínico de fallos |

---

## Conclusión: El proyecto como primer paso

Imagina que estás construyendo un avión. Este prototipo sería como haber construido el **motor** y haber demostrado que enciende. Todavía falta el fuselaje, las alas, las pruebas de vuelo y la certificación. Pero el motor funciona, y eso es un gran avance.

El framework demuestra que es técnicamente posible conectar modelos de inteligencia artificial de imágenes médicas con explicaciones SHAP automáticas. El siguiente paso es aplicarlo con datos reales, más imágenes y más épocas de entrenamiento para que los médicos puedan confiar realmente en sus resultados.

Como afirman Arrieta et al. (2020, p. 82), la explicabilidad en inteligencia artificial no es un lujo sino una necesidad ética y regulatoria, especialmente en dominios de alto impacto como la medicina.

---

## Referencias Bibliográficas

Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion, 58*, 82–115.

Botero Arcila, B. (2024). AI liability in Europe: How does it complement risk regulation and deal with the problem of human oversight? *Computer Law & Security Review, 54*, 106012. (Botero Arcila, 2024, p. 106012)

Carvalho, D. V., Pereira, E. M., & Cardoso, J. S. (2019). Machine Learning Interpretability: A survey on methods and metrics. *Electronics, 8*(8), 832.

Doshi-Velez, F., & Kim, B. (2021). Evaluating the quality of machine learning explanations: A survey on methods and metrics. *ACM Computing Surveys, 54*(5), 1–38. (Doshi-Velez y Kim, 2021)

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., & Pedreschi, D. (2018). A survey of methods for explaining black box models. *ACM Computing Surveys, 51*(5), Article 93. https://doi.org/10.1145/3236009. (Guidotti et al., 2018)

Maslej, N., Fattorini, L., Perrault, R., Gil, Y., Parli, V., Kariuki, N., ... & Oak, S. (2025). *The AI Index 2025 Annual Report*. AI Index Steering Committee, Institute for Human-Centered AI, Stanford University. (Maslej et al., 2025, p. 20)

Pezzini, M. (2024). *Explainable AI: A survey of techniques and applications*. (p. 16)

Ramachandranpillai, R., Baeza-Yates, R., & Heintz, F. (2025). FairXAI - A Taxonomy and Framework for Fairness and Explainability Synergy in Machine Learning. *IEEE Transactions on Neural Networks and Learning Systems, 36*(6), 9819–9836. (Ramachandranpillai, Baeza-Yates y Heintz, 2025, p. 2)

Sachoulidou, A. (2024). AI Systems and Criminal Liability. *Oslo Law Review, 11*(1), 1–10. https://doi.org/10.18261/olr.11.1.3. (Sachoulidou, 2024)

Samek, W., Montavon, G., Lapuschkin, S., Anders, C. J., & Müller, K. R. (2019). Explainable AI methods – A brief overview. En *Explainable AI: Interpreting, explaining and visualizing deep learning* (pp. 1–18). Springer. (Samek et al., 2019, p. 1)

Serrano Avendaño, K. (2026). *Framework para la integración de SHAP y explicaciones contrafactuales en arquitecturas Transformer de inteligencia artificial aplicadas a imágenes médicas* [Propuesta de tesis de Maestría]. Universidad de Costa Rica.

Vilone, G., & Longo, L. (2020). Explainable Artificial Intelligence: A systematic review. *Information Fusion, 76*, 89–106. https://doi.org/10.1016/j.inffus.2021.05.009. (Vilone y Longo, 2020, p. 89)

Covert, I., Kim, C., & Lee, S.-I. (2026). Learning to Estimate Shapley Values with Vision Transformers. GitHub. https://github.com/suinleelab/vit-shapley. (Covert, Kim y Lee, 2026)

Fuse, Y., Murphy, S. N., Ikari, H., Takahashi, A., Fuse, K., & Kawakami, E. (2025). Artificial Intelligence in Clinical Data Analysis: A Review of Large Language Models, Foundation Models, Digital Twins, and Allergy Applications. *Allergology International*. (Fuse et al., 2025)

Gaggion, N., Mosquera, C., Aineseder, M., Mansilla, L., Milone, D., & Ferrante, E. (2023). CheXmask Database: a large-scale dataset of anatomical segmentation masks for chest x-ray images (versión 0.1). PhysioNet. https://doi.org/10.13026/dx54-8351. (Gaggion et al., 2023)

Gildenblat, J., & contributors. (2021). PyTorch library for CAM methods [Software]. GitHub. https://github.com/jacobgil/pytorch-grad-cam. (Gildenblat y contributors, 2021)

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2024). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *arXiv preprint arXiv:2103.14030*. (Liu et al., 2024)

Lundberg, S. (2018). An introduction to explainable AI with Shapley values. SHAP Documentation, Read the Docs. https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html. (Lundberg, 2018)

Ma, D., Hosseinzadeh Taher, M. R., Pang, J., Islam, N. U., Haghighi, F., Gotway, M. B., & Liang, J. (2022). Benchmarking and Boosting Transformers for Medical Image Classification. En *Domain Adaptation and Representation Transfer* (pp. 12–22). Springer Nature Switzerland. (Ma et al., 2022, pp. 12–22)

Mitchell, R., Frank, E., & Holmes, G. (2022). GPUTreeShap: Massively Parallel Exact Calculation of SHAP Scores for Tree Ensembles. GitHub. https://github.com/shap/shap. (Mitchell, Frank y Holmes, 2022)
