# Framework SHAP-XAI · Transformers para Imagen Médica

Prototipo en Google Colab (GPU T4) que entrena seis arquitecturas transformer sobre imágenes médicas y aplica SHAP para explicar sus predicciones.

## Modelos

| Modelo | Tarea | Dataset |
|---|---|---|
| ViT | Clasificación radiografías de tórax | ChestX-ray14 |
| Swin Transformer | Clasificación histopatología | PatchCamelyon |
| BEiT v2 | Clasificación dermatoscopía | ISIC 2019 |
| Swin UNETR | Segmentación 3D cerebral | BraTS 2021 |
| TransUNet | Segmentación 2D abdominal | Synapse |
| Swin-UNet | Segmentación 2D cardíaca | ACDC |

## Pipeline

1. Instalación de dependencias (`transformers`, `timm`, `monai`, `shap`, `nibabel`, `einops`)
2. Descarga automática de datasets (Hugging Face Datasets + Kaggle API)
3. Entrenamiento de cada modelo en su carpeta dedicada
4. Generación de valores SHAP (GradientExplainer / PartitionExplainer)
5. Evaluación con métricas de explicabilidad y rendimiento clínico

## Métricas

**Explicabilidad:** Fidelidad, Robustez, Localización, Complejidad

**Clasificación:** AUC, F1, Sensibilidad, Especificidad

**Segmentación:** Dice, IoU, HD95

## Requisitos

```
transformers==4.44.2  timm==1.0.9  monai>=1.4.0
einops==0.8.0  nibabel==5.2.1  shap  torch
```
