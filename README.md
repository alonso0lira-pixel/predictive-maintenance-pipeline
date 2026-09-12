# Predictive Maintenance Pipeline

TFM sobre un pipeline reproducible de ingeniería de datos y detección de anomalías en MetroPT-3, un conjunto de mediciones de una unidad de producción de aire ferroviaria.

El proyecto transforma las señales originales en ventanas temporales, extrae 44 características y compara Isolation Forest, One-Class SVM, Local Outlier Factor y un autoencoder denso. Las etiquetas de los cuatro fallos documentados se utilizan exclusivamente para evaluar; no intervienen en el entrenamiento ni en el cálculo de las puntuaciones.

## Datos y protocolo

Descarga el dataset desde [UCI MetroPT-3](https://doi.org/10.24432/C5VW3R) y extrae el fichero `MetroPT3(AirCompressor).csv`. Consulta en la fuente sus condiciones de uso y cita el dataset al reutilizarlo.

Guarda el CSV en `data/raw/MetroPT3(AirCompressor).csv`, dentro de la raíz del repositorio. Los datos originales y procesados no se distribuyen con el código.

| Elemento | Configuración del experimento |
|---|---|
| Entrenamiento | Febrero de 2020 |
| Evaluación | Desde el 1 de marzo de 2020 hasta el final del fichero, el 1 de septiembre |
| Corte temporal | `2020-03-01` |
| Segmentación | Una diferencia entre registros superior a 13 segundos inicia otro segmento |
| Ventanas | 60 observaciones, desplazamiento de 30 |
| Características | 28 de señales analógicas y 16 de señales digitales |
| Etiqueta positiva | Solapamiento con un fallo documentado igual o superior al 50 % |
| Métricas principales | ROC-AUC y Average Precision |

Los modelos se entrenan con configuraciones fijas, sin tuning sobre evaluación. One-Class SVM, LOF y el autoencoder ajustan StandardScaler únicamente con entrenamiento. Isolation Forest utiliza las características sin escalado.

## Instalación

El proyecto declara **Python 3.14 o posterior**. Las dependencias y sus intervalos de versiones están en [pyproject.toml](pyproject.toml). Instalar el extra `dev` incorpora pytest.

Desde una terminal:

```bash
git clone https://github.com/alonso0lira-pixel/predictive-maintenance-pipeline.git
cd predictive-maintenance-pipeline
```

En Windows PowerShell, crea el entorno y utiliza directamente su intérprete:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe
```

En Linux o macOS, con Python 3.14 instalado:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python
```

El último comando abre Python. Ejecuta los bloques siguientes en esa sesión, manteniendo la raíz del repositorio como directorio de trabajo. No es necesario utilizar notebooks.

## Reproducción del experimento

### 1. Validar y transformar el CSV

Crea `data/raw` si no existe y coloca allí el CSV descargado antes de ejecutar:

```python
from pathlib import Path
from predictive_maintenance.pipeline import run_pipeline

root = Path.cwd()
processed = root / "data/processed/metropt3.parquet"

pipeline_report = run_pipeline(
    input_path=root / "data/raw/MetroPT3(AirCompressor).csv",
    output_path=processed,
)
print(pipeline_report)
```

Las discontinuidades temporales se registran como advertencias y se gestionan mediante segmentación. Un resultado `completed_with_warnings` puede ser esperado para este dataset; revisa el informe.

### 2. Generar las características con separación temporal

```python
from predictive_maintenance.feature_pipeline import run_cutoff_feature_pipeline

features_dir = root / "data/processed/features"
feature_report = run_cutoff_feature_pipeline(
    input_path=processed,
    output_dir=features_dir,
    cutoff_timestamp="2020-03-01",
    window_size=60,
    step_size=30,
)
print(feature_report)

train_path = features_dir / "train_february.parquet"
evaluation_path = features_dir / "evaluation_mar_sep.parquet"
```

Utiliza `run_cutoff_feature_pipeline` para reproducir la memoria. La función alternativa `run_feature_pipeline` implementa una separación por proporciones y no corresponde al experimento final.

### 3. Comparar los cuatro detectores

```python
from predictive_maintenance.model_comparison import run_model_comparison

comparison_dir = root / "data/processed/results/model_comparison"
comparison = run_model_comparison(
    train_features_path=train_path,
    evaluation_features_path=evaluation_path,
    output_dir=comparison_dir,
)
print(comparison["model_comparison"].to_string(index=False))
print(comparison["model_failure_metrics"].to_string(index=False))

# Las métricas locales se devuelven en memoria; esta línea las exporta.
comparison["model_local_horizon_metrics"].to_csv(
    comparison_dir / "model_local_horizon_metrics.csv", index=False
)
```

La función guarda automáticamente `model_comparison.csv` y `model_failure_metrics.csv`. La exportación del tercer CSV se hace explícitamente en el ejemplo.

### 4. Reproducir el análisis detallado de Isolation Forest

```python
from predictive_maintenance.experiment import run_anomaly_experiment

experiment = run_anomaly_experiment(
    train_features_path=train_path,
    evaluation_features_path=evaluation_path,
    n_estimators=100,
    random_state=42,
    overlap_threshold=0.50,
    output_dir=root / "data/processed/results/isolation_forest",
)
print(experiment["global_metrics"])
print(experiment["output_files"])
```

Esta ejecución vuelve a entrenar Isolation Forest con la misma configuración y genera:

- `global_metrics.json`.
- `failure_metrics.csv`.
- `local_horizon_metrics.csv`.
- `failure_roc_auc.png`.
- `local_horizon_roc_auc.png`.

Las funciones actuales devuelven métricas agregadas; no exportan las puntuaciones individuales de todas las ventanas ni los modelos entrenados.

## Resultados de referencia de la memoria

Las cifras siguientes proceden del experimento documentado en la memoria. Sirven para contrastar una reproducción; no son resultados de una ejecución automática al consultar este repositorio.

- 1.516.948 registros originales.
- 368 segmentos temporales.
- 7.125 ventanas de entrenamiento y 42.941 de evaluación.
- 990 ventanas positivas y 41.951 negativas según el criterio de etiquetado.
- Ventanas positivas por fallo: 287, 79, 572 y 52.

| Modelo | ROC-AUC | Average Precision |
|---|---:|---:|
| Isolation Forest | 0,9791 | 0,3548 |
| One-Class SVM | 0,9636 | 0,2448 |
| Local Outlier Factor | 0,9692 | 0,2512 |
| Autoencoder denso | 0,9660 | 0,2479 |

Las versiones de las bibliotecas y el entorno numérico pueden afectar a la reproducción exacta; los tiempos dependen también del hardware. Para identificar una entrega, registra el commit y las versiones instaladas:

```bash
git rev-parse HEAD
```

En Windows:

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

En Linux o macOS:

```bash
.venv/bin/python -m pip freeze
```

Actualmente `pyproject.toml` declara intervalos de versiones, no un entorno bloqueado con versiones exactas.

## Interpretación y limitaciones

- Una puntuación mayor indica mayor anomalía; no mide directamente la gravedad física de un fallo.
- Las ventanas negativas no cumplen el criterio de asociación a los fallos documentados. No necesariamente representan funcionamiento normal.
- El autoencoder utiliza un umbral interno P95 del MSE de entrenamiento para `predict()`. ROC-AUC y AP utilizan puntuaciones continuas y no dependen de ese umbral. No se ha validado una política industrial de alarmas.
- El análisis local compara cada horizonte con las ventanas de referencia situadas entre siete días y 24 horas antes del fallo, excluyendo las etiquetadas como fallo de esa referencia.
- Los horizontes se asignan por el **punto medio** de la ventana. Cerca de una frontera temporal, una ventana puede incluir muestras del intervalo siguiente. Por ello, este análisis es retrospectivo y no demuestra por sí solo anticipación en tiempo real; una validación operativa tendría que considerar cuándo termina de observarse la ventana.
- Solo hay cuatro eventos documentados y las ventanas consecutivas se solapan. Las métricas describen este experimento y no establecen un horizonte universal de anticipación.
- El autoencoder alcanzó el límite de 100 iteraciones en el experimento documentado sin satisfacer el criterio de convergencia. Los avisos de convergencia deben revisarse; no justifican ajustar hiperparámetros mirando la evaluación.

## Organización del código

| Ruta o módulo | Función |
|---|---|
| `notebooks/` | Exploración inicial y calidad de datos |
| `validation.py`, `transformation.py`, `storage.py` | Validación, segmentación y almacenamiento |
| `pipeline.py` | Orquestación del procesamiento del CSV |
| `splitting.py`, `windowing.py`, `features.py` | Separación temporal, ventanas y características |
| `feature_pipeline.py` | Generación y persistencia de los conjuntos de características |
| `modeling.py`, `anomaly_detection.py` | Entrada de modelos, entrenamiento y puntuaciones |
| `ground_truth.py`, `labeling.py`, `evaluation.py` | Fallos documentados, etiquetado y evaluación |
| `experiment.py`, `model_comparison.py`, `visualization.py` | Experimentos, comparación y figuras |
| `tests/` | Pruebas automatizadas |

Los módulos se encuentran en `src/predictive_maintenance/`. La suite utiliza casos sintéticos para verificar comportamiento y contratos; esos casos no sustituyen la evaluación experimental sobre MetroPT-3.
