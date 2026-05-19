# GPBoost Spatial Predictor — QGIS Plugin

<div align="center">

![GPBoost](https://img.shields.io/badge/GPBoost-v1.6.7-2e7d32?style=for-the-badge)
![QGIS](https://img.shields.io/badge/QGIS-3.16+-589632?style=for-the-badge&logo=qgis)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Combines Tree-Boosting + Gaussian Processes for spatial prediction directly in QGIS**  
**Combina Tree-Boosting + Procesos Gaussianos para predicción espacial directamente en QGIS**

*Based on / Basado en: Sigrist F. (2022). Gaussian Process Boosting. JMLR 23(232):1–51.*

</div>

---

## 📋 Table of Contents / Tabla de Contenidos

- [What is GPBoost?](#-what-is-gpboost--qué-es-gpboost)
- [Mathematical Model](#-mathematical-model--modelo-matemático)
- [Installation](#-installation--instalación)
- [Plugin Structure](#-plugin-structure--estructura-del-plugin)
- [Step-by-Step Usage (with images)](#-step-by-step-usage--uso-paso-a-paso)
- [Testing with Sample Data](#-testing-with-sample-data--prueba-con-datos-de-ejemplo)
- [Parameters Reference](#-parameters-reference--referencia-de-parámetros)
- [Processing Toolbox](#-processing-toolbox)
- [Troubleshooting](#-troubleshooting--solución-de-problemas)
- [References](#-references--referencias)

---

## 🌿 What is GPBoost? / ¿Qué es GPBoost?

### English

**GPBoost** is a machine learning algorithm that combines two powerful techniques:

1. **Tree-Boosting** (like XGBoost/LightGBM): learns non-linear relationships between predictor variables and the response variable.
2. **Gaussian Process (GP)**: models spatial autocorrelation in the residuals (Tobler's First Law of Geography — *"everything is related to everything else, but near things are more related than distant things"*).

This combination overcomes the limitations of each method used independently:
- Standard kriging assumes **linear** relationships between covariates and the response → GPBoost learns **non-linear** relationships.
- Standard boosting ignores **spatial structure** → GPBoost models **spatial autocorrelation** explicitly.

### Español

**GPBoost** es un algoritmo de aprendizaje automático que combina dos técnicas poderosas:

1. **Tree-Boosting** (como XGBoost/LightGBM): aprende relaciones no lineales entre las variables predictoras y la variable respuesta.
2. **Proceso Gaussiano (GP)**: modela la autocorrelación espacial en los residuos (Primera Ley de Tobler — *"todo está relacionado con todo, pero las cosas cercanas están más relacionadas que las distantes"*).

Esta combinación supera las limitaciones de cada método usado de forma independiente:
- El kriging estándar asume relaciones **lineales** entre covariables y respuesta → GPBoost aprende relaciones **no lineales**.
- El boosting estándar ignora la **estructura espacial** → GPBoost modela la **autocorrelación espacial** explícitamente.

---

## 📐 Mathematical Model / Modelo Matemático

The GPBoost model decomposes the response variable as:

```
y(s) = F(X) + b(s) + ε
```

Where / Donde:

| Symbol / Símbolo | Description (EN) | Descripción (ES) |
|---|---|---|
| `y(s)` | Response variable at location s | Variable respuesta en la ubicación s |
| `F(X)` | Non-linear function learned via tree-boosting | Función no lineal aprendida via tree-boosting |
| `b(s)` | Gaussian Process: b(s) ~ GP(0, K_θ(s,s')) | Proceso Gaussiano para correlación espacial |
| `ε` | i.i.d. error: ε ~ N(0, σ²) | Error independiente: ε ~ N(0, σ²) |

### Exponential Covariance Function (default)

```
K_θ(sᵢ, sⱼ) = σ²_GP · exp(−‖sᵢ − sⱼ‖ / ρ)
```

Where `σ²_GP` is the GP marginal variance and `ρ` is the range parameter.

### Estimated Parameters / Parámetros estimados

After training, the model reports:

| Parameter | Interpretation (EN) | Interpretación (ES) |
|---|---|---|
| `error_variance` (σ²) | Nugget — pure measurement error | Pepita — error de medición puro |
| `gp_variance` (σ²_GP) | Sill — total spatial variability | Meseta — variabilidad espacial total |
| `gp_range` (ρ) | Range — distance of spatial influence | Rango — distancia de influencia espacial |

---

## ⚙️ Installation / Instalación

### Requirements / Requisitos

| Requirement | Version |
|---|---|
| QGIS | ≥ 3.16 |
| Python | 3.10 – 3.12 |
| gpboost | ≥ 1.5.0 |
| numpy | ≥ 1.20 |

### Step 1 — Copy plugin files / Copiar archivos del plugin

Copy the `gpboost_qgis_pluging/` folder to your QGIS plugins directory:

**Linux:**
```bash
cp -r gpboost_qgis_pluging/ ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

**Windows:**
```
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\
```

**macOS:**
```
~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/
```

### Step 2 — Install the `gpboost` Python package / Instalar el paquete Python `gpboost`

Open the QGIS Python Console (`Plugins → Python Console` or `Ctrl+Alt+P`) and run:

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "gpboost", "--break-system-packages"
])
```

> **Note (Linux):** The `--break-system-packages` flag is required on Python 3.12 with Debian/Ubuntu/Arch-based systems due to PEP 668. This is safe for a development environment.

> **Nota (Linux):** El flag `--break-system-packages` es necesario en Python 3.12 con sistemas basados en Debian/Ubuntu/Arch debido al PEP 668. Es seguro en entornos de desarrollo.

Verify the installation / Verificar la instalación:

```python
import sys, site
for p in site.getsitepackages():
    if p not in sys.path: sys.path.insert(0, p)
import gpboost as gpb
print("GPBoost version:", gpb.__version__)
# Expected output: GPBoost version: 1.6.7
```

### Step 3 — Enable the plugin in QGIS / Activar el plugin en QGIS

1. Open `Plugins → Manage and Install Plugins`
2. Click the **"Installed"** tab
3. Find **"GPBoost Spatial Predictor"**
4. Check the ✓ checkbox to activate it
5. The plugin appears under `Plugins → GPBoost` and in the toolbar

---

## 📁 Plugin Structure / Estructura del Plugin

```
gpboost_qgis_pluging/
│
├── __init__.py                 # Plugin entry point / Punto de entrada
├── metadata.txt                # QGIS plugin registry / Registro del plugin
├── gpboost_plugin.py           # Main plugin class / Clase principal
├── gpboost_provider.py         # Processing provider / Proveedor Processing
├── gpboost_algorithm.py        # Core algorithm (Processing Toolbox)
├── gpboost_dialog.py           # Interactive Qt dialog / Diálogo interactivo Qt
├── install_gpboost_deps.py     # Dependency installer helper
└── requirements.txt            # Python dependencies
```

```
test_plugin/
└── test_maiz_meta_gpboost.kml  # Sample test data (Meta, Colombia)
                                 # Datos de prueba (Meta, Colombia)
```

---

## 🖥️ Step-by-Step Usage / Uso Paso a Paso

The plugin offers **two ways** to run GPBoost / El plugin ofrece **dos formas** de ejecutar GPBoost:

| Mode / Modo | Access / Acceso | Best for / Ideal para |
|---|---|---|
| **Interactive Dialog** | `Plugins → GPBoost → GPBoost Spatial Predictor` | Quick exploration and model inspection |
| **Processing Toolbox** | `Processing → Toolbox → GPBoost` | Batch processing, raster export, Graphical Modeler |

---

### 🔷 Mode 1: Interactive Dialog / Modo 1: Diálogo Interactivo

#### Step 1 — Open the plugin / Abrir el plugin

Go to / Ve a: **`Plugins → GPBoost → GPBoost Spatial Predictor`**

The dialog has three tabs / El diálogo tiene tres pestañas:
- 📊 **Datos** (Data) — select layer and fields
- ⚙️ **Modelo** (Model) — configure hyperparameters
- 📈 **Resultados** (Results) — view output

```
┌─────────────────────────────────────────────┐
│  🌿 GPBoost Spatial Predictor               │
│  Tree-Boosting + Gaussian Process           │
│─────────────────────────────────────────────│
│  [ 📊 Datos ] [ ⚙️ Modelo ] [ 📈 Resultados ]│
│─────────────────────────────────────────────│
│  Capa de puntos:    [ gpboost_test     ▼ ]  │
│  Variable resp. (y):[ rendimiento      ▼ ]  │
│  Covariables (X):   [ ndvi              ]   │
│                     [ evi               ]   │
│                     [ elevacion         ]   │
│                     [ temp_media        ]   │
│─────────────────────────────────────────────│
│  [▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ 0%             ]     │
│  [ ▶ Ejecutar GPBoost ] [ ✕ Cancelar ]      │
└─────────────────────────────────────────────┘
```

#### Step 2 — Configure the Data tab / Configurar la pestaña Datos

In the **📊 Datos** tab:

1. **Capa de puntos** → Select your point vector layer  
   *(Must be a Point geometry layer with numeric attribute fields)*

2. **Variable respuesta (y)** → Select the field you want to predict  
   *(e.g., `rendimiento`, `ndvi`, `disease_severity`)*

3. **Covariables (X)** → Select one or more predictor fields  
   *(Hold `Ctrl+click` to select multiple fields)*  
   *(Leave empty to use coordinates as covariates — pure spatial GP kriging)*

> **Important / Importante:** The response variable field and covariate fields must be **numeric** (Double or Integer). String fields are automatically excluded from the list.

#### Step 3 — Configure the Model tab / Configurar la pestaña Modelo

In the **⚙️ Modelo** tab:

| Parameter (ES) | Parameter (EN) | Default | Recommended range |
|---|---|---|---|
| Función de covarianza GP | GP Covariance Function | `exponential` | See table below |
| Learning rate | Learning rate | `0.01` | 0.001 – 0.1 |
| Num. hojas | Num. leaves per tree | `31` | 8 – 128 |
| Profundidad máxima | Max depth | `3` | 2 – 6 |
| Iteraciones boosting | Boosting iterations | `50` | 50 – 500 |

**GP Covariance functions / Funciones de covarianza GP:**

| Function | Shape | Best for / Ideal para |
|---|---|---|
| `exponential` | Exponential decay | Default, most spatial data / Por defecto, mayoría de datos espaciales |
| `gaussian` | Gaussian bell | Very smooth spatial fields / Campos muy suaves |
| `matern32` | Matérn ν=3/2 | Moderately smooth / Moderadamente suave |
| `matern52` | Matérn ν=5/2 | Smoother than exponential / Más suave que exponencial |
| `powered_exponential` | Flexible power | Custom decay shapes / Formas de decaimiento personalizadas |

#### Step 4 — Run the model / Ejecutar el modelo

Click **`▶ Ejecutar GPBoost`**

The progress bar shows the training stages / La barra de progreso muestra las etapas de entrenamiento:
```
10%  → Extracting data from layer / Extrayendo datos de la capa
25%  → Configuring GP model / Configurando modelo GP
40%  → Training (n iterations) / Entrenando (n iteraciones)
80%  → Computing RMSE and GP parameters / Calculando RMSE y parámetros GP
100% → Complete! / ¡Completo!
```

#### Step 5 — Interpret results / Interpretar resultados

In the **📈 Resultados** tab you will see / En la pestaña verás:

```
==============================================
   ✅ GPBoost Entrenamiento Completo
==============================================
  GPBoost versión     : 1.6.7
  Muestras usadas     : 150
  RMSE (entrenamiento): 0.1823

  Parámetros GP estimados:
    error_variance         : 0.089200
    gp_variance            : 0.003800
    gp_range               : 0.074800
==============================================
  ℹ️  Para exportar raster de predicción:
  Processing → GPBoost → Train & Predict
==============================================
```

**Interpreting GP parameters / Interpretando los parámetros GP:**

- **`error_variance` (σ²)** — Pure error/nugget. Low value → data has good quality. If very high relative to `gp_variance`, the spatial signal is weak. / Error puro/pepita. Valor bajo → datos de buena calidad.
- **`gp_variance` (σ²_GP)** — Spatial variance captured by the GP. Higher values indicate strong spatial autocorrelation. / Varianza espacial capturada por el GP.
- **`gp_range` (ρ)** — Distance (in coordinate units) over which spatial correlation decays. Interpret in the same CRS units as your layer. / Distancia en la que la correlación espacial decae.

---

### 🔷 Mode 2: Processing Toolbox / Modo 2: Processing Toolbox

For **batch processing** and **raster prediction export**, use the Processing Toolbox algorithm.  
Para **procesamiento por lotes** y **exportación de rasters de predicción**, usa el algoritmo del Processing Toolbox.

#### Access / Acceso

`Processing → Toolbox → GPBoost → Spatial Prediction → GPBoost: Train & Predict (Spatial)`

#### Inputs / Entradas

| Input | Type | Description (EN) | Descripción (ES) |
|---|---|---|---|
| Training point layer | Vector (Point) | Layer with observations | Capa con observaciones |
| Target field (y) | Field | Response variable | Variable respuesta |
| Covariate fields (X) | Field(s) | Predictor variables | Variables predictoras |
| GP covariance function | Enum | Spatial correlation model | Modelo de correlación espacial |
| Learning rate | Float | Shrinkage per iteration | Reducción por iteración |
| Num. leaves | Int | Tree complexity | Complejidad del árbol |
| Max depth | Int | Max tree depth | Profundidad máxima |
| Boosting iterations | Int | Number of trees | Número de árboles |
| Prediction extent | Extent | Spatial extent to predict | Extensión para predecir |
| Pixel size | Float | Output raster resolution | Resolución del raster de salida |
| Use cross-validation | Boolean | Find optimal n_iter via CV | Encontrar n_iter óptimo via VC |
| CV folds | Int | Number of CV folds | Número de folds de validación cruzada |

#### Output / Salida

- **Prediction raster** (`.tif`): GeoTIFF with predicted values, LZW compressed, tiled.
- **RMSE**: Numeric output — training RMSE or cross-validation RMSE if CV enabled.

---

## 🧪 Testing with Sample Data / Prueba con Datos de Ejemplo

A test dataset is provided at / Se provee un dataset de prueba en:

```
test_plugin/test_maiz_meta_gpboost.kml
```

This dataset simulates **maize yield observations** in the **Meta department, Colombia**, with:  
Este dataset simula **observaciones de rendimiento de maíz** en el **departamento del Meta, Colombia**, con:

| Field / Campo | Description (EN) | Descripción (ES) | Units / Unidades |
|---|---|---|---|
| `rendimiento` | Maize grain yield | Rendimiento de grano de maíz | t/ha |
| `ndvi` | Normalized Difference Vegetation Index | Índice de Vegetación de Diferencia Normalizada | 0 – 1 |
| `evi` | Enhanced Vegetation Index | Índice de Vegetación Mejorado | 0 – 1 |
| `elevacion` | Elevation above sea level | Elevación sobre el nivel del mar | m |
| `temp_media` | Mean air temperature | Temperatura media del aire | °C |

**Spatial coverage / Cobertura espacial:** Meta, Colombia (lon: -73.5° to -71.5°, lat: 2.5° to 5.5°)  
**N observations / N observaciones:** 150 points  
**CRS:** EPSG:4326 (WGS 84)

### Quick test steps / Pasos de prueba rápida

**Step 1** — Load the test layer in QGIS / Cargar la capa de prueba en QGIS:
```
Layer → Add Layer → Add Vector Layer → test_plugin/test_maiz_meta_gpboost.kml
```
Or drag and drop the `.kml` file directly into the QGIS canvas.  
O arrastra el archivo `.kml` directamente al canvas de QGIS.

**Step 2** — Open the plugin / Abrir el plugin:
```
Plugins → GPBoost → GPBoost Spatial Predictor
```

**Step 3** — Configure as follows / Configurar de la siguiente manera:

```
📊 Datos tab:
  ├── Capa de puntos    →  test_maiz_meta_gpboost
  ├── Variable resp.(y) →  rendimiento
  └── Covariables (X)   →  ndvi  (Ctrl+click)
                           evi   (Ctrl+click)

⚙️ Modelo tab:
  ├── Función GP        →  exponential
  ├── Learning rate     →  0.01
  ├── Num. hojas        →  31
  ├── Profundidad       →  3
  └── Iteraciones       →  50
```

**Step 4** — Click `▶ Ejecutar GPBoost` and check the **📈 Resultados** tab.

**Expected results / Resultados esperados:**

```
RMSE (entrenamiento): ~0.15 – 0.25 t/ha
gp_variance         : > 0.001  (spatial signal present)
gp_range            : 0.05 – 2.0  (in decimal degrees)
```

### Quick test from Python Console / Prueba rápida desde la consola Python

You can also verify the full pipeline works from the QGIS Python Console before using the GUI:

```python
import sys, site
for p in site.getsitepackages():
    if p not in sys.path: sys.path.insert(0, p)

import gpboost as gpb
import numpy as np

# Simulate 100 points in Meta, Colombia
np.random.seed(42)
n = 100
coords = np.column_stack([
    np.random.uniform(-73.5, -71.5, n),  # longitude
    np.random.uniform(2.5, 5.5, n)       # latitude
])
X = np.random.uniform(0.3, 0.9, (n, 2))  # ndvi, evi
y = np.sin(3 * np.pi * X[:,0]) + np.random.normal(0, 0.2, n)

# Train GPBoost
gp_model  = gpb.GPModel(gp_coords=coords, cov_function="exponential", likelihood="gaussian")
data_train = gpb.Dataset(data=X, label=y)
bst = gpb.train(
    params={"learning_rate": 0.01, "max_depth": 3, "num_leaves": 31, "verbose": -1},
    train_set=data_train,
    gp_model=gp_model,
    num_boost_round=50
)
gp_model.summary()
print("✅ GPBoost works correctly / GPBoost funciona correctamente!")
```

---

## 📊 Parameters Reference / Referencia de Parámetros

### Boosting Parameters / Parámetros de Boosting

| Parameter | Default | Range | Effect (EN) | Efecto (ES) |
|---|---|---|---|---|
| `learning_rate` | 0.01 | 0.001 – 1.0 | Lower = slower but more precise learning | Menor = aprendizaje más lento pero preciso |
| `num_leaves` | 31 | 2 – 1024 | Higher = more complex trees, risk of overfitting | Mayor = árboles más complejos, riesgo sobreajuste |
| `max_depth` | 3 | -1 – 20 | Limits tree depth; -1 = unlimited | Limita profundidad; -1 = sin límite |
| `n_iter` | 50 | 10 – 5000 | Number of trees in ensemble | Número de árboles en el ensamble |

### GP Parameters (auto-estimated) / Parámetros GP (auto-estimados)

These are **not set by the user** — they are estimated from the data via maximum likelihood during training. They appear in the Results tab after training.

Estos **no son configurados por el usuario** — se estiman de los datos mediante máxima verosimilitud durante el entrenamiento.

| Parameter | Formula | Interpretation |
|---|---|---|
| `error_variance` | σ² | Nugget effect — measurement noise |
| `gp_variance` | σ²_GP | Partial sill — spatial signal strength |
| `gp_range` | ρ | Range in map units — spatial influence distance |

### Recommendations by use case / Recomendaciones por caso de uso

| Use Case | Learning Rate | Leaves | Depth | Iterations |
|---|---|---|---|---|
| Small dataset (n < 100) | 0.05 | 15 | 2 | 50 |
| Medium dataset (100 – 500) | 0.01 | 31 | 3 | 100 – 200 |
| Large dataset (> 500) | 0.005 | 63 | 4 | 200 – 500 |
| Many covariates (> 10) | 0.01 | 63 | 5 | 200 |

---

## 🔧 Troubleshooting / Solución de Problemas

### Problem: Plugin does not appear in the Plugins menu
### Problema: El plugin no aparece en el menú de Plugins

**Cause / Causa:** Plugin folder name must match the `name` in `metadata.txt`.

**Solution / Solución:**
```bash
# Check the folder exists
ls ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/gpboost_qgis_pluging/
# Must contain metadata.txt
```

---

### Problem: `ModuleNotFoundError: No module named 'gpboost'`

**Cause / Causa:** QGIS Python cannot find the `gpboost` package installed in the system.

**Solution / Solución:** Run in QGIS Python Console:
```python
import sys, site
for p in site.getsitepackages():
    if p not in sys.path: sys.path.insert(0, p)
sys.path.insert(0, site.getusersitepackages())
import gpboost
print(gpboost.__version__)  # Should print 1.6.7
```

This is **permanently fixed** in `gpboost_plugin.py` via the `_fix_sys_path()` function that runs automatically when the plugin loads.

---

### Problem: Combo boxes (layer, fields) appear empty
### Problema: Los combos (capa, campos) aparecen vacíos

**Cause / Causa:** The dialog was opened before the layer was loaded, or `sys.path` was not configured.

**Solution / Solución:**
```python
# Reload the plugin from QGIS Python Console
from qgis.utils import reloadPlugin
reloadPlugin('gpboost_qgis_pluging')
```
Then reopen the dialog. The `_populate_layers()` function refreshes the layer list every time the dialog is opened.

---

### Problem: Training error — `CalledProcessError` or `Booster init failed`
### Problema: Error de entrenamiento

**Common causes / Causas comunes:**
1. Not enough valid points (minimum 10 required / mínimo 10 requeridos)
2. Missing values (None/NULL) in target or covariate fields
3. Covariate matrix is 1D instead of 2D (fixed in `gpboost_dialog.py` v2)

**Diagnostic / Diagnóstico:**
```python
# Check your layer for null values
layer = iface.activeLayer()
nulls = sum(1 for f in layer.getFeatures() if f["rendimiento"] is None)
print(f"Null values in target field: {nulls}")
```

---

### Problem: Grid prediction raster is too large
### Problema: El raster de predicción es demasiado grande

**Solution / Solución:** Increase the pixel size in the Processing Toolbox algorithm. The limit is 2,000,000 pixels (n_cols × n_rows). For a 1° × 1° extent in EPSG:4326, use pixel size ≥ 0.001 (≈ 111m).

---

## 📚 References / Referencias

1. Sigrist, F. (2022). **Gaussian Process Boosting**. *Journal of Machine Learning Research (JMLR)*, 23(232), 1–51. https://www.jmlr.org/papers/v23/20-1262.html

2. Sigrist, F. (2023). **Latent Gaussian Model Boosting**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. https://doi.org/10.1109/TPAMI.2022.3168152

3. GPBoost Official Documentation. https://gpboost.readthedocs.io

4. GPBoost GitHub Repository. https://github.com/fabsig/GPBoost

5. Sigrist, F. (2021). Tree-Boosting for Spatial Data. *Towards Data Science*. https://medium.com/data-science/tree-boosting-for-spatial-data-789145d6d97d

---

## 👥 Authors / Autores

**Santiago Gallego** & **Jesús Enrique Flores Riera**  
GPBoost QGIS Spatial Predictor Plugin, 2026  
Repository: https://github.com/jf-floresriera/GPBoost-Spatial-Predictor

---

## 📄 License / Licencia

MIT License — Free to use, modify and distribute with attribution.  
MIT License — Libre para usar, modificar y distribuir con atribución.

