# GPBoost Spatial Predictor — QGIS Plugin

<div align="center">

![GPBoost](https://img.shields.io/badge/GPBoost-≥1.4.0-2e7d32?style=for-the-badge)
![QGIS](https://img.shields.io/badge/QGIS-3.16+-589632?style=for-the-badge&logo=qgis)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Plugin-v2.0.0-orange?style=for-the-badge)

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
- [New in V2: Model Comparison & Tuning](#-new-in-v2-model-comparison--tuning--nuevo-en-v2)
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
| gpboost | ≥ 1.4.0 |
| numpy | ≥ 1.20 |

### Step 1 — Copy plugin files / Copiar archivos del plugin

Copy the `gpboost_spatial_predictor_V2/` folder to your QGIS plugins directory:

**Linux:**
```bash
cp -r gpboost_spatial_predictor_V2/ ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
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

**Option A — Automatic installer (recommended) / Instalador automático (recomendado):**

Open the QGIS Python Console (`Plugins → Python Console` or `Ctrl+Alt+P`) and run:

```python
exec(open("/path/to/plugin/install_gpboost_deps.py").read())
```

**Option B — Manual installation / Instalación manual:**

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "gpboost>=1.4.0", "--break-system-packages"
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
gpboost_spatial_predictor_V2/
│
├── __init__.py                 # Plugin entry point / Punto de entrada
├── metadata.txt                # QGIS plugin registry / Registro del plugin
├── gpboost_plugin.py           # Main plugin class / Clase principal
├── gpboost_provider.py         # Processing provider / Proveedor Processing
├── gpboost_algorithm.py        # Core algorithm (Processing Toolbox)
├── gpboost_dialog.py           # Interactive Qt dialog v3.1 / Diálogo interactivo Qt v3.1
├── install_gpboost_deps.py     # Dependency installer helper
└── icons/
    └── icon.png                # Plugin icon
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
| **Interactive Dialog** | `Plugins → GPBoost → GPBoost Spatial Predictor` | Exploration, model inspection, multi-model comparison |
| **Processing Toolbox** | `Processing → Toolbox → GPBoost` | Batch processing, raster export, Graphical Modeler |

---

### 🔷 Mode 1: Interactive Dialog v3.1 / Modo 1: Diálogo Interactivo v3.1

The dialog now has **four tabs** / El diálogo ahora tiene **cuatro pestañas**:

| Tab / Pestaña | Function (EN) | Función (ES) |
|---|---|---|
| 📊 **Datos** | Select layer and fields | Seleccionar capa y campos |
| ⚙️ **Modelo** | Configure hyperparameters | Configurar hiperparámetros |
| 🔬 **Comparar** | Multi-model tuning table | Tabla de comparación de múltiples modelos |
| 📈 **Resultados** | View output and GP parameters | Ver salida y parámetros GP |

#### Step 1 — Open the plugin / Abrir el plugin

Go to / Ve a: **`Plugins → GPBoost → GPBoost Spatial Predictor`**

#### Step 2 — Configure the Data tab / Configurar la pestaña 📊 Datos

1. **Capa de puntos** → Select your point vector layer
2. **Variable respuesta (y)** → Select the numeric field to predict
3. **Covariables (X)** → Select one or more predictor fields (`Ctrl+click` for multiple)
4. **Normalizar variables** ✓ → *(New in V2)* Standardizes y and X before training (recommended for heterogeneous covariates / recomendado para covariables heterogéneas)

> **Tip:** Leave covariates empty to use coordinates as covariates — pure spatial GP kriging.

#### Step 3 — Configure the Model tab / Configurar la pestaña ⚙️ Modelo

| Parameter (ES) | Parameter (EN) | Default | Recommended range |
|---|---|---|---|
| Función de covarianza GP | GP Covariance Function | `exponential` | See table below |
| Learning rate | Learning rate | `0.01` | 0.001 – 0.1 |
| Num. hojas | Num. leaves per tree | `31` | 8 – 128 |
| Profundidad máxima | Max depth | `3` | 2 – 6 |
| Iteraciones boosting | Boosting iterations | `50` | 50 – 500 |

**GP Covariance functions / Funciones de covarianza GP (V2 expanded):**

| Function | Parameters | Best for / Ideal para |
|---|---|---|
| `exponential` | shape = 0.0 | Default, most spatial data |
| `gaussian` | shape = 0.0 | Very smooth spatial fields |
| `matern (v=0.5)` | shape = 0.5 | Equivalent to exponential |
| `matern (v=1.5)` | shape = 1.5 | Moderately smooth fields |
| `matern (v=2.5)` | shape = 2.5 | Smoother than exponential |
| `powered_exponential` | shape = 1.0 | Custom decay shapes |
| `wendland` | shape = 0.0 | Compact support, large datasets |

#### Step 4 — Run the model / Ejecutar el modelo

Click **`▶ Ejecutar GPBoost`** for a single model, or **`⚡ Comparar/Tuning`** for multi-model comparison.

The progress bar shows / La barra de progreso muestra:
```
10%  → Extracting data from layer
15%  → Normalizing variables (if enabled)
25%  → Configuring GP model
40%  → Training (n iterations)
80%  → Computing RMSE, R² and GP parameters
100% → Complete!
```

#### Step 5 — Interpret results / Interpretar resultados

In the **📈 Resultados** tab:

```
==================================================
   GPBoost -- Resultado del modelo unico
==================================================
  GPBoost version     : 1.6.x
  Muestras            : 150
  Normalizado         : Si / No
  Funcion GP          : exponential
  Learning rate       : 0.01
  Iteraciones         : 50

  RMSE                : 0.1823
  R²                  : 0.8741          ← New in V2

  Parametros GP estimados:
    error_variance         : 0.089200
    gp_variance            : 0.003800
    gp_range               : 0.074800
==================================================
  Para comparar modelos -> pestaña Comparar
==================================================
```

**Interpreting GP parameters / Interpretando los parámetros GP:**

- **`error_variance` (σ²)** — Pure error/nugget. Low value → good data quality.
- **`gp_variance` (σ²_GP)** — Spatial variance captured by the GP. Higher → strong spatial autocorrelation.
- **`gp_range` (ρ)** — Distance (in CRS units) over which spatial correlation decays.

---

## 🆕 New in V2: Model Comparison & Tuning / Nuevo en V2

The **🔬 Comparar** tab allows comparing multiple model configurations in a single run.

### How to use / Cómo usar

1. Configure the base parameters in the **📊 Datos** and **⚙️ Modelo** tabs
2. Go to **🔬 Comparar** tab
3. Add experiments manually to the table, or load a **Preset** (e.g., "Grid Search básico")
4. Each row defines: `Label | Learning Rate | Num. Leaves | Max Depth | N. Iter | Cov. Function | Cov. Shape`
5. Click **`⚡ Comparar/Tuning`** — the plugin trains all experiments sequentially

### Results table / Tabla de resultados

| Column | Description |
|---|---|
| Modelo | Experiment label |
| LR | Learning rate used |
| N. iter | Boosting iterations |
| Función GP | Covariance function |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination (new) |
| GP Var | Estimated GP variance (σ²_GP) |
| Rango | Estimated GP range (ρ) |

The **best model** (lowest RMSE) is highlighted in green. All failed models are highlighted in red.

### Variable normalization / Normalización de variables

When **Normalizar variables** is enabled:
- `y` is standardized: `y_norm = (y − ȳ) / σ_y`
- `X` is standardized column-wise: `X_norm = (X − X̄) / σ_X`
- Predictions are back-transformed to original units automatically
- Recommended when covariates have very different scales (e.g., NDVI ∈ [0,1] and elevation ∈ [0, 5000 m])

---

### 🔷 Mode 2: Processing Toolbox / Modo 2: Processing Toolbox

For **batch processing** and **raster prediction export**, use the Processing Toolbox algorithm.

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
| Boosting iterations | Int | Number of trees (default: **200**) | Número de árboles (defecto: **200**) |
| Prediction extent | Extent | Spatial extent to predict | Extensión para predecir |
| Pixel size | Float | Output raster resolution | Resolución del raster de salida |
| Use cross-validation | Boolean | Find optimal n_iter via CV | Encontrar n_iter óptimo via VC |
| CV folds | Int | Number of CV folds (default: 4) | Número de folds (defecto: 4) |

#### Output / Salida

- **Prediction raster** (`.tif`): GeoTIFF with predicted values, LZW compressed, tiled.
- **RMSE**: Training RMSE or CV RMSE if cross-validation is enabled.

> **Note on cross-validation:** When `Use cross-validation = True`, the algorithm uses `gpb.cv()` with `early_stopping_rounds=10` to find the optimal number of iterations, then retrains the final model with that optimal value. The reported RMSE corresponds to CV performance.

---

## 🧪 Testing with Sample Data / Prueba con Datos de Ejemplo

A test dataset is provided at / Se provee un dataset de prueba en:

```
test_plugin/test_maiz_meta_gpboost.kml
```

This dataset simulates **maize yield observations** in the **Meta department, Colombia**:

| Field / Campo | Description (EN) | Descripción (ES) | Units |
|---|---|---|---|
| `rendimiento` | Maize grain yield | Rendimiento de grano de maíz | t/ha |
| `ndvi` | Normalized Difference Vegetation Index | NDVI | 0 – 1 |
| `evi` | Enhanced Vegetation Index | Índice de Vegetación Mejorado | 0 – 1 |
| `elevacion` | Elevation above sea level | Elevación sobre el nivel del mar | m |
| `temp_media` | Mean air temperature | Temperatura media del aire | °C |

**Spatial coverage:** Meta, Colombia (lon: -73.5° to -71.5°, lat: 2.5° to 5.5°) | **N:** 150 points | **CRS:** EPSG:4326

### Quick test configuration / Configuración de prueba rápida

```
📊 Datos:
  ├── Capa de puntos    →  test_maiz_meta_gpboost
  ├── Variable resp.(y) →  rendimiento
  ├── Covariables (X)   →  ndvi, evi  (Ctrl+click)
  └── Normalizar        →  ✓ (recommended)

⚙️ Modelo:
  ├── Función GP        →  exponential
  ├── Learning rate     →  0.01
  ├── Num. hojas        →  31
  ├── Profundidad       →  3
  └── Iteraciones       →  50
```

**Expected results / Resultados esperados:**

```
RMSE            : ~0.15 – 0.25 t/ha
R²              : ~0.70 – 0.90
gp_variance     : > 0.001  (spatial signal present)
gp_range        : 0.05 – 2.0  (decimal degrees)
```

### Quick test from Python Console / Prueba rápida desde la consola Python

```python
import sys, site
for p in site.getsitepackages():
    if p not in sys.path: sys.path.insert(0, p)

import gpboost as gpb
import numpy as np

np.random.seed(42)
n = 100
coords = np.column_stack([
    np.random.uniform(-73.5, -71.5, n),
    np.random.uniform(2.5, 5.5, n)
])
X = np.random.uniform(0.3, 0.9, (n, 2))
y = np.sin(3 * np.pi * X[:,0]) + np.random.normal(0, 0.2, n)

gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential", likelihood="gaussian")
data_train = gpb.Dataset(data=X, label=y)
bst = gpb.train(
    params={"learning_rate": 0.01, "max_depth": 3, "num_leaves": 31, "verbose": -1},
    train_set=data_train,
    gp_model=gp_model,
    num_boost_round=50
)
gp_model.summary()
print("✅ GPBoost works correctly!")
```

---

## 📊 Parameters Reference / Referencia de Parámetros

### Boosting Parameters / Parámetros de Boosting

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `learning_rate` | 0.01 | 0.001 – 1.0 | Lower = slower but more precise |
| `num_leaves` | 31 | 2 – 256 | Higher = more complex trees, risk of overfitting |
| `max_depth` | 3 | -1 – 20 | Limits tree depth; -1 = unlimited |
| `n_iter` | 50 (dialog) / **200 (toolbox)** | 10 – 5000 | Number of trees in ensemble |

### Recommendations by use case / Recomendaciones por caso de uso

| Use Case | Learning Rate | Leaves | Depth | Iterations |
|---|---|---|---|---|
| Small dataset (n < 100) | 0.05 | 15 | 2 | 50 |
| Medium dataset (100 – 500) | 0.01 | 31 | 3 | 100 – 200 |
| Large dataset (> 500) | 0.005 | 63 | 4 | 200 – 500 |
| Many covariates (> 10) | 0.01 | 63 | 5 | 200 |
| With normalization enabled | 0.01 | 31 | 3 | 100 – 200 |

---

## 🔧 Troubleshooting / Solución de Problemas

### Problem: Plugin does not appear in the Plugins menu

**Cause:** Plugin folder name must match the `name` in `metadata.txt` → `GPBoost Spatial Predictor`.

```bash
ls ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
# Must contain metadata.txt
```

---

### Problem: `ModuleNotFoundError: No module named 'gpboost'`

**Solution:** Run in QGIS Python Console:

```python
import sys, site
for p in site.getsitepackages():
    if p not in sys.path: sys.path.insert(0, p)
sys.path.insert(0, site.getusersitepackages())
import gpboost
print(gpboost.__version__)
```

This is **permanently fixed** in `gpboost_plugin.py` via the `_fix_sys_path()` function that runs automatically when the plugin loads.

---

### Problem: Combo boxes (layer, fields) appear empty

```python
from qgis.utils import reloadPlugin
reloadPlugin('gpboost_spatial_predictor_V2')
```

---

### Problem: Training error — not enough points

The plugin requires **a minimum of 10 valid points** (non-null geometry + non-null target + non-null covariates).

```python
layer = iface.activeLayer()
nulls = sum(1 for f in layer.getFeatures() if f["rendimiento"] is None)
print(f"Null values in target field: {nulls}")
```

---

### Problem: All experiments in Comparison tab fail

**Common causes:**
1. Mistyped numeric values in the experiment table (verify with the row number shown in the error)
2. Covariance function name not matching the internal map (use the dropdown labels exactly)
3. Dataset too small for the number of CV folds (increase point count or reduce folds)

---

### Problem: Grid prediction raster is too large

The maximum grid size is **2,000,000 pixels** (n_cols × n_rows). For a 1° × 1° extent in EPSG:4326, use pixel size ≥ 0.001 (≈ 111 m).

---

## 📚 References / Referencias

1. Sigrist, F. (2022). **Gaussian Process Boosting**. *Journal of Machine Learning Research (JMLR)*, 23(232), 1–51. https://www.jmlr.org/papers/v23/20-1262.html

2. Sigrist, F. (2023). **Latent Gaussian Model Boosting**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. https://doi.org/10.1109/TPAMI.2022.3168152

3. GPBoost Official Documentation. https://gpboost.readthedocs.io

4. GPBoost GitHub Repository. https://github.com/fabsig/GPBoost

5. Sigrist, F. (2021). Tree-Boosting for Spatial Data. *Towards Data Science*. https://medium.com/data-science/tree-boosting-for-spatial-data-789145d6d97d

---

## 🆕 Changelog / Historial de cambios

### v1.0.0 — V2 (2026)
- **New tab: 🔬 Comparar** — multi-model comparison and hyperparameter tuning table with preset experiments
- **New metric: R²** — coefficient of determination reported alongside RMSE in all outputs
- **Variable normalization** — optional standardization of y and X before training, with automatic back-transformation
- **Extended GP covariance functions** — added `matern (v=0.5)`, `matern (v=1.5)`, `matern (v=2.5)`, and `wendland` with explicit `cov_fct_shape` mapping
- **Improved cross-validation** — `early_stopping_rounds=10` in Processing Toolbox for more robust optimal iteration search
- **Default iterations increased** — Processing Toolbox default changed from 50 → 200
- **Best model highlighting** — tuning results table highlights best model in green, failed runs in red

---

## 👥 Authors / Autores

**Santiago Gallego** & **Jesús Enrique Flores Riera**  
GPBoost QGIS Spatial Predictor Plugin, 2026  
Repository: https://github.com/jf-floresriera/GPBoost-Spatial-Predictor

---

## 📄 License / Licencia

MIT License — Free to use, modify and distribute with attribution.  
MIT License — Libre para usar, modificar y distribuir con atribución.
