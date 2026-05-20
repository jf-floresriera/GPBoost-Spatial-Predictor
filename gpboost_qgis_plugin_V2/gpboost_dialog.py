"""
GPBoost Dialog v3.1 - Comparacion de modelos + Tuning + Normalizacion
Sigrist F. (2022). GPBoost Algorithm. JMLR, 23(104), 1-49.
Flores-Riera J.E. & Gallegos (2026).
"""
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QAbstractItemView,
    QPushButton, QProgressBar, QTextEdit, QTabWidget,
    QWidget, QHBoxLayout, QMessageBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont, QColor
from qgis.core import QgsProject, QgsWkbTypes, QgsVectorLayer
import traceback


# =============================================================================
#  MAPA DE FUNCIONES DE COVARIANZA
#  Etiqueta UI  ->  (cov_function,  cov_fct_shape)
#  Referencia: gpboost.readthedocs.io/en/latest/Main_parameters.html
# =============================================================================
COV_FN_MAP = {
    "exponential":         ("exponential",         0.0),
    "gaussian":            ("gaussian",            0.0),
    "matern (v=0.5)":      ("matern",              0.5),
    "matern (v=1.5)":      ("matern",              1.5),
    "matern (v=2.5)":      ("matern",              2.5),
    "powered_exponential": ("powered_exponential", 1.0),
    "wendland":            ("wendland",            0.0),
}
COV_FN_LABELS = list(COV_FN_MAP.keys())


def _parse_cov_fn(label):
    """Convierte etiqueta del combo al par (cov_function, cov_fct_shape)."""
    return COV_FN_MAP.get(label, ("exponential", 0.0))


# =============================================================================
#  WORKER: Entrenamiento de UN modelo
# =============================================================================
class GPBoostWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            import sys, site
            for p in site.getsitepackages():
                if p not in sys.path:
                    sys.path.insert(0, p)
            sys.path.insert(0, site.getusersitepackages())
            import gpboost as gpb
            import numpy as np

            cfg = self.config
            coords, y_list, X_list = [], [], []

            self.progress.emit(10, "Extrayendo datos de la capa...")
            for feat in cfg["layer"].getFeatures():
                geom = feat.geometry()
                if geom.isNull():
                    continue
                pt = feat.geometry().asPoint()
                y_val = feat[cfg["target_field"]]
                if y_val is None or y_val == "":
                    continue
                row = []
                valid = True
                for cf in cfg["cov_fields"]:
                    val = feat[cf]
                    if val is None or val == "":
                        valid = False
                        break
                    row.append(float(val))
                if not valid:
                    continue
                coords.append([pt.x(), pt.y()])
                y_list.append(float(y_val))
                X_list.append(row)

            if len(y_list) < 10:
                self.error.emit(
                    "Puntos insuficientes: {}. Minimo 10.".format(len(y_list))
                )
                return

            coords_np = np.array(coords, dtype=np.float64)
            y_np      = np.array(y_list, dtype=np.float64)
            if X_list and X_list[0]:
                X_np = np.array(X_list, dtype=np.float64)
            else:
                X_np = coords_np.copy()
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)

            # Normalizacion opcional
            y_mean, y_std = 0.0, 1.0
            if cfg.get("normalize", False):
                self.progress.emit(15, "Normalizando variables...")
                y_mean = float(y_np.mean())
                y_std  = float(y_np.std()) if y_np.std() > 1e-10 else 1.0
                y_np   = (y_np - y_mean) / y_std
                X_mean = X_np.mean(axis=0)
                X_std  = np.where(X_np.std(axis=0) > 1e-10,
                                  X_np.std(axis=0), 1.0)
                X_np   = (X_np - X_mean) / X_std

            cov_fn    = cfg["cov_function"]
            cov_shape = cfg.get("cov_fct_shape", 0.0)

            self.progress.emit(
                25, "{} puntos. GP: {}...".format(len(y_np), cfg["cov_fn_label"])
            )
            gp_model = gpb.GPModel(
                gp_coords=coords_np,
                cov_function=cov_fn,
                cov_fct_shape=cov_shape,
                likelihood="gaussian"
            )
            params = {
                "learning_rate": cfg["learning_rate"],
                "num_leaves":    cfg["num_leaves"],
                "max_depth":     cfg["max_depth"],
                "verbose":       -1,
            }
            data_train = gpb.Dataset(data=X_np, label=y_np)
            self.progress.emit(
                40, "Entrenando {} iteraciones...".format(cfg["n_iter"])
            )
            booster = gpb.train(
                params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=cfg["n_iter"]
            )
            self.progress.emit(80, "Calculando metricas...")
            pred_result = booster.predict(
                data=X_np,
                gp_coords_pred=coords_np,
                predict_var=False,
                pred_latent=False
            )
            if isinstance(pred_result, dict):
                pv = pred_result.get(
                    "response_mean", list(pred_result.values())[0]
                )
            else:
                pv = pred_result
            pv = np.array(pv).flatten()

            if cfg.get("normalize", False):
                pv_orig = pv * y_std + y_mean
                y_orig  = np.array(y_list, dtype=np.float64)
            else:
                pv_orig = pv
                y_orig  = y_np

            rmse   = float(np.sqrt(np.mean((pv_orig - y_orig) ** 2)))
            ss_res = np.sum((y_orig - pv_orig) ** 2)
            ss_tot = np.sum((y_orig - y_orig.mean()) ** 2)
            r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            cov_params = {}
            try:
                cp = np.array(gp_model.get_cov_pars()).flatten()
                keys = ["error_variance", "gp_variance", "gp_range"]
                for k, v in zip(keys, cp):
                    cov_params[k] = float(v)
            except Exception:
                pass

            self.progress.emit(100, "Completo!")
            self.finished.emit({
                "rmse":            rmse,
                "r2":              r2,
                "n_train":         len(y_np),
                "cov_params":      cov_params,
                "gpboost_version": gpb.__version__,
                "config":          cfg
            })
        except Exception:
            self.error.emit(traceback.format_exc())


# =============================================================================
#  WORKER: Comparacion / Tuning de multiples modelos
# =============================================================================
class GPBoostTuningWorker(QThread):
    progress  = pyqtSignal(int, str)
    row_ready = pyqtSignal(dict)
    finished  = pyqtSignal(list)
    error     = pyqtSignal(str)

    def __init__(self, base_config, experiments):
        super().__init__()
        self.base_config = base_config
        self.experiments = experiments

    def run(self):
        try:
            import sys, site
            for p in site.getsitepackages():
                if p not in sys.path:
                    sys.path.insert(0, p)
            sys.path.insert(0, site.getusersitepackages())
            import gpboost as gpb
            import numpy as np

            cfg = self.base_config
            coords, y_list, X_list = [], [], []

            self.progress.emit(5, "Extrayendo datos...")
            for feat in cfg["layer"].getFeatures():
                geom = feat.geometry()
                if geom.isNull():
                    continue
                pt = feat.geometry().asPoint()
                y_val = feat[cfg["target_field"]]
                if y_val is None or y_val == "":
                    continue
                row = []
                valid = True
                for cf in cfg["cov_fields"]:
                    val = feat[cf]
                    if val is None or val == "":
                        valid = False
                        break
                    row.append(float(val))
                if not valid:
                    continue
                coords.append([pt.x(), pt.y()])
                y_list.append(float(y_val))
                X_list.append(row)

            if len(y_list) < 10:
                self.error.emit(
                    "Puntos insuficientes: {}.".format(len(y_list))
                )
                return

            coords_np = np.array(coords, dtype=np.float64)
            y_np      = np.array(y_list, dtype=np.float64)
            if X_list and X_list[0]:
                X_np = np.array(X_list, dtype=np.float64)
            else:
                X_np = coords_np.copy()
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)

            y_orig_np = y_np.copy()
            X_np_use  = X_np.copy()
            y_mean, y_std = 0.0, 1.0

            if cfg.get("normalize", False):
                y_mean   = float(y_np.mean())
                y_std    = float(y_np.std()) if y_np.std() > 1e-10 else 1.0
                y_np     = (y_np - y_mean) / y_std
                X_mean   = X_np.mean(axis=0)
                X_std    = np.where(X_np.std(axis=0) > 1e-10,
                                    X_np.std(axis=0), 1.0)
                X_np_use = (X_np - X_mean) / X_std

            results = []
            n_exp   = len(self.experiments)

            for i, exp in enumerate(self.experiments):
                pct   = int(10 + (i / n_exp) * 85)
                label = exp.get("label", "Modelo {}".format(i + 1))
                self.progress.emit(
                    pct,
                    "Entrenando {} ({}/{})...".format(label, i + 1, n_exp)
                )

                lr     = exp.get("learning_rate", cfg["learning_rate"])
                leaves = exp.get("num_leaves",    cfg["num_leaves"])
                depth  = exp.get("max_depth",     cfg["max_depth"])
                n_iter = exp.get("n_iter",         cfg["n_iter"])
                cov_fn = exp.get("cov_function",  cfg["cov_function"])
                cov_sh = float(exp.get("cov_fct_shape",
                                       cfg.get("cov_fct_shape", 0.0)))

                try:
                    gp = gpb.GPModel(
                        gp_coords=coords_np,
                        cov_function=cov_fn,
                        cov_fct_shape=cov_sh,
                        likelihood="gaussian"
                    )
                    ds = gpb.Dataset(data=X_np_use, label=y_np)
                    bst = gpb.train(
                        params={
                            "learning_rate": lr,
                            "num_leaves":    leaves,
                            "max_depth":     depth,
                            "verbose":       -1
                        },
                        train_set=ds,
                        gp_model=gp,
                        num_boost_round=n_iter
                    )
                    pred = bst.predict(
                        data=X_np_use,
                        gp_coords_pred=coords_np,
                        predict_var=False,
                        pred_latent=False
                    )
                    if isinstance(pred, dict):
                        pv = pred.get("response_mean",
                                      list(pred.values())[0])
                    else:
                        pv = pred
                    pv = np.array(pv).flatten()

                    if cfg.get("normalize", False):
                        pv_orig = pv * y_std + y_mean
                    else:
                        pv_orig   = pv
                        y_orig_np = y_np

                    rmse   = float(np.sqrt(np.mean((pv_orig - y_orig_np) ** 2)))
                    ss_res = np.sum((y_orig_np - pv_orig) ** 2)
                    ss_tot = np.sum((y_orig_np - y_orig_np.mean()) ** 2)
                    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                    cp = {}
                    try:
                        cp_flat = np.array(gp.get_cov_pars()).flatten()
                        for k, v in zip(["err_var", "gp_var", "rango"], cp_flat):
                            cp[k] = float(v)
                    except Exception:
                        pass

                    cov_label = cov_fn
                    if cov_sh > 0:
                        cov_label = "{}(v={})".format(cov_fn, cov_sh)

                    row = {
                        "label":  label,
                        "lr":     lr,
                        "leaves": leaves,
                        "depth":  depth,
                        "n_iter": n_iter,
                        "cov_fn": cov_label,
                        "rmse":   rmse,
                        "r2":     r2,
                        "ok":     True,
                        **cp
                    }
                except Exception as e:
                    row = {
                        "label":   label,
                        "lr":      lr,
                        "leaves":  leaves,
                        "depth":   depth,
                        "n_iter":  n_iter,
                        "cov_fn":  cov_fn,
                        "rmse":    float("inf"),
                        "r2":      0.0,
                        "err_var": 0.0,
                        "gp_var":  0.0,
                        "rango":   0.0,
                        "ok":      False,
                        "error":   str(e)
                    }

                results.append(row)
                self.row_ready.emit(row)

            self.progress.emit(
                100,
                "Tuning completo! {} modelos evaluados.".format(n_exp)
            )
            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


# =============================================================================
#  DIALOGO PRINCIPAL
# =============================================================================
class GPBoostDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface  = iface
        self.worker = None
        self.setWindowTitle("GPBoost Spatial Predictor v3.1")
        self.setMinimumWidth(600)
        self.setMinimumHeight(640)
        self._build_ui()

    # -------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("GPBoost Spatial Predictor")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel(
            "Tree-Boosting + Gaussian Process  |  "
            "Flores-Riera & Gallegos  |  Sigrist (2022, JMLR)"
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#666; font-size:9px;")
        layout.addWidget(sub)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_tab_datos()
        self._build_tab_modelo()
        self._build_tab_comparar()
        self._build_tab_resultados()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel(
            "Listo. Configura los parametros y presiona Ejecutar."
        )
        self.status_lbl.setStyleSheet("color:#555; font-size:10px;")
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Ejecutar modelo unico")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;padding:7px 14px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#1b5e20;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.run_btn.clicked.connect(self._run_single)
        btn_row.addWidget(self.run_btn)

        self.tune_btn = QPushButton("Comparar / Tuning")
        self.tune_btn.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;padding:7px 14px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#0d47a1;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.tune_btn.clicked.connect(self._run_tuning)
        btn_row.addWidget(self.tune_btn)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)
        self._on_layer_changed(0)

    # -------------------------------------------------------------------------
    def _build_tab_datos(self):
        w  = QWidget()
        dl = QFormLayout(w)
        dl.setSpacing(8)

        self.layer_combo = QComboBox()
        self._populate_layers()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        dl.addRow("Capa de puntos:", self.layer_combo)

        self.target_combo = QComboBox()
        dl.addRow("Variable respuesta (y):", self.target_combo)

        self.cov_list = QListWidget()
        self.cov_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cov_list.setMaximumHeight(100)
        dl.addRow("Covariables (X):", self.cov_list)

        self.normalize_chk = QCheckBox(
            "Normalizar variables (media=0, std=1) — recomendado para tuning"
        )
        self.normalize_chk.setChecked(False)
        dl.addRow("", self.normalize_chk)

        hint = QLabel(
            "Ctrl+clic para seleccion multiple. "
            "Sin seleccion -> usa coordenadas como X."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:9px;")
        dl.addRow("", hint)

        self.tabs.addTab(w, "Datos")

    # -------------------------------------------------------------------------
    def _build_tab_modelo(self):
        w  = QWidget()
        ml = QFormLayout(w)
        ml.setSpacing(8)

        self.cov_fn_combo = QComboBox()
        self.cov_fn_combo.addItems(COV_FN_LABELS)
        ml.addRow("Funcion de covarianza GP:", self.cov_fn_combo)

        cov_note = QLabel(
            "exponential = Matern v=0.5 (menos suave) | "
            "gaussian = RBF (infinitamente suave)\n"
            "matern (v=1.5) = una vez diferenciable | "
            "matern (v=2.5) = dos veces diferenciable"
        )
        cov_note.setWordWrap(True)
        cov_note.setStyleSheet("color:#777; font-size:9px;")
        ml.addRow("", cov_note)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(3)
        self.lr_spin.setSingleStep(0.005)
        ml.addRow("Learning rate:", self.lr_spin)

        self.leaves_spin = QSpinBox()
        self.leaves_spin.setRange(2, 512)
        self.leaves_spin.setValue(31)
        ml.addRow("Num. hojas:", self.leaves_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(-1, 20)
        self.depth_spin.setValue(3)
        ml.addRow("Profundidad maxima (-1=sin limite):", self.depth_spin)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 2000)
        self.iter_spin.setValue(50)
        ml.addRow("Iteraciones boosting:", self.iter_spin)

        self.tabs.addTab(w, "Modelo")

    # -------------------------------------------------------------------------
    def _build_tab_comparar(self):
        w  = QWidget()
        vl = QVBoxLayout(w)

        info = QLabel(
            "Define los experimentos a comparar. "
            "Cada fila es un modelo con diferentes hiperparametros.\n"
            "La columna cov_fct_shape es el parametro de suavidad Matern "
            "(0.5 / 1.5 / 2.5). Usa 0.0 para exponential/gaussian/wendland."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#444; font-size:10px;")
        vl.addWidget(info)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets rapidos:"))
        for label, fn in [
            ("Sensibilidad lr",   self._preset_lr),
            ("Sensibilidad iter", self._preset_iter),
            ("Funciones GP",      self._preset_cov),
            ("Grid completo",     self._preset_grid),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet("padding:4px 8px; font-size:10px;")
            btn.clicked.connect(fn)
            preset_row.addWidget(btn)
        preset_row.addStretch()
        vl.addLayout(preset_row)

        self.exp_table = QTableWidget(0, 7)
        self.exp_table.setHorizontalHeaderLabels([
            "Nombre", "learning_rate", "num_leaves",
            "max_depth", "n_iter", "cov_function", "cov_fct_shape"
        ])
        self.exp_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.exp_table.setMinimumHeight(160)
        vl.addWidget(self.exp_table)

        btn_row = QHBoxLayout()
        for label, fn in [
            ("+ Agregar fila",  self._add_exp_row),
            ("- Eliminar fila", self._del_exp_row),
            ("Limpiar todo",    lambda: self.exp_table.setRowCount(0)),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet("padding:4px 8px; font-size:10px;")
            btn.clicked.connect(fn)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        vl.addLayout(btn_row)

        vl.addWidget(QLabel("Resultados del tuning (verde = mejor RMSE):"))
        self.result_table = QTableWidget(0, 8)
        self.result_table.setHorizontalHeaderLabels([
            "Modelo", "lr", "iter", "cov_fn",
            "RMSE", "R2", "gp_var", "rango"
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.result_table.setMinimumHeight(130)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        vl.addWidget(self.result_table)

        self.tabs.addTab(w, "Comparar")

    # -------------------------------------------------------------------------
    def _build_tab_resultados(self):
        w  = QWidget()
        rl = QVBoxLayout(w)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Monospace", 9))
        self.results_text.setPlaceholderText(
            "Los resultados apareceran aqui tras ejecutar el modelo unico..."
        )
        self.results_text.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; border-radius:4px;"
        )
        rl.addWidget(self.results_text)
        self.tabs.addTab(w, "Resultados")

    # -------------------------------------------------------------------------
    def _populate_layers(self):
        self.layer_combo.clear()
        for _, layer in QgsProject.instance().mapLayers().items():
            if (isinstance(layer, QgsVectorLayer) and
                    layer.geometryType() == QgsWkbTypes.PointGeometry):
                self.layer_combo.addItem(layer.name(), layer)

    def _on_layer_changed(self, idx):
        layer = self.layer_combo.itemData(idx)
        self.target_combo.clear()
        self.cov_list.clear()
        if layer:
            for field in layer.fields():
                if field.isNumeric():
                    self.target_combo.addItem(field.name())
                    self.cov_list.addItem(field.name())

    # -------------------------------------------------------------------------
    def _preset_lr(self):
        self.exp_table.setRowCount(0)
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
            self._add_row_data(
                "lr={}".format(lr), lr, 31, 3, 50, "exponential", 0.0
            )

    def _preset_iter(self):
        self.exp_table.setRowCount(0)
        for it in [25, 50, 100, 200, 500]:
            self._add_row_data(
                "iter={}".format(it), 0.01, 31, 3, it, "exponential", 0.0
            )

    def _preset_cov(self):
        self.exp_table.setRowCount(0)
        configs = [
            ("exponential",   "exponential",         0.0),
            ("gaussian",      "gaussian",            0.0),
            ("matern v=0.5",  "matern",              0.5),
            ("matern v=1.5",  "matern",              1.5),
            ("matern v=2.5",  "matern",              2.5),
            ("powered_exp",   "powered_exponential", 1.0),
        ]
        for label, cov_fn, cov_sh in configs:
            self._add_row_data(label, 0.01, 31, 3, 50, cov_fn, cov_sh)

    def _preset_grid(self):
        self.exp_table.setRowCount(0)
        for lr in [0.01, 0.05]:
            for it in [50, 100, 200]:
                for cov_fn, cov_sh in [("exponential", 0.0), ("matern", 1.5)]:
                    label = "lr={} it={} {}".format(lr, it, cov_fn)
                    self._add_row_data(label, lr, 31, 3, it, cov_fn, cov_sh)

    def _add_row_data(self, name, lr, leaves, depth, n_iter, cov_fn, cov_sh=0.0):
        r = self.exp_table.rowCount()
        self.exp_table.insertRow(r)
        for c, val in enumerate(
            [name, lr, leaves, depth, n_iter, cov_fn, cov_sh]
        ):
            self.exp_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _add_exp_row(self):
        cov_fn, cov_sh = _parse_cov_fn(self.cov_fn_combo.currentText())
        self._add_row_data(
            "Modelo {}".format(self.exp_table.rowCount() + 1),
            self.lr_spin.value(),
            self.leaves_spin.value(),
            self.depth_spin.value(),
            self.iter_spin.value(),
            cov_fn,
            cov_sh
        )

    def _del_exp_row(self):
        row = self.exp_table.currentRow()
        if row >= 0:
            self.exp_table.removeRow(row)

    # -------------------------------------------------------------------------
    def _base_config(self):
        layer = self.layer_combo.currentData()
        if not layer:
            QMessageBox.warning(
                self, "Sin capa", "Selecciona una capa de puntos."
            )
            return None
        target = self.target_combo.currentText()
        if not target:
            QMessageBox.warning(
                self, "Sin campo", "Selecciona la variable respuesta."
            )
            return None
        cov_fields = [
            self.cov_list.item(i).text()
            for i in range(self.cov_list.count())
            if self.cov_list.item(i).isSelected()
        ]
        cov_label = self.cov_fn_combo.currentText()
        cov_fn, cov_sh = _parse_cov_fn(cov_label)
        return {
            "layer":         layer,
            "target_field":  target,
            "cov_fields":    cov_fields,
            "cov_fn_label":  cov_label,
            "cov_function":  cov_fn,
            "cov_fct_shape": cov_sh,
            "learning_rate": self.lr_spin.value(),
            "num_leaves":    self.leaves_spin.value(),
            "max_depth":     self.depth_spin.value(),
            "n_iter":        self.iter_spin.value(),
            "normalize":     self.normalize_chk.isChecked(),
        }

    # -------------------------------------------------------------------------
    def _run_single(self):
        cfg = self._base_config()
        if not cfg:
            return
        self._set_running(True)
        self.results_text.clear()
        self.worker = GPBoostWorker(cfg)
        self.worker.progress.connect(
            lambda v, m: (
                self.progress_bar.setValue(v),
                self.status_lbl.setText(m)
            )
        )
        self.worker.finished.connect(self._on_single_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_single_finished(self, r):
        self._set_running(False)
        norm_str = "Si" if r["config"].get("normalize") else "No"
        lines = [
            "=" * 50,
            "   GPBoost -- Resultado del modelo unico",
            "=" * 50,
            "  GPBoost version   : {}".format(r.get("gpboost_version", "?")),
            "  Muestras          : {}".format(r["n_train"]),
            "  Normalizado       : {}".format(norm_str),
            "  Funcion GP        : {}".format(
                r["config"].get("cov_fn_label", "?")
            ),
            "  Learning rate     : {}".format(r["config"]["learning_rate"]),
            "  Iteraciones       : {}".format(r["config"]["n_iter"]),
            "",
            "  RMSE              : {:.4f}".format(r["rmse"]),
            "  R2                : {:.4f}".format(r["r2"]),
            "",
            "  Parametros GP estimados:",
        ]
        for k, v in r.get("cov_params", {}).items():
            lines.append("    {:<22}: {:.6f}".format(k, v))
        lines += [
            "",
            "  Para comparar modelos -> pestana Comparar",
            "=" * 50,
        ]
        self.results_text.setPlainText("\n".join(lines))
        self.tabs.setCurrentIndex(3)

    # -------------------------------------------------------------------------
    def _run_tuning(self):
        cfg = self._base_config()
        if not cfg:
            return
        if self.exp_table.rowCount() == 0:
            QMessageBox.warning(
                self,
                "Sin experimentos",
                "Agrega experimentos en la pestana Comparar o usa un Preset."
            )
            return

        experiments = []
        for r in range(self.exp_table.rowCount()):
            def cell(c, row=r):
                item = self.exp_table.item(row, c)
                return item.text() if item else ""
            try:
                experiments.append({
                    "label":         cell(0),
                    "learning_rate": float(cell(1)),
                    "num_leaves":    int(cell(2)),
                    "max_depth":     int(cell(3)),
                    "n_iter":        int(cell(4)),
                    "cov_function":  cell(5),
                    "cov_fct_shape": float(cell(6)) if cell(6) else 0.0,
                })
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Error en tabla",
                    "Verifica los valores numericos en la fila {}.".format(
                        r + 1
                    )
                )
                return

        self._set_running(True)
        self.result_table.setRowCount(0)

        self.worker = GPBoostTuningWorker(cfg, experiments)
        self.worker.progress.connect(
            lambda v, m: (
                self.progress_bar.setValue(v),
                self.status_lbl.setText(m)
            )
        )
        self.worker.row_ready.connect(self._add_tuning_row)
        self.worker.finished.connect(self._on_tuning_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _add_tuning_row(self, row):
        r = self.result_table.rowCount()
        self.result_table.insertRow(r)
        if row.get("ok", True) and row["rmse"] != float("inf"):
            rmse_str = "{:.4f}".format(row["rmse"])
        else:
            rmse_str = "ERROR"
        vals = [
            row["label"],
            "{:.3f}".format(row["lr"]),
            str(row["n_iter"]),
            row["cov_fn"],
            rmse_str,
            "{:.4f}".format(row["r2"]),
            "{:.4f}".format(row.get("gp_var", 0.0)),
            "{:.4f}".format(row.get("rango",  0.0)),
        ]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignCenter)
            if not row.get("ok", True):
                item.setBackground(QColor("#b71c1c"))
                item.setForeground(QColor("white"))
            self.result_table.setItem(r, c, item)

    def _on_tuning_finished(self, results):
        self._set_running(False)
        valid = [
            r for r in results
            if r.get("ok", True) and r["rmse"] != float("inf")
        ]
        if not valid:
            self.status_lbl.setText(
                "Todos los modelos fallaron. Revisa los parametros."
            )
            return

        best_rmse = min(r["rmse"] for r in valid)
        for row_idx in range(self.result_table.rowCount()):
            item = self.result_table.item(row_idx, 4)
            if item and item.text() == "{:.4f}".format(best_rmse):
                for c in range(self.result_table.columnCount()):
                    it = self.result_table.item(row_idx, c)
                    if it:
                        it.setBackground(QColor("#1b5e20"))
                        it.setForeground(QColor("white"))

        best = next(r for r in valid if r["rmse"] == best_rmse)
        self.status_lbl.setText(
            "Mejor modelo: '{}' | RMSE={:.4f} | R2={:.4f}".format(
                best["label"], best["rmse"], best["r2"]
            )
        )
        self.tabs.setCurrentIndex(2)

    # -------------------------------------------------------------------------
    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.tune_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if not running:
            self.progress_bar.setValue(100)

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self._set_running(False)
        self.status_lbl.setText("Cancelado por el usuario.")

    def _on_error(self, msg):
        self._set_running(False)
        self.results_text.setPlainText("ERROR:\n\n{}".format(msg))
        self.status_lbl.setText("Error durante el entrenamiento.")
        QMessageBox.critical(self, "Error GPBoost", msg[:800])
