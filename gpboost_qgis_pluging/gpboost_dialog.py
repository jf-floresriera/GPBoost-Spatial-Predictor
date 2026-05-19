"""
GPBoost Dialog - Interfaz Qt interactiva para el plugin QGIS.
Ejecuta el entrenamiento en un hilo secundario para no bloquear QGIS.
"""
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QAbstractItemView,
    QPushButton, QProgressBar, QTextEdit, QTabWidget,
    QWidget, QHBoxLayout, QMessageBox
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont
from qgis.core import QgsProject, QgsWkbTypes, QgsVectorLayer
import traceback


class GPBoostWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            import gpboost as gpb
            import numpy as np

            cfg = self.config
            layer = cfg["layer"]
            coords, y_list, X_list = [], [], []

            self.progress.emit(10, "Extrayendo datos de la capa...")

            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom.isNull():
                    continue
                pt = geom.asPoint()
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
                if row:
                    X_list.append(row)

            coords_np = np.array(coords)
            y_np = np.array(y_list)
            X_np = np.array(X_list) if X_list else None

            self.progress.emit(25, f"{len(y_np)} puntos cargados. Entrenando...")

            gp_model = gpb.GPModel(gp_coords=coords_np, cov_function=cfg["cov_function"])
            params = {
                "learning_rate": cfg["learning_rate"],
                "num_leaves": cfg["num_leaves"],
                "max_depth": cfg["max_depth"],
                "verbose": -1,
            }
            dataset = gpb.Dataset(data=X_np, label=y_np)
            booster = gpb.train(
                params=params, train_set=dataset,
                gp_model=gp_model, num_boost_round=cfg["n_iter"])

            self.progress.emit(80, "Calculando RMSE...")

            pred_tr = booster.predict(data=X_np, gp_coords_pred=coords_np, predict_var=False)
            pv = pred_tr["response_mean"] if isinstance(pred_tr, dict) else pred_tr
            rmse = float(np.sqrt(np.mean((pv - y_np) ** 2)))

            cov_params = {}
            try:
                cp = gp_model.get_cov_pars()
                for k, v in zip(["error_variance", "gp_variance", "gp_range"], cp):
                    cov_params[k] = float(v)
            except Exception:
                pass

            self.progress.emit(100, "¡Entrenamiento completo!")
            self.finished.emit({
                "rmse": rmse, "n_train": len(y_np), "cov_params": cov_params
            })

        except Exception:
            self.error.emit(traceback.format_exc())


class GPBoostDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("GPBoost Spatial Predictor")
        self.setMinimumWidth(500)
        self.setMinimumHeight(560)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("🌿 GPBoost Spatial Predictor")
        f = QFont(); f.setPointSize(13); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Tree-Boosting + Gaussian Process  |  Sigrist (2022, JMLR)")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#666;font-size:10px;")
        layout.addWidget(sub)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab Datos
        dt = QWidget(); dl = QFormLayout(dt); dl.setSpacing(8)
        self.layer_combo = QComboBox()
        self._populate_layers()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        dl.addRow("Capa de puntos:", self.layer_combo)

        self.target_combo = QComboBox()
        dl.addRow("Variable respuesta (y):", self.target_combo)

        self.cov_list = QListWidget()
        self.cov_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cov_list.setMaximumHeight(110)
        dl.addRow("Covariables (X):", self.cov_list)

        hint = QLabel("Ctrl+clic para selección múltiple. Sin selección = solo coordenadas.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#888;font-size:9px;")
        dl.addRow("", hint)
        tabs.addTab(dt, "📊 Datos")

        # Tab Modelo
        mt = QWidget(); ml = QFormLayout(mt); ml.setSpacing(8)

        self.cov_fn_combo = QComboBox()
        self.cov_fn_combo.addItems(["exponential","gaussian","matern32","matern52","powered_exponential"])
        ml.addRow("Función de covarianza GP:", self.cov_fn_combo)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0); self.lr_spin.setValue(0.01); self.lr_spin.setDecimals(3)
        ml.addRow("Learning rate:", self.lr_spin)

        self.leaves_spin = QSpinBox()
        self.leaves_spin.setRange(2, 256); self.leaves_spin.setValue(31)
        ml.addRow("Num. hojas:", self.leaves_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(-1, 20); self.depth_spin.setValue(3)
        ml.addRow("Profundidad máxima:", self.depth_spin)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 5000); self.iter_spin.setValue(200)
        ml.addRow("Iteraciones boosting:", self.iter_spin)

        tabs.addTab(mt, "⚙️ Modelo")

        # Tab Resultados
        rt = QWidget(); rl = QVBoxLayout(rt)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Monospace", 9))
        self.results_text.setPlaceholderText("Los resultados aparecerán aquí...")
        self.results_text.setStyleSheet("background:#1e1e1e;color:#d4d4d4;border-radius:4px;")
        rl.addWidget(self.results_text)
        tabs.addTab(rt, "📈 Resultados")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Listo. Configura los parámetros y presiona Ejecutar.")
        self.status_lbl.setStyleSheet("color:#555;font-size:10px;")
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Ejecutar GPBoost")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;padding:8px 18px;"
            "border-radius:4px;font-weight:bold;font-size:11px;}"
            "QPushButton:hover{background:#1b5e20;}"
            "QPushButton:disabled{background:#aaa;}")
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("✕  Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)
        self._on_layer_changed(0)

    def _populate_layers(self):
        self.layer_combo.clear()
        for _, layer in QgsProject.instance().mapLayers().items():
            if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PointGeometry:
                self.layer_combo.addItem(layer.name(), layer)

    def _on_layer_changed(self, idx):
        layer = self.layer_combo.itemData(idx)
        self.target_combo.clear(); self.cov_list.clear()
        if layer:
            for field in layer.fields():
                if field.isNumeric():
                    self.target_combo.addItem(field.name())
                    self.cov_list.addItem(field.name())

    def _run(self):
        layer = self.layer_combo.currentData()
        if not layer:
            QMessageBox.warning(self, "Sin capa", "Selecciona una capa de puntos.")
            return
        selected_covs = [self.cov_list.item(i).text()
                         for i in range(self.cov_list.count())
                         if self.cov_list.item(i).isSelected()]
        config = {
            "layer": layer,
            "target_field": self.target_combo.currentText(),
            "cov_fields": selected_covs,
            "cov_function": self.cov_fn_combo.currentText(),
            "learning_rate": self.lr_spin.value(),
            "num_leaves": self.leaves_spin.value(),
            "max_depth": self.depth_spin.value(),
            "n_iter": self.iter_spin.value(),
        }
        self.run_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0); self.results_text.clear()
        self.worker = GPBoostWorker(config)
        self.worker.progress.connect(lambda v, m: (self.progress_bar.setValue(v), self.status_lbl.setText(m)))
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.status_lbl.setText("Cancelado.")

    def _on_finished(self, r):
        self.progress_bar.setValue(100)
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        lines = [
            "=" * 44,
            "  Resultados GPBoost",
            "=" * 44,
            f"  Muestras entrenamiento : {r['n_train']}",
            f"  RMSE                   : {r['rmse']:.4f}",
            "",
            "  Parámetros GP:"]
        for k, v in r.get("cov_params", {}).items():
            lines.append(f"    {k:<22}: {v:.6f}")
        lines += ["", "  Usa el Processing Toolbox para", "  exportar el raster de predicción.", "=" * 44]
        self.results_text.setPlainText("\n".join(lines))

    def _on_error(self, msg):
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.results_text.setPlainText(f"ERROR:\n{msg}")
        QMessageBox.critical(self, "Error GPBoost", msg[:500])
