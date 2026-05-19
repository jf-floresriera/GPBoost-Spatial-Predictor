"""
GPBoost Spatial Predictor - QGIS Plugin
Sigrist (2022, JMLR) - Gaussian Process Boosting
"""
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication
import os

from .gpboost_provider import GPBoostProvider
from .gpboost_dialog import GPBoostDialog


class GPBoostPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.provider = None
        self.action = None
        self.dialog = None

    def initProcessing(self):
        self.provider = GPBoostProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()
        icon_path = os.path.join(self.plugin_dir, "icons", "icon.png")
        self.action = QAction(QIcon(icon_path), "GPBoost Spatial Predictor", self.iface.mainWindow())
        self.action.setToolTip("Predict spatial variables using GPBoost")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&GPBoost", self.action)

    def unload(self):
        self.iface.removePluginMenu("&GPBoost", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

    def run(self):
        if not self._check_gpboost():
            return
        if self.dialog is None:
            self.dialog = GPBoostDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _check_gpboost(self):
        try:
            import gpboost  # noqa
            return True
        except ImportError:
            QMessageBox.critical(
                self.iface.mainWindow(), "GPBoost Not Found",
                "Instala gpboost desde la consola Python de QGIS:\n"
                "  import subprocess, sys\n"
                "  subprocess.call([sys.executable, '-m', 'pip', 'install', 'gpboost'])"
            )
            return False
