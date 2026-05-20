"""
GPBoost Spatial Predictor - QGIS Plugin
Sigrist (2022, JMLR) - Gaussian Process Boosting
"""
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication
import os
import sys
import site


def _fix_sys_path():
    """
    Agrega los site-packages del sistema al sys.path de QGIS.
    Necesario en Linux donde QGIS usa /usr/bin/python3 pero
    los paquetes instalados con pip están en dist-packages.
    """
    paths_to_add = []
    try:
        paths_to_add += site.getsitepackages()
    except AttributeError:
        pass
    try:
        paths_to_add.append(site.getusersitepackages())
    except AttributeError:
        pass
    for p in paths_to_add:
        if p and p not in sys.path:
            sys.path.insert(0, p)


# Ejecutar al cargar el módulo (cuando QGIS carga el plugin)
_fix_sys_path()

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
        self.action = QAction(
            QIcon(icon_path),
            "GPBoost Spatial Predictor",
            self.iface.mainWindow()
        )
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
        # Re-aplicar fix de path en cada apertura (por si QGIS reinició)
        _fix_sys_path()

        if not self._check_gpboost():
            return

        if self.dialog is None:
            # Primera apertura: crear el diálogo
            self.dialog = GPBoostDialog(self.iface)
        else:
            # Reaperturas: refrescar la lista de capas disponibles
            self.dialog._populate_layers()

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _check_gpboost(self):
        try:
            import gpboost  # noqa
            return True
        except ImportError:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "GPBoost No Encontrado",
                "La librería 'gpboost' no está disponible para QGIS.\n\n"
                "Solución — ejecuta en la consola Python de QGIS:\n\n"
                "  import subprocess, sys, site\n"
                "  for p in site.getsitepackages():\n"
                "      if p not in sys.path: sys.path.insert(0, p)\n"
                "  import subprocess\n"
                "  subprocess.check_call([\n"
                "      sys.executable, '-m', 'pip', 'install',\n"
                "      'gpboost', '--break-system-packages'\n"
                "  ])"
            )
            return False
            
