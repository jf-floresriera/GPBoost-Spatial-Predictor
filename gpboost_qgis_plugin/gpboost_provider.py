from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os
from .gpboost_algorithm import GPBoostTrainPredictAlgorithm


class GPBoostProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(GPBoostTrainPredictAlgorithm())

    def id(self):
        return "gpboost"

    def name(self):
        return "GPBoost"

    def longName(self):
        return "GPBoost - Tree Boosting + Gaussian Process"

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), "icons", "icon.png"))
