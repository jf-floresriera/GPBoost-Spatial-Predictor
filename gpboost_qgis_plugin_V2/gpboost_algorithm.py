"""
GPBoost Train & Predict Algorithm
Modelo: y = F(X) + b(s) + epsilon
  F(X)  : función no lineal via tree-boosting
  b(s)  : Proceso Gaussiano para correlación espacial
  epsilon: error i.i.d.
"""
from qgis.core import (
    QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterNumber,
    QgsProcessingParameterEnum, QgsProcessingParameterExtent,
    QgsProcessingParameterBoolean, QgsProcessingParameterRasterDestination,
    QgsProcessingOutputNumber, QgsProcessingException, QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication
import numpy as np


class GPBoostTrainPredictAlgorithm(QgsProcessingAlgorithm):

    INPUT_LAYER       = "INPUT_LAYER"
    TARGET_FIELD      = "TARGET_FIELD"
    COVARIATE_FIELDS  = "COVARIATE_FIELDS"
    COV_FUNCTION      = "COV_FUNCTION"
    LEARNING_RATE     = "LEARNING_RATE"
    NUM_LEAVES        = "NUM_LEAVES"
    MAX_DEPTH         = "MAX_DEPTH"
    N_ITER            = "N_ITER"
    PREDICTION_EXTENT = "PREDICTION_EXTENT"
    PIXEL_SIZE        = "PIXEL_SIZE"
    USE_CROSSVAL      = "USE_CROSSVAL"
    CV_FOLDS          = "CV_FOLDS"
    OUTPUT_RASTER     = "OUTPUT_RASTER"
    OUTPUT_RMSE       = "OUTPUT_RMSE"

    COV_FUNCTIONS = ["exponential", "gaussian", "matern32", "matern52", "powered_exponential"]

    def tr(self, s):
        return QCoreApplication.translate("GPBoostAlgorithm", s)

    def createInstance(self):
        return GPBoostTrainPredictAlgorithm()

    def name(self):
        return "gpboost_train_predict"

    def displayName(self):
        return self.tr("GPBoost: Train & Predict (Spatial)")

    def group(self):
        return self.tr("Spatial Prediction")

    def groupId(self):
        return "gpboost_prediction"

    def shortHelpString(self):
        return self.tr(
            "<b>GPBoost Spatial Predictor</b><br><br>"
            "Entrena un modelo GPBoost combinando:<br>"
            "• <b>Tree-Boosting</b> para relaciones no lineales entre covariables y variable respuesta<br>"
            "• <b>Proceso Gaussiano</b> para modelar autocorrelación espacial residual<br><br>"
            "<b>Modelo:</b> y = F(X) + b(s) + ε<br><br>"
            "Referencia: Sigrist F. (2022). Gaussian Process Boosting. JMLR 23(232):1-51."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_LAYER, self.tr("Capa de puntos (entrenamiento)"),
            [QgsWkbTypes.PointGeometry]))

        self.addParameter(QgsProcessingParameterField(
            self.TARGET_FIELD, self.tr("Campo variable respuesta (y)"),
            parentLayerParameterName=self.INPUT_LAYER,
            type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterField(
            self.COVARIATE_FIELDS, self.tr("Campos covariables (X)"),
            parentLayerParameterName=self.INPUT_LAYER,
            type=QgsProcessingParameterField.Numeric,
            allowMultiple=True, optional=True))

        self.addParameter(QgsProcessingParameterEnum(
            self.COV_FUNCTION, self.tr("Función de covarianza GP"),
            options=self.COV_FUNCTIONS, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.LEARNING_RATE, self.tr("Learning rate"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.01, minValue=0.001, maxValue=1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.NUM_LEAVES, self.tr("Num. hojas por árbol"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=31, minValue=2, maxValue=256))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_DEPTH, self.tr("Profundidad máxima (-1 = sin límite)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=-1, maxValue=20))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_ITER, self.tr("Iteraciones de boosting"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=200, minValue=10, maxValue=5000))

        self.addParameter(QgsProcessingParameterExtent(
            self.PREDICTION_EXTENT, self.tr("Extensión de predicción")))

        self.addParameter(QgsProcessingParameterNumber(
            self.PIXEL_SIZE, self.tr("Tamaño de píxel (unidades del mapa)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=100.0, minValue=0.0001))

        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_CROSSVAL, self.tr("Usar validación cruzada para n_iter óptimo"),
            defaultValue=False))

        self.addParameter(QgsProcessingParameterNumber(
            self.CV_FOLDS, self.tr("Número de folds para CV"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=4, minValue=2, maxValue=10))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_RASTER, self.tr("Raster de predicción de salida")))

        self.addOutput(QgsProcessingOutputNumber(
            self.OUTPUT_RMSE, self.tr("RMSE (entrenamiento o CV)")))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            import gpboost as gpb
        except ImportError:
            raise QgsProcessingException(
                "gpboost no está instalado. Ejecuta: pip install gpboost")

        feedback.setProgressText("Cargando datos de entrenamiento...")
        feedback.setProgress(5)

        layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        target_field = self.parameterAsString(parameters, self.TARGET_FIELD, context)
        cov_fields = self.parameterAsFields(parameters, self.COVARIATE_FIELDS, context)

        coords, y_list, X_list = [], [], []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom.isNull():
                continue
            pt = geom.asPoint()
            y_val = feat[target_field]
            if y_val is None or y_val == "":
                continue
            row = []
            valid = True
            for cf in cov_fields:
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

        if len(y_list) < 10:
            raise QgsProcessingException(
                f"Puntos insuficientes ({len(y_list)}). Se necesitan al menos 10.")

        coords_np = np.array(coords)
        y_np = np.array(y_list)
        X_np = np.array(X_list) if X_list else None

        feedback.pushInfo(f"Muestras de entrenamiento: {len(y_np)}")
        feedback.setProgress(20)

        cov_idx = self.parameterAsEnum(parameters, self.COV_FUNCTION, context)
        lr = self.parameterAsDouble(parameters, self.LEARNING_RATE, context)
        num_leaves = self.parameterAsInt(parameters, self.NUM_LEAVES, context)
        max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)
        n_iter = self.parameterAsInt(parameters, self.N_ITER, context)
        use_cv = self.parameterAsBoolean(parameters, self.USE_CROSSVAL, context)
        cv_folds = self.parameterAsInt(parameters, self.CV_FOLDS, context)

        gp_model = gpb.GPModel(
            gp_coords=coords_np,
            cov_function=self.COV_FUNCTIONS[cov_idx])

        params = {
            "learning_rate": lr, "num_leaves": num_leaves,
            "max_depth": max_depth, "verbose": -1}

        dataset = gpb.Dataset(data=X_np, label=y_np)

        optimal_iter = n_iter
        rmse_val = None

        if use_cv:
            feedback.setProgressText(f"Validación cruzada {cv_folds}-fold...")
            cv_result = gpb.cv(
                params=params, train_set=dataset, gp_model=gp_model,
                nfold=cv_folds, num_boost_round=n_iter,
                early_stopping_rounds=10, metrics="rmse",
                verbose_eval=False, seed=42)
            key = [k for k in cv_result if "mean" in k][0]
            rmse_vals = cv_result[key]
            optimal_iter = int(np.argmin(rmse_vals)) + 1
            rmse_val = float(np.min(rmse_vals))
            feedback.pushInfo(f"CV n_iter óptimo: {optimal_iter} | CV RMSE: {rmse_val:.4f}")

        feedback.setProgressText(f"Entrenando modelo final ({optimal_iter} iters)...")
        feedback.setProgress(40)

        booster = gpb.train(
            params=params, train_set=dataset,
            gp_model=gp_model, num_boost_round=optimal_iter)

        feedback.setProgress(65)

        if rmse_val is None:
            pred_tr = booster.predict(data=X_np, gp_coords_pred=coords_np, predict_var=False)
            pv = pred_tr["response_mean"] if isinstance(pred_tr, dict) else pred_tr
            rmse_val = float(np.sqrt(np.mean((pv - y_np) ** 2)))
            feedback.pushInfo(f"RMSE entrenamiento: {rmse_val:.4f}")

        try:
            cp = gp_model.get_cov_pars()
            for k, v in zip(["error_var", "gp_var", "gp_range"], cp):
                feedback.pushInfo(f"  {k}: {v:.6f}")
        except Exception:
            pass

        feedback.setProgress(70)

        # Grid de predicción
        extent = self.parameterAsExtent(parameters, self.PREDICTION_EXTENT, context)
        if not extent or extent.isNull():
            extent = layer.extent()

        pixel_size = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)
        x_min, x_max = extent.xMinimum(), extent.xMaximum()
        y_min, y_max = extent.yMinimum(), extent.yMaximum()
        n_cols = max(1, int((x_max - x_min) / pixel_size))
        n_rows = max(1, int((y_max - y_min) / pixel_size))

        if n_cols * n_rows > 2_000_000:
            raise QgsProcessingException(
                f"Grid demasiado grande ({n_cols*n_rows:,} píxeles). "
                "Aumenta el tamaño de píxel o reduce la extensión.")

        xs = np.linspace(x_min + pixel_size/2, x_max - pixel_size/2, n_cols)
        ys = np.linspace(y_max - pixel_size/2, y_min + pixel_size/2, n_rows)
        xx, yy = np.meshgrid(xs, ys)
        pred_coords = np.column_stack([xx.ravel(), yy.ravel()])
        X_pred = np.tile(np.median(X_np, axis=0), (len(pred_coords), 1)) if X_np is not None else None

        feedback.setProgressText("Prediciendo...")
        feedback.setProgress(80)

        pred_result = booster.predict(data=X_pred, gp_coords_pred=pred_coords, predict_var=False)
        pred_array = pred_result["response_mean"] if isinstance(pred_result, dict) else pred_result
        pred_grid = pred_array.reshape(n_rows, n_cols).astype(np.float32)

        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)

        from osgeo import gdal, osr
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(output_path, n_cols, n_rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=LZW", "TILED=YES"])
        ds.SetGeoTransform([x_min, pixel_size, 0.0, y_max, 0.0, -pixel_size])
        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(pred_grid)
        band.SetNoDataValue(-9999.0)
        band.FlushCache()
        ds = None

        feedback.setProgress(100)
        feedback.pushInfo(
            f"\n=== GPBoost Predicción Completa ===\n"
            f"Muestras       : {len(y_np)}\n"
            f"Iteraciones    : {optimal_iter}\n"
            f"Función cov.   : {self.COV_FUNCTIONS[cov_idx]}\n"
            f"RMSE           : {rmse_val:.4f}\n"
            f"Grid (cols×rows): {n_cols} × {n_rows}\n"
        )

        return {self.OUTPUT_RASTER: output_path, self.OUTPUT_RMSE: rmse_val}
