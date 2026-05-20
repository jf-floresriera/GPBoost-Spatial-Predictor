"""
Ejecuta desde la consola Python de QGIS:
  exec(open("/ruta/plugin/install_gpboost_deps.py").read())
"""
import subprocess, sys

print("Instalando gpboost...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "gpboost>=1.4.0"])
print("✅ gpboost instalado. Reinicia QGIS.")
