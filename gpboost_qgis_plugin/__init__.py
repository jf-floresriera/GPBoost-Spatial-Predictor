def classFactory(iface):
    from .gpboost_plugin import GPBoostPlugin
    return GPBoostPlugin(iface)
