import importlib

def build_from_config(cfg: dict):
    """
    Generic constructor for any class given a config with module, class, and args.
    """
    module_path = cfg['module']
    class_name = cfg['class']
    args = cfg.get('args', {})

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    print(args)
    return cls(**args)