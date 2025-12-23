import yaml
from types import SimpleNamespace

def recursive_namespace(d):
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = recursive_namespace(v)
        return SimpleNamespace(**d)
    return d

def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return recursive_namespace(cfg_dict)