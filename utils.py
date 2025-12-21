import yaml
from types import SimpleNamespace

def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = SimpleNamespace(**cfg_dict)
    cfg.system = SimpleNamespace(**cfg_dict['system'])
    cfg.data = SimpleNamespace(**cfg_dict['data'])
    cfg.model = SimpleNamespace(**cfg_dict['model'])
    cfg.train = SimpleNamespace(**cfg_dict['train'])
    return cfg