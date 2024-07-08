import os
import yaml
from munch import munchify

def load(config='config'):    
    with open(os.path.join('configs', config + '.yaml')) as f:
        parameters = munchify(yaml.load(f, Loader=yaml.SafeLoader))
        parameters.config = config
    return parameters