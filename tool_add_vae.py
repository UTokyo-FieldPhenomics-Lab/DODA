import os


orginal_weight_path = 'models/control_sd15_ini.ckpt'
trained_weight_path = 'logs/***.ckpt'
output_path = 'models/DODA-coco-wvae.ckpt'

assert os.path.exists(orginal_weight_path), 'Original model does not exist.'
assert os.path.exists(trained_weight_path), 'Trained model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./configs/controlnet/coco_train.yaml')

orginal_weight = torch.load(orginal_weight_path)
trained_weight = torch.load(trained_weight_path)

model.load_state_dict(orginal_weight, strict=True)
model.load_state_dict(trained_weight, strict=False)
torch.save(model.state_dict(), output_path)
print('Done.')
