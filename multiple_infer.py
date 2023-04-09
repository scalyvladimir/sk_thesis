from subprocess import call
import yaml


model_name = 'segformer' # 'unet'

domain_list = [
    'philips3',
    'philips15',
    'siemens3',
    'siemens15',
    'ge3',
    'ge15',
]

config_path = f'configs/infer_{model_name}.yaml'

with open(config_path) as f:
    data = yaml.safe_load(f)

for domain in domain_list:
    data['model']['checkpoint_path'] = f'checkpoints/{domain}_{model_name}.ckpt'

    with open(config_path, 'w') as f:
        yaml.dump(data, f)

    for domain in domain_list:
        data['data']['data_path'] = f'/home/v_chernyy/thesis/dataframes/{domain}.csv'

        with open(config_path, 'w') as f:
            yaml.dump(data, f)

        call(['python', f'infer_{model_name}.py'])