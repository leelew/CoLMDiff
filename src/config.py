import argparse
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--path_config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU IDs (e.g., 0,1,2,3)')
    args = parser.parse_args()
    
    with open(args.path_config, 'r') as f:
        opt = yaml.safe_load(f)
    opt["gpu_ids"] = [int(x.strip()) for x in args.gpu_ids.split(',')]
    return opt