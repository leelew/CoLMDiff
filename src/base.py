import torch



class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0:
            self.device = torch.device(f'cuda:{opt["gpu_ids"][0]}')
        else:
            self.device = torch.device('cpu')        
        self.begin_step = 0
        self.begin_epoch = 0

    def load_one_sample(self, data):
        pass

    def train_one_sample(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        """Assign different data types to device."""
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x
