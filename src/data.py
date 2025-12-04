from torch.utils.data import Dataset



class TrainDataset(Dataset):
    """
    Dataset class for training diffusion models
    
    Data Structure:
    During training, the diffusion model learns to recover clean data x_0 
    from the noisy distribution q(x_t|x_0)
    
    Args:
        opt: Configuration dictionary containing model and training parameters
        x0: Clean target data (Ground Truth)
            - Shape: [N, C, H, W] or [N, D]
            - N: Number of samples
            - C: Number of dimension of features
            - H, W: Height and width of the data
        cond: Conditional input data (optional)
            - Shape: [N, F, H, W] or [N, F]
            - F: Number of dimension of conditional features
            - Used for conditional generation p(x|y), such as super-resolution, 
              inpainting, etc.
            - If None, performs unconditional generation p(x)
    
    Returns:
        data_dict: Dictionary containing the following key-value pairs:
            - 'x0': Target clean data used for loss calculation
            - 'cond': Conditional input (if provided), concatenated with x_t 
                      before feeding into the model
    """
    def __init__(self, x0, cond=None):
        super(TrainDataset, self).__init__()
        self.x0 = x0
        self.cond = cond
    
    def __len__(self):
        return self.x0.size(0)
    
    def __getitem__(self, index):
        data_dict = {'x0': self.x0[index]}
         
        if self.cond is not None:
            data_dict['cond'] = self.cond[index]
        
        return data_dict
