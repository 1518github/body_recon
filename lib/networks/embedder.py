import torch
from lib.config import cfg

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']       # 3
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)   # len: 1
            out_dim += d    # +3
        
        max_freq = self.kwargs['max_freq_log2']     # 9  / 3
        N_freqs = self.kwargs['num_freqs']          # 10 / 4

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        # freq_bands = tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        # freq_bands = tensor([  1.,   2.,   4.,   8])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:    # [cos , sin]
                embed_fns.append(   
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))     # cos(2^n * x)共2*freq_bands种
                out_dim += d    # freq_bands * 6 = 60 / 24

        self.embed_fns = embed_fns
        self.out_dim = out_dim  # 63 / 27
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)     # torch.Size([1, 1024, 64, 63])


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,           # 3
        'max_freq_log2': multires - 1,      # 9     /     3
        'num_freqs': multires,              # 10    /     4
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    
    return embed, embedder_obj.out_dim


xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res)       # cfg.xyz_res = 10
view_embedder, view_dim = get_embedder(cfg.view_res)    # cfg.view_res = 4

baseline_xyz_embedder, baseline_xyz_dim = get_embedder(6)
