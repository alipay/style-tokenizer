import torch
import torch.nn as nn


class StyleTokenizer(nn.Module):
    """backbone + projection head"""
    def __init__(self, input_size=None, intermediate_size=512, out_size=768, n_tokens=8, with_placeholder=False, prefix_model="mlp"):
        super(StyleTokenizer, self).__init__()
        if input_size is None:
            input_size=512
        #intermediate_size=512
        # out_size=768
        self.n_tokens=n_tokens
        self.with_placeholder=with_placeholder
        self.out_size=out_size
        self.prefix_model = prefix_model
        print(f"StyleTokenizer with prefix_model: {prefix_model}")
        if self.prefix_model == "mlp" or self.prefix_model == "vgg" or self.prefix_model == "clip":
            self.proj = nn.Sequential(
                nn.Linear(input_size, intermediate_size), # apply a fully connected layer with output size intermediate_size
                nn.SiLU(),
                nn.Linear(intermediate_size, out_size * n_tokens),
            )
        elif self.prefix_model == "transformer":
            input_size=768
            self.proj=nn.Linear(input_size, out_size)
        else:
            assert False

        if self.with_placeholder:
            self.start_embedding=nn.Parameter(torch.randn(1, 1, out_size))
            self.end_embedding=nn.Parameter(torch.randn(1, 1, out_size))

    def forward(self, x):
        x = self.proj(x)
        if self.prefix_model == "mlp" or self.prefix_model == "vgg" or self.prefix_model == "clip":
            x = x.reshape(x.shape[0], self.n_tokens, self.out_size)

        if self.with_placeholder:
            start = self.start_embedding.repeat(x.shape[0], 1, 1)
            end = self.end_embedding.repeat(x.shape[0], 1, 1)
            x = torch.cat([start, x, end], dim=1)
        
        return x
