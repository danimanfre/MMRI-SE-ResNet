import torch
from torch import nn

klg_labels = {0: "None", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"}


use_slices = list(range(10, 70)) + list(range(90, 150))

slice_norm_cfg = {"center": (max(use_slices) + min(use_slices)) / 2,
                  "scale": (max(use_slices) - min(use_slices)) / 2}


# %%
class SE_injSlice(nn.Module):
    def __init__(self, latent_channel, use_slice):
        super().__init__()
        self.use_slice = use_slice

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1_layers = nn.Sequential(
            nn.Linear(latent_channel + (1 if use_slice else 0), latent_channel // 2),
            # nn.BatchNorm1d(latent_channel // 2),
            nn.LeakyReLU(),
        )
        self.fc2_layers = nn.Sequential(
            nn.Linear(latent_channel // 2 + (1 if use_slice else 0), latent_channel),
            # nn.BatchNorm1d(latent_channel),
            nn.Sigmoid(),
        )

    def forward(self, input, slice):
        x = self.avgpool(input).view(input.shape[0], -1)
        x = self.fc1_layers(torch.cat([x, slice], dim=-1) if self.use_slice else x)
        x = self.fc2_layers(torch.cat([x, slice], dim=-1) if self.use_slice else x)
        x = x.view(input.shape[0], -1, 1, 1)

        return x * input


class Residual_SE(nn.Module):
    def __init__(self, in_channel, out_channel, downsample_hw=True, use_slice=False):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Dropout2d(0.25),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=(3 // 2)),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=(3 // 2)),
            # nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(),
        )

        self.squeeze_excitation = SE_injSlice(in_channel, use_slice)

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=2 // 2, count_include_pad=False) if downsample_hw else nn.Identity(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )

    def forward(self, input, slice):
        x = self.residual_block(input)
        x = self.squeeze_excitation(x, slice)
        x = x + input

        x = self.downsample(x)
        return x


class OAI_KLG_Model(nn.Module):
    def __init__(self, in_channel, latent_channels, use_slice=False, is_binary_problem = False):
        super().__init__()
        self.use_slice = use_slice
        self.is_binary_problem = is_binary_problem

        self.first_convs = nn.Sequential(
            nn.Conv2d(in_channel, latent_channels[0], kernel_size=5, stride=2, padding=(5 // 2) + (2 - 1)),
            # nn.BatchNorm2d(latent_channels[0]),
            nn.LeakyReLU(),
        )

        self.residual_se_blocks = nn.ModuleList([
            Residual_SE(latent_channels[i], latent_channels[i + 1], downsample_hw=i < 4, use_slice=use_slice)
            for i in range(len(latent_channels) - 1)
        ])

        self.avgpools = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.25),
        )

        if(not is_binary_problem):
            self.last_layer = nn.Linear(latent_channels[-1] + (1 if use_slice else 0), len(klg_labels))
        else:
            self.last_layer = nn.Linear(latent_channels[-1] + (1 if use_slice else 0), 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, input, slice):
        input = input.unsqueeze(1)
        if self.use_slice:
            slice = (slice.unsqueeze(-1) - slice_norm_cfg["center"]) / slice_norm_cfg["scale"]

        x = self.first_convs(input)
        for block in self.residual_se_blocks:
            x = block(x, slice)

        x = self.avgpools(x).view(input.shape[0], -1)
        x = self.last_layer(torch.cat([x, slice], dim=-1) if self.use_slice else x)
        if (self.is_binary_problem):
            x = self.sigmoid(x)
        return x
