import torch
import torch.nn as nn
import math


base_model = [
    #expand_ratio, channels, repeats,  stride, kernel_size
    [1, 16,1,1,3],
    [6,24,2,2,3],
    [6,40,2,2,5],
    [6,80,2,2,3],
    [6,112,3,1,5],
    [6,192,4,2,5],
    [6,320,1,1,3]
]

phi_values = {
    # tuple of (phi_value, resolution and droprate)
    "b0" : (0,224, 0.2), # alpha, beta, gamma, depth = alpha **phi
    "b1" : (0.5, 240, 0.2),
    "b2" : (1, 260, 0.3),
    "b3" : (2, 300, 0.3),
    "b4" : (3, 380, 0.4),
    "b5" : (4, 456, 0.4),
    "b6" : (5, 528, 0.5),
    "b7" : (6, 600, 0.5),

}


class CNNBlock(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            inchannels,
            outchannels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(outchannels)
        self.silu = nn.ReLU() # needs to be SiLU but not present in torch 1.4.0
    
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    # calculate attention scores for each of the channels
    def __init__(self, inchannels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels, reduced_dim, 1),
            nn.ReLU(), #SiLU(),
            nn.Conv2d(reduced_dim, inchannels, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, inchannels, 
                outchannels, 
                kernel_size, 
                stride, 
                padding, 
                expand_ratio, 
                reduction=4, # squeeze excitation
                survival_prob=0.8 # for stochatic depth
                ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = inchannels==outchannels and stride==1
        hidden_dims = inchannels*expand_ratio
        self.expand = inchannels!=hidden_dims
        reduced_dim = inchannels//reduction

        if self.expand:
            self.expand_conv = CNNBlock(
                inchannels, hidden_dims, kernel_size=3, stride=1, padding=1
            )
        
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dims, hidden_dims, kernel_size, stride, padding, groups=hidden_dims
            ),
            SqueezeExcitation(hidden_dims, reduced_dim),
            nn.Conv2d(hidden_dims, outchannels, 1, bias=False),
            nn.BatchNorm2d(outchannels)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0],1,1,1,device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob)*binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)



class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = math.ceil(1280*width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32*width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        inchannels = channels
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*math.ceil(int(channels*width_factor)/4)
            layer_repeats = math.ceil(repeats*depth_factor)
            for layer in range(layer_repeats):
                features.append(
                    InvertedResidualBlock(
                        inchannels,
                        out_channels, 
                        expand_ratio=expand_ratio,
                        stride=stride if layer==0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2
                    )
                )
                inchannels=out_channels
        features.append(CNNBlock(
            inchannels, last_channels, kernel_size=1, stride=1, padding=0
        ))

        return nn.Sequential(*features)
        

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_clases = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    print(x.shape)
    model = EfficientNet(
        version=version, 
        num_classes=num_clases
    ).to(device)

    print(model(x).shape)



if __name__=="__main__":
    test()








