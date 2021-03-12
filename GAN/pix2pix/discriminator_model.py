import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, 4, stride, bias=False, padding_mode="reflect"), # supposedly produces artifacts hence reflect
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2) # always uses leaky relu in discrominator
        )
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 256 -> 30x30
        super().__init__()
        # input channels 2 since it gets the concatenation of the input and the output image side by side
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2 , features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2) # for the last one we have stride 1
            )
            in_channels = feature # to set the inchannel of the next layer
        
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
        self.model = nn.Sequential(*layers) # unpack the layers list

    def forward(self, x, y): # either a y fake or a y real
        x = torch.cat([x,y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)

if __name__ == "__main__":
    test()