import torch
import torch.nn as nn
import torch.nn.functional as F

def merge_conv_normalization(conv_layer, normalization_type):
    new_layer = torch.nn.Conv2d(
        conv_layer.in_channels,
        conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias,
    )

    if normalization_type == "group":
        if conv_layer.out_channels % 2 == 0:
            group = 2
        else:
            group = 1
        nl = nn.GroupNorm(group, conv_layer.out_channels)

    elif normalization_type == "layer":
        nl = nn.GroupNorm(1, conv_layer.out_channels)

    elif normalization_type == "batch":
        nl = nn.BatchNorm2d(conv_layer.out_channels)

    return nn.Sequential(new_layer, nl)

class Net(nn.Module):
    def __init__(self, normalization_type):
        super(Net, self).__init__()
        # RF = 1
        self.normalization_type = normalization_type
        
        #Input Block
        self.convblock1 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=0, bias=False),   
                self.normalization_type
            ),
            nn.ReLU(),
        ) 
        # input_size = 28*28*1
        # output_size = 26*26*10
        # RF = 3

        self.convblock2 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        ) 
        # input_size = 26*26*10
        # output_size = 24*24*20
        # RF = 5
        
        self.convblock3 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 24*24*20
        # output_size = 24*24*10
        # RF = 5

        # Transition block
        self.maxPool = nn.MaxPool2d(2,2)
        # input_size = 24*24*10
        # output_size = 12*12*10
        # RF = 10

        self.convblock4 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 12*12*10
        # output_size = 10*10*20
        # RF = 12

        self.convblock5 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 10*10*20
        # output_size = 8*8*20
        # RF = 14

        self.convblock6 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=20, out_channels=10, kernel_size=1, stride=1, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 8*8*20
        # output_size = 8*8*10
        # RF = 14

        self.convblock7 = nn.Sequential(
            merge_conv_normalization(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False),
                self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 8*8*10
        # output_size = 6*6*10
        # RF = 16

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6)
        ) 
        # output_size = 1



    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.maxPool(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim = -1)
      

