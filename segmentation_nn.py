"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        self.features= models.vgg16_bn(pretrained=True, progress=True).features
        for param in self.features:
            param.requires_grad = False

        #1D convolutions  which bring features output to number of classes
        self.conv1D_downstream1 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1D_downstream2 = torch.nn.Conv2d(1024,num_classes, 1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                                                  padding_mode='zeros')

        #1D convolutions bringing pool4 to number of classes
        self.conv1D_pool4 = torch.nn.Conv2d(512,num_classes, 1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                                                  padding_mode='zeros')

        #1D convolutions bringing pool3 to number of classes
        self.conv1D_pool3= torch.nn.Conv2d(256, num_classes, 1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                                            padding_mode='zeros')

        #now combinations of sums and upsamples to merge the 3 downstreams:
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=8, mode='bilinear')


        #print(self.pretrained)


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #print('output of fetures.children() : %s'%str([i for i in self.features.children()]))
        #print("shape of input is %s" % str(x.size()))
        for layer_no, layer in enumerate(self.features.children()):

            if layer_no is 23:
                y = layer(x)
            if layer_no is 33:
                z = layer(x)
            x = layer(x)

            #print('debug')
            #print('layer info: %s'%str(layer))
            #print("shape of x is %s" % str(x.size()))

        x = self.conv1D_downstream1(x)
        x = self.conv1D_downstream2(x)
        x = self.upsample_1(x)

        z = self.conv1D_pool4(z)
        y = self.conv1D_pool3(y)
        #print('debug')
        #print("shape of x is %s"%str(x.size()))
        #print("shape of z is %s" % str(z.size()))

        if x.size() is not z.size():
            x = nn.functional.interpolate(x,size = (z.size()[2],z.size()[3]), mode = 'nearest')
        x = x+ z
        x = self.upsample_2(x)
        x = x+y
        x = self.upsample_3(x)

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
