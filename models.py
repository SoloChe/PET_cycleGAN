
import torch.nn as nn
import torch.nn.functional as F
import torch


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         if hasattr(m, "bias") and m.bias is not None:
#             torch.nn.init.constant_(m.bias.data, 0.0)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
class Generater_MLP(nn.Module):
    def __init__(self, input_shape, latent_dim, width=256):
        super(Generater_MLP, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, self.input_shape),
        )
    
    def forward(self, x):
        return self.model(x)
    
class Discriminator_MLP(nn.Module):
    def __init__(self, input_shape, width=256):
        super(Discriminator_MLP, self).__init__()
        self.input_shape = input_shape 
        self.model = nn.Sequential(
            nn.Linear(input_shape, width),
            nn.LeakyReLU(),
            nn.Linear(width, width),
            nn.LeakyReLU(),
            nn.Linear(width, width),
            nn.LeakyReLU(),
            # nn.Linear(width, width),
            # nn.LeakyReLU(),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
class Generater_MLP_Skip(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Generater_MLP_Skip, self).__init__()
        self.num_layers = num_layers
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Define activation function
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        # Input layer
        out = self.activation(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i in range(self.num_layers):
            residual = out
            out = self.activation(self.hidden_layers[i](out))
            out = out + residual  # Skip connection
        
        # Output layer
        out = self.output_layer(out)
        return out
    
class Discriminator_MLP_Skip(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(Discriminator_MLP_Skip, self).__init__()
        self.num_layers = num_layers
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        # Define the output layer
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())
    
        # Define activation function
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        # Input layer
        out = self.activation(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i in range(self.num_layers):
            residual = out
            out = self.activation(self.hidden_layers[i](out))
            out = out + residual  # Skip connection
        
        # Output layer
        out = self.output_layer(out)
        return out
