
import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.03)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    
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
        self.apply(weights_init_normal)
        
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
        self.apply(weights_init_normal)
        
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
