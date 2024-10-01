
import torch.nn as nn
import torch.nn.functional as F
import torch

#%% old version    
class Generater_MLP_Skip_OLD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Generater_MLP_Skip_OLD, self).__init__()
        self.num_layers = num_layers
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Define activation function
        self.activation = nn.LeakyReLU()
        # self.activation = nn.SiLU()
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
    
    
class Discriminator_MLP_Skip_OLD(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(Discriminator_MLP_Skip_OLD, self).__init__()
        self.num_layers = num_layers
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        # Define the output layer
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())
        # self.output_layer = nn.Linear(hidden_size, output_size)
    
        # Define activation function
        self.activation = nn.LeakyReLU()
        # self.activation = nn.SiLU()
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
#%%
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.03) # 0.03
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
class MLPResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(MLPResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out

class Generater_MLP_Skip(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_residual_blocks):
        super(Generater_MLP_Skip, self).__init__()
    
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList([MLPResidualBlock(hidden_size) for _ in range(num_residual_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)
        self.apply(weights_init_normal)
        
    def forward(self, x):
        out = self.activation(self.input_layer(x))
        
        for block in self.residual_blocks:
            out = block(out)
            
        out = self.output_layer(out)
        return out
    
class Discriminator_MLP_Skip(nn.Module):
    def __init__(self, input_size, hidden_size, num_residual_blocks):
        super(Discriminator_MLP_Skip, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList([MLPResidualBlock(hidden_size) for _ in range(num_residual_blocks)])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.apply(weights_init_normal)
        
    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for block in self.residual_blocks:
            out = block(out)
        out = self.output_layer(out)
        return out
    
class PatchMLPDiscriminator_1D(nn.Module):
    def __init__(self, num_patches, patch_size, hidden_size, num_layers):
        """
        input_size: Length of the 1D input sequence
        patch_size: Size of each patch
        hidden_size: Size of hidden layers in the MLP
        num_layers: Number of layers in the MLP
        num_patches: Number of patches the input is divided into
        """
        super(PatchMLPDiscriminator_1D, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Define an MLP that will process each patch
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size),
            nn.LeakyReLU(0.2),
            *[
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2)) for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_size, 1)  # Output for each patch
        )
        
    def forward(self, x):
        # Split the input into patches
        batch_size, seq_len = x.size()
        assert seq_len % self.patch_size == 0, "Sequence length must be divisible by the patch size"
        
        patches = x.view(batch_size, self.num_patches, self.patch_size)  # Shape: (batch_size, num_patches, patch_size)
        
        # Process each patch through the MLP
        patch_outputs = []
        for i in range(self.num_patches):
            patch = patches[:, i, :]  # Shape: (batch_size, patch_size)
            patch_output = self.mlp(patch)  # Shape: (batch_size, 1)
            patch_outputs.append(patch_output)
        
        # Concatenate all patch outputs
        patch_outputs = torch.cat(patch_outputs, dim=1)  # Shape: (batch_size, num_patches)
        
        return patch_outputs  # Outputs for each patch (real or fake for each patch)
    
    
class PatchMLPDiscriminator_1D_Res(nn.Module):
    def __init__(self, num_patches, patch_size, hidden_size, num_residual_blocks):
        """
        input_size: Length of the 1D input sequence
        patch_size: Size of each patch
        hidden_size: Size of hidden layers in the MLP Residual Blocks
        num_residual_blocks: Number of residual blocks in the MLP
        num_patches: Number of patches the input is divided into
        """
        super(PatchMLPDiscriminator_1D_Res, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Initial linear layer to map patch_size to hidden_size
        self.input_layer = nn.Linear(patch_size, hidden_size)
        self.activation = nn.LeakyReLU(0.2)
        
        # Residual blocks for each patch
        self.residual_blocks = nn.Sequential(
            *[MLPResidualBlock(hidden_size) for _ in range(num_residual_blocks)]
        )
        
        # Output layer for each patch
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Split the input into patches
        batch_size, seq_len = x.size()
        
        # padding
        if seq_len % self.patch_size != 0:
            pad = torch.zeros(batch_size, self.patch_size - seq_len % self.patch_size)
            x = torch.cat((x, pad), dim=1)
            seq_len = x.size(1)
            
        assert seq_len % self.patch_size == 0, "Sequence length must be divisible by the patch size"
        
        # Reshape into (batch_size * num_patches, patch_size)
        patches = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Apply the input layer and activation to all patches at once
        out = self.activation(self.input_layer(patches))  # Shape: (batch_size, num_patches, hidden_size)
        
        # Apply residual blocks (automatically applies to all patches in parallel)
        out = self.residual_blocks(out)  # Shape: (batch_size, num_patches, hidden_size)
        
        # Apply the output layer to classify each patch
        out = self.output_layer(out)  # Shape: (batch_size, num_patches, 1)
        
        # Remove the last dimension (batch_size, num_patches, 1) -> (batch_size, num_patches)
        out = out.squeeze(-1)
        
        return out  # Outputs for each patch (real or fake for each patch)

    
    
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
    if __name__ == '__main__':
        model = PatchMLPDiscriminator_1D_Res(17, 5, 100, 2)
        input = torch.randn(32, 85)
        output = model(input)
        print(output.size())