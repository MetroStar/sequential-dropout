import torch
import torch.nn as nn
import numpy as np
from SequentialDropout import SequentialDropout




class MNISTAutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim,fc2_input_dim=128, use_sq_dr= True,dr_min_p= .2, scale_output=False):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        self.fc2_input_dim = fc2_input_dim

            ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()

        ### Define an optimizer (both for the encoder and the decoder!)
        lr= 0.001

        ### Set the random seed for reproducible results
        #torch.manual_seed(0)


        #model = Autoencoder(encoded_space_dim=encoded_space_dim)
        self.encoder = Encoder(encoded_space_dim=self.encoded_space_dim,fc2_input_dim=self.fc2_input_dim,dr_min_p= dr_min_p, use_sq_dr= use_sq_dr)
        self.decoder = Decoder(encoded_space_dim=self.encoded_space_dim,fc2_input_dim=self.fc2_input_dim)
        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        self.optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

        # Check if the GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        # Move both the encoder and the decoder to the selected device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        #summary(encoder, input_size=(1, 28, 28))
        #summary(decoder, input_size=(self.encoded_space_dim,))
        
    def train_epoch(self, dataloader):
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            
            image_batch = image_batch.to(self.device)
            
            # Encode data
            encoded_data = self.encoder(image_batch)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_batch)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Print batch loss
            #print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)
    
    ### Testing function
    def test_epoch(self, dataloader):
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in dataloader:
                # Move tensor to the proper device
                image_batch = image_batch.to(self.device)
                # Encode data
                encoded_data = self.encoder(image_batch)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data
    

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim,use_sq_dr=True, dr_min_p= .2, scale_output=False):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        if use_sq_dr:
            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 32, 128),
                nn.ReLU(True),
                nn.Linear(128, encoded_space_dim),
                SequentialDropout(min_p = dr_min_p,scale_output=scale_output)
            )
        else:
            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 32, 128),
                nn.ReLU(True),
                nn.Linear(128, encoded_space_dim)
            )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x