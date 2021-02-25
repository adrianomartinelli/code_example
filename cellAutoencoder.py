# %%
from torch import nn
# %%

class CellAutoencoder(nn.Module):

    def __init__(self,
                 in_features=20,
                 encoder_layers=[128, 64, 32, 16, 8],
                 decoder_layers=[32, 128, 64]):
        super().__init__()

        # append size of input to decoder layers
        decoder_layers.append(in_features)

        # encoder
        encoder = [nn.Linear(in_features, encoder_layers[0])]
        for l in range(len(encoder_layers) - 1):
            encoder.append(nn.ReLU())
            encoder.append(nn.Linear(encoder_layers[l], encoder_layers[l + 1]))
        encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        # decoder
        decoder = [nn.Linear(encoder_layers[-1], decoder_layers[0])]
        for l in range(len(decoder_layers) - 1):
            decoder.append(nn.ReLU())
            decoder.append(nn.Linear(decoder_layers[l], decoder_layers[l + 1]))
        decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


