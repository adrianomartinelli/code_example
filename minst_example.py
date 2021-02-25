# %%
import numpy as np

import torch.optim
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

from trainer import Trainer
from cellAutoencoder import CellAutoencoder

# %% load data
minst = MNIST('data', transform=None, download=True)
data = [np.asarray(minst[i][0], dtype=np.float32) for i in range(len(minst))]
data = np.stack(data)

# MinMax transform
_min, _max = -1,1
data = (data - data.min()) / (data.max() - data.min()) * (_max - _min) + _min

X = data.reshape(len(data), -1)
# print(X.shape)

# %% train model

model = CellAutoencoder(in_features=784)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = ExponentialLR(optimizer, gamma=0.999)
criterion = nn.MSELoss()

trainer = Trainer(model, X, X, optimizer, criterion, scheduler)
trainer.train_model(epochs=50)
trainer.plot_loss('loss.png')

# %% plot result
test_set = trainer.test_set
n_img = 5
fig, axes = plt.subplots(2, n_img)
rnd_idx = np.random.choice(range(len(test_set)), size=n_img, replace=False)

ax = axes[0,]
for idx, a in zip(rnd_idx, ax):
    img, _ = test_set[idx]
    img = img.reshape((28, 28))
    a.imshow(img, cmap='gray')

ax = axes[1,]
for idx, a in zip(rnd_idx, ax):
    img, _ = test_set[idx]
    img = img.reshape(1, *img.shape)
    img = model(img).detach().numpy()
    img = img.reshape((28, 28))
    a.imshow(img, cmap='gray')

for ax in axes.flat:
    ax.set_axis_off()

fig.subplots_adjust(top=1 - .2, bottom=.2)
fig.tight_layout(pad=.2)
fig.savefig('examples.png')
fig.show()

# %%
