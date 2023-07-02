import torch.nn as nn 
import torch.nn.functional as F
import torch 

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256):
        super().__init__()
        print(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        x = F.selu(self.fc1(x))
        return self.fc2(x)


# model = MLP(image_batch.shape)


# (batch, channels, h, w)
image_batch = torch.rand(32, 3, 256, 256) # Do we need the channels?
image_batch = image_batch.view(32, -1) # 3x256x256 (?)

model = MLP(input_dim=3*256*256)


result = model(image_batch)
print(result.shape)

