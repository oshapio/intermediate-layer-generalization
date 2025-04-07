''' The main piece of code for probing the model throughout the notebook '''

import torch
import torch.nn.functional as F

class LinearProbeModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        lr=0.0001,
        reg_type="l2",
        reg_weight=0.0,
        sgd_momentum=0.0,
        optimizer="SGD",
    ):
        super(LinearProbeModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.lr = lr
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.sgd_momentum = sgd_momentum

        # make an optimizer
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=sgd_momentum)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.fc(x)

    def step_loss(self, x, y):
        ''' 
            x: representation from the backbone at the layer of interest (batch_size, feature_size)
            y: labels (batch_size)
        '''
        # get the output
        out = self.forward(x)
        accuracy = (torch.argmax(out, dim=1) == y).float().mean()
        loss = F.cross_entropy(out, y)
        # add regularization
        if self.reg_type == "l2":
            loss += self.reg_weight * torch.norm(self.fc.weight, p=2)
        elif self.reg_type == "l1":
            loss += self.reg_weight * torch.norm(self.fc.weight, p=1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "preds": out
        }
