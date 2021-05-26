import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

def nn_blocks(in_feats, out_feats, *args, **kwargs):
    return nn.Sequential(nn.Linear(in_feats, out_feats),
                        nn.ReLU()
                        )



class Discriminator(nn.Module):
    def __init__(self, in_c, enc_sizes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        
        blocks = [nn_blocks(in_f, out_f) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*blocks)
        self.out = nn.Sequential(
                    nn.Linear(self.enc_sizes[-1], 1) 
                            )

        

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.out(x)
        
        
        return x




def _gradient_penalty(self, z_cell, z_drug):
        batch_size = z_cell.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(z_cell)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * z_cell + (1 - alpha) * z_drug
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.Discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, hidden_dim),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
      #  self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
