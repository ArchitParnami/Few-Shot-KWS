import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from protonets.models import register_model
from protonets.models.encoder.TCResNet import TCResNet8
from .utils import euclidean_dist

class ProtonetWithMargin(nn.Module):
    def __init__(self, encoder,  alpha, margin):
        super(ProtonetWithMargin, self).__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.margin = margin
       

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        k = torch.arange(n_class)
        query_onehots = torch.zeros(n_class, n_query, n_class)
        query_onehots[k,:,k] = 1
        query_onehots = query_onehots.view(n_class * n_query, n_class)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()

        if xq.is_cuda:
            target_inds = target_inds.cuda()
            query_onehots = query_onehots.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)


        
        z = self.encoder.forward(x)
        zn = torch.norm(z,p=2, dim=1).view(-1,1)
        z = z.div(zn.expand_as(z))
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        sq_dists = euclidean_dist(zq, z_proto)
        dists = torch.pow(sq_dists, 1/2)

        loss_pull  = self.alpha * torch.sum(torch.mul(sq_dists, query_onehots))
        loss_push =  (1-self.alpha) * torch.sum(
            torch.mul(torch.pow((self.margin - dists), 2), (1-query_onehots)))
        
        loss_val = (loss_pull + loss_push) / (n_class * n_query)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)


        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
