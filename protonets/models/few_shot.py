import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model
from protonets.models.encoder.default import C64
from protonets.models.encoder.GoogleKWS import cnn_trad_fpool3
from protonets.models.encoder.TCResNet import TCResNet8
#from torch.utils.tensorboard import SummaryWriter

from .utils import euclidean_dist

class Protonet(nn.Module):
    def __init__(self, encoder, encoding):
        super(Protonet, self).__init__()
        #self.encoding = encoding
        #self.encoder = encoder
        #self.write = False

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        #if not self.write:
            #writer = SummaryWriter('runs/{}'.format(self.encoding))
            #writer.add_graph(self.encoder, x)
            #writer.close()
            #self.write = True
        
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


def get_enocder(encoding, in_dim, hid_dim, out_dim):
    if encoding == 'C64':
        return C64(in_dim, hid_dim, out_dim)
    elif encoding == 'cnn-trad-fpool3':
        return cnn_trad_fpool3(in_dim, hid_dim, out_dim)
    elif encoding == 'TCResNet8':
        return TCResNet8(1, 51, 40)

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    encoder = get_enocder(encoding, x_dim[0], hid_dim, z_dim)
    return Protonet(encoder, encoding)
