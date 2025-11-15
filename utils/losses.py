import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math

class NC1Loss(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super(NC1Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # a = torch.pow(x, 2).sum(dim=1, keepdim=True)
        # b = torch.pow(self.means, 2).sum(dim=1, keepdim=True)
        # print(f'a: {a.shape}, b: {b.shape}')
        # print(a)
        # print(f'distmat 1: {distmat.shape}')
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2) # (batch_size, num_classes)
        # print(f'distmat 2: {distmat.shape}')

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        # print(f'labels: {labels.shape}')
        # input()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # print(f'labels: {labels}')
        # print(f'classes: {classes.expand(batch_size, self.num_classes)}')
        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}')

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-10
        # print(f'D: {D}')
        # print(f'N: {N}')
        # print(f'D/N: {D/N}')
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC1Loss_v1(nn.Module):
    '''
    Modified Center loss, 1 / n_k**2 ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-5
        N = N ** 2
        # print()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC1Loss_v2(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2) # (batch_size, num_classes)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot

        dist = distmat * mask.float()
        # D = torch.sum(dist, dim=0)
        # N = mask.float().sum(dim=0) + 1e-10
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

def pairwise_cosine_distance(x, y):
    '''
    x: (N, D)
    y: (M, D)
    return: (N, M)
    '''
    # print(f'norm x: {x.norm(dim=1, keepdim=True)}')
    # print(f'norm y: {y.norm(dim=1, keepdim=True)}')
    # Normalize x and y to have unit norm
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-10)
    y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-10)

    # Compute the cosine similarity
    cosine_similarity = torch.mm(x_norm, y_norm.t())

    # Compute the cosine distance
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

class NC1Loss_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k * ||h-miu|| / (||h|| * ||miu||)
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        # print(f'distmat: {distmat}')
        # if torch.isnan(distmat).any():
        #     print('nan')
        #     input()
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-10 # torch.size(num_classes)
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC1Loss_v1_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k**2 * ||h-miu|| / (||h|| * ||miu||)
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        # print(f'distmat: {distmat}')
        # if torch.isnan(distmat).any():
        #     print('nan')
        #     input()
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-5
        N = N ** 2
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC1Loss_v2_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means
    
class NC1Loss_v3_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^2 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v3_cosine, which uses squared n_k')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 2 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means
    
class NC1Loss_v4_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^1.5 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v4_cosine, which uses n_k^1.5')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 1.5 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means
    
class NC1Loss_v5_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^0.5 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v5_cosine, which uses n_k^0.5')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 0.5 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_v6_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^0.0 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v6_cosine, which uses n_k^0.0')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 0.5 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        loss = D.clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_v7_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^0.8 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v7_cosine, which uses n_k^0.8')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 0.8 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_v8_cosine(nn.Module):
    '''
    Modified Center loss, 1 / n_k^1.2 * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v8_cosine, which uses n_k^1.2')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** 1.2 + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_v9_cosine(nn.Module):
    '''
    Modified Center loss, 1 / log2(n_k + 1) * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False):
        super().__init__()
        print(f'Use NC1Loss_v9_cosine, which uses log2(n_k + 1)')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = torch.log2(self.N + 1) + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_v10_cosine_weight_factor(nn.Module):
    '''
    Modified Center loss, 1 / n_k^weight_factor * ||h-miu|| / (||h|| * ||miu||), here n_k will be calculated from the entire training set insead of the mini-batch
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0', occurance_list=None, fixed_means=False, weight_factor=1.0):
        super().__init__()
        print(f'Use NC1Loss_v10_cosine_weight_factor, which uses n_k^{weight_factor}')
        # input()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.fixed_means = fixed_means
        
        if not fixed_means:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            print('#####################Fixed means##################')
            self.means = torch.randn(self.num_classes, self.feat_dim).to(self.device)
        self.N = torch.tensor(occurance_list).to(self.device)
        self.weight_factor = weight_factor

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        # print(f'mask: {mask}, {mask.shape}')
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0
        # print(f'mask2: {mask2}, {mask2.shape}, {(self.N > 0).sum()}')

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N ** self.weight_factor + 1e-10 # torch.size(num_classes)
        # print(f'D: {D}, {D.dtype}, {D.min()}, {D.max()}')
        # print(f'N: {N}, {N.dtype}, {N.min()}, {N.max()}')
        # print(f'D/N: {D/N}, {(D/N).dtype}, {(D/N).min()}, {(D/N).max()}')
        # i = torch.argmax(D/N)
        # print(f'argmax D/N: {i}, {D[i]}, {N[i]}')
        # input()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # print(f'loss: {loss}')
        # input()
        return loss, self.means

class NC1Loss_cosine_mlab(nn.Module):
    def __init__(self, num_single_classes=10, num_multi_classes=10, feat_dim=128, multi_class_idx_list=None, device='cuda:0', occurance_list=None):
        super().__init__()
        self.num_single_classes = num_single_classes
        self.num_multi_classes = num_multi_classes
        self.num_classes = num_single_classes + num_multi_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        self.single_class_means = torch.eye(self.num_single_classes, self.feat_dim).to(self.device)
        self.multi_class_means = []
        for multi_class_idx in multi_class_idx_list:
            mean = torch.mean(self.single_class_means[multi_class_idx], dim=0)
            mean = mean / mean.norm(p=2)
            self.multi_class_means.append(mean)
        self.multi_class_means = torch.vstack(self.multi_class_means)
        self.means = torch.cat((self.single_class_means, self.multi_class_means), dim=0)
        print(f'means: {self.means.shape}')
        
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N + 1e-10 # torch.size(num_classes)

        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC1Loss_cosine_mlab_v1(nn.Module):
    def __init__(self, num_single_classes=10, num_multi_classes=10, feat_dim=128, multi_class_idx_list=None, device='cuda:0', occurance_list=None):
        super().__init__()
        self.num_single_classes = num_single_classes
        self.num_multi_classes = num_multi_classes
        self.num_classes = num_single_classes + num_multi_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        self.single_class_means = torch.randn(self.num_single_classes, self.feat_dim).to(self.device)
        self.multi_class_means = []
        for multi_class_idx in multi_class_idx_list:
            mean = torch.mean(self.single_class_means[multi_class_idx], dim=0)
            mean = mean / mean.norm(p=2)
            self.multi_class_means.append(mean)
        self.multi_class_means = torch.vstack(self.multi_class_means)
        self.means = torch.cat((self.single_class_means, self.multi_class_means), dim=0)
        print(f'means: {self.means.shape}')
        # print(f'means: {self.means}')
        # input()
        self.N = torch.tensor(occurance_list).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        distmat = pairwise_cosine_distance(x, self.means)
        
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes)) # one-hot
        mask2 = self.N.unsqueeze(0).expand(batch_size, self.num_classes) > 0

        dist = distmat * mask.float() * mask2.float()
        D = torch.sum(dist, dim=0)
        N = self.N + 1e-10 # torch.size(num_classes)

        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means
    
class NC2Loss(nn.Module):
    '''
    NC2 loss v0: maximize the average minimum angle of each centered class mean
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        # dim=1 means the maximum angle of the other class to each class
        loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NC2Loss_v1(nn.Module):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        # dim=1 means the maximum angle of the other class to each class
        loss = -torch.acos(max_cosine)
        min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NC2Loss_v2(nn.Module):
    '''
    NC2 loss v2: make the cosine of any pair of class-means be close to -1/(C-1))
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        C = means.size(0)
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
        cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        loss = cosine.norm()
        # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NCLoss(nn.Module):
    def __init__(self, sup_criterion, lambda1, lambda2, lambda_CE=1.0, nc1='NC1Loss', nc2='NC2Loss', num_classes=1920, feat_dim=2000, device='cuda:0', occurance_list=None, fixed_means=False, weight_factor=None):
        super().__init__()
        # if nc1 not in ['NC1Loss_v2_cosine', 'NC1Loss_v3_cosine']:
        #     self.NC1 = globals()[nc1](num_classes, feat_dim, device)
        # else:
        #     self.NC1 = globals()[nc1](num_classes, feat_dim, device, occurance_list, fixed_means)
        if weight_factor is None:
            self.NC1 = globals()[nc1](num_classes, feat_dim, device, occurance_list, fixed_means)
        else:
            self.NC1 = globals()[nc1](num_classes, feat_dim, device, occurance_list, fixed_means, weight_factor)
        self.NC2 = globals()[nc2]()
        self.sup_criterion = globals()[sup_criterion]()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_CE = lambda_CE
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    def forward(self, y_pred, labels, features):
        sup_loss = self.sup_criterion(y_pred, labels)

        nc1_loss, means = self.NC1(features, labels)
        nc2_loss, max_cosine = self.NC2(means)
        
        # print(sup_loss, nc1_loss, nc2_loss)
        # input()
        loss = self.lambda_CE * sup_loss + self.lambda1 * nc1_loss + self.lambda2 * nc2_loss
        return loss, (sup_loss, nc1_loss, nc2_loss, max_cosine, means)

    def set_lambda(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        print(f'Set weights: lambda1={self.lambda1}, lambda2={self.lambda2}, lambda_CE={self.lambda_CE}')
        
    def set_lambda_CE(self, lambda_CE):
        self.lambda_CE = lambda_CE
        print(f'Set weights: lambda1={self.lambda1}, lambda2={self.lambda2}, lambda_CE={self.lambda_CE}')
        
    def freeze_means(self):
        if self.NC1.fixed_means:
            print('The means of NC1 are fixed, not need to freeze')
            return
        self.NC1.means.requires_grad = False
        print('Freeze the means of NC1')
    
    def unfreeze_means(self):
        if self.NC1.fixed_means:
            print('The means of NC1 are fixed, not need to unfreeze')
            return
        self.NC1.means.requires_grad = True
        print('Unfreeze the means of NC1')
    
if __name__ == '__main__':
    device = 'cuda:0'
    # criterion = NC1Loss_cosine_mlab(num_single_classes=5, num_multi_classes=2, feat_dim=8, multi_class_idx_list=[[0, 1], [2, 3]], device=device, occurance_list=[1, 1, 1, 10, 1, 1, 1])
    # print(criterion.means)
    criterion = NC1Loss_v2_cosine(num_classes=5, feat_dim=8, device=device, occurance_list=[1, 1, 1, 10, 1])
    state_dict = criterion.state_dict()
    criterion.load_state_dict(state_dict)
    print(state_dict)
    print(criterion.means)
    print(criterion.parameters())
    