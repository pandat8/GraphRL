import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolutionLayer(Module):
    """
    Convolution layer for Graph
    """

    def __init__(self, nfeatures_in, nfeatures_out, bias = True):

        super(GraphConvolutionLayer, self).__init__()
        self.nfeatures_in = nfeatures_in
        self.nfeatures_out = nfeatures_out

        # initialize the parameters
        self.weight = Parameter(torch.FloatTensor(self.nfeatures_in,self.nfeatures_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.nfeatures_out))
        else:
            self.register_parameter('bias',None)
        self.init_parameters()

    def init_parameters(self):
        """
        initialize the parameters by uniform distribution
        """
        uni_dis = 1./ math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-uni_dis,uni_dis)
        if self.bias is not None:
            self.bias.data.uniform_(-uni_dis,uni_dis)

    def forward(self, features, adj_matrix):
        features = torch.mm(features, self.weight) # features * weight
        features = torch.mm(adj_matrix, features) # adjacency matrix * features
        if self.bias is not None:
            return features+ self.bias
        else:
            return features

    def __repr__(self):
        return self.__class__.__name__ \
        +'('+str(self.nfeatures_in) \
        +' -> '+str(self.nfeatures_out)+')'




