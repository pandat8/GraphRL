import torch.nn as nn
import torch.nn.functional as F

from gcn.layers_gcn import GraphConvolutionLayer

class GCN(nn.Module):
    """
    Graph convolution model architecture
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer(nhidden, nout) # second graph conv layer
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # first layer with relu
        features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()

        return features


