import torch.nn as nn
import torch.nn.functional as F

from gcn.layers_gcn import GraphConvolutionLayer, GraphConvolutionLayer_Sparse, GraphConvolutionLayer_Sparse_Memory, GraphAttentionConvLayer


class GCN_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Policy_SelectNode, self).__init__()

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


class GCN_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Value, self).__init__()

        self.gc1 = GraphConvolutionLayer(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer(nhidden, nhidden) # second graph conv layer
        self.gc3 = GraphConvolutionLayer(nhidden, nhidden)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        # features = self.gc2(features, adj_matrix)
        # features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        features = features

        return features


class GCN_Sparse_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_SelectNode, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()

        return features

class GCN_Sparse_Policy_5(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_5, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nin, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nin, nhidden)  # first graph conv layer
        self.gc4 = GraphConvolutionLayer_Sparse(nin, nhidden)  # first graph conv layer
        self.gc5 = GraphConvolutionLayer_Sparse(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)

        features = self.gc3(features, adj_matrix)
        features = F.relu(features)

        features = self.gc4(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc5(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()

        return features


class GCN_Sparse_Memory_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Memory_Policy_SelectNode, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse_Memory(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc2(features, adj_matrix)
        features = F.softmax(features.t())
        features = features.t()

        return features


class GCN_Sparse_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Value, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nhidden) # second graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        # features = self.gc2(features, adj_matrix)
        # features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        features = features

        return features

class GAN(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN, self).__init__()

        self.gc1 = GraphAttentionConvLayer(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayer(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features


class GAN_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN_Value, self).__init__()

        self.gc1 = GraphAttentionConvLayer(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha) # second graph conv layer
        self.gc3 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        features = features

        return features


class GNN_GAN(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GNN_GAN, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphAttentionConvLayer(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features
