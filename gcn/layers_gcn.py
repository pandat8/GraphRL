import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolutionLayer(Module):
    """
    Convolution layer for Graph
    """

    def __init__(self, nfeatures_in, nfeatures_out, init='xavier', bias = True):

        super(GraphConvolutionLayer, self).__init__()
        self.nfeatures_in = nfeatures_in
        self.nfeatures_out = nfeatures_out

        # initialize the parameters
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.weight_hat = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                                   requires_grad=True)
        #Parameter(torch.Tensor(self.nfeatures_in,self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        if bias:
            self.bias = nn.Parameter(nn.init.constant_(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),0.0))
            #Parameter(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        else:
            self.register_parameter('bias',None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)


    def forward(self, features, adj_matrix):
        features_hat = torch.mm(features, self.weight_hat)  # features * weight
        features = torch.mm(features, self.weight) # features * weight
        features = torch.mm(adj_matrix, features) +features_hat # adjacency matrix * features
        if self.bias is not None:
            return features+ self.bias
        else:
            return features

    def __repr__(self):
        return self.__class__.__name__ \
        +'('+str(self.nfeatures_in) \
        +' -> '+str(self.nfeatures_out)+')'


class GraphConvolutionLayer_Sparse(Module):
    """
    Convolution layer for Graph
    """

    def __init__(self, nfeatures_in, nfeatures_out, init='xavier', bias = True):

        super(GraphConvolutionLayer_Sparse, self).__init__()
        self.nfeatures_in = nfeatures_in
        self.nfeatures_out = nfeatures_out

        # initialize the parameters
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        #Parameter(torch.Tensor(self.nfeatures_in,self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        if bias:
            self.bias = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
            #Parameter(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        else:
            self.register_parameter('bias',None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias.data, gain=0.02)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)


    def forward(self, features, adj_matrix):

        features = torch.mm(features, self.weight) # features * weight
        features = torch.spmm(adj_matrix, features) # adjacency matrix * features
        if self.bias is not None:
            return features+ self.bias
        else:
            return features

    def __repr__(self):
        return self.__class__.__name__ \
        +'('+str(self.nfeatures_in) \
        +' -> '+str(self.nfeatures_out)+')'


class GraphConvolutionLayer_Sparse_Memory(Module):
    """
    Convolution layer for Graph
    """

    def __init__(self, nfeatures_in, nfeatures_out, init='xavier', bias = True):

        super(GraphConvolutionLayer_Sparse_Memory, self).__init__()
        self.nfeatures_in = nfeatures_in
        self.nfeatures_out = nfeatures_out

        # initialize the parameters
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.weight_hat = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                                   requires_grad=True)

        #Parameter(torch.Tensor(self.nfeatures_in,self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        if bias:
            self.bias = nn.Parameter(nn.init.constant_(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),0.0))
            #Parameter(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        else:
            self.register_parameter('bias',None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)


    def forward(self, features, adj_matrix):
        features_hat = torch.mm(features, self.weight)  # features * weight
        features = torch.mm(features, self.weight) # features * weight
        features = torch.spmm(adj_matrix, features)+features_hat # adjacency matrix * features
        if self.bias is not None:
            return features+ self.bias
        else:
            return features

    def __repr__(self):
        return self.__class__.__name__ \
        +'('+str(self.nfeatures_in) \
        +' -> '+str(self.nfeatures_out)+')'



class GraphAttentionConvLayer(Module):
    """
    Convolution layer for Graph
    """

    def __init__(self, nfeatures_in, nfeatures_out, dropout, alpha, init='xavier', bias = True):

        super(GraphAttentionConvLayer, self).__init__()
        self.nfeatures_in = nfeatures_in
        self.nfeatures_out = nfeatures_out

        self.alpha = alpha
        # initialize the parameters
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.nfeatures_in, self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*nfeatures_out, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                                   requires_grad=True)
        self.actvation = nn.LeakyReLU(alpha)

        #Parameter(torch.Tensor(self.nfeatures_in,self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        if bias:
            self.bias = nn.Parameter(nn.init.constant_(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),0.0))
            #Parameter(torch.Tensor(self.nfeatures_out).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        else:
            self.register_parameter('bias',None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    # def __init__(self, nfeatures_in, nfeatures_out, dropout, alpha, bias = True):
    #
    #     super(GraphAttentionConvLayer, self).__init__()
    #     self.nfeatures_in = nfeatures_in
    #     self.nfeatures_out = nfeatures_out
    #     self.dropout = dropout
    #     self.alpha = alpha
    #
    #     self.weight = Parameter(torch.FloatTensor(nfeatures_in,nfeatures_out))
    #     self.a = Parameter(torch.FloatTensor(2*nfeatures_out,1))
    #     self.actvation = nn.LeakyReLU(alpha)
    #
    #
    #     if bias:
    #         self.bias = Parameter(torch.FloatTensor(nfeatures_out))
    #     else:
    #         self.register_parameter('bias',None)
    #     self.init_parameters()
    #     # self.sparse_softmax()
    #
    # def init_parameters(self):
    #     uni_dis = 1./ math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-uni_dis,uni_dis)
    #
    #     uni_dis_a = 1./ math.sqrt(self.a.size(1))
    #     self.a.data.uniform_(-uni_dis_a,uni_dis_a)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-uni_dis,uni_dis)

    def sparse_softmax(self, sparse_matrix, dim=0):
        """
        Return the softmax of torch.sparse 2d-tensor on
        :param sparse_matrix:
        :return:
        """
        i = sparse_matrix._indices()
        v = sparse_matrix._values()
        v =F.softmax(v)
        # for n in range(i[dim,:].max()+1):
        #     set=np.where(i[dim,:]==n)[0]
        #     v[set]=F.softmax(v[set])
        #     # print(rowset)
        #     # print(a._values()[rowset])
        # v = F.dropout(v, self.dropout, training=self.training)
        sparse_matrix._values = v
        return sparse_matrix

    def forward(self, features, adj_sparse):

        features = torch.mm(features, self.weight) # features parameterized by weight
        if self.bias is not None:
            features += self.bias

        # build attention sparse matrix
        adj_idx = adj_sparse._indices() # get index set of adj

        if adj_idx.numel(): # if index set is not empty
            adj_features = torch.cat([ features[adj_idx[0],:], features[adj_idx[1],:] ], dim=1) # get adj feature pairs
            adj_v= self.actvation(torch.matmul(adj_features, self.a)) # calculate values of attention adj
            adj_v = adj_v.view(-1)

            if torch.cuda.is_available():
                atten_sparse = torch.cuda.sparse.FloatTensor(adj_idx, adj_v, adj_sparse.size()) # build attention sparse adj matrix
            else:
                atten_sparse = torch.sparse.FloatTensor(adj_idx, adj_v, adj_sparse.size()) # build attention sparse adj matrix

            # atten_sparse = self.sparse_softmax(sparse_matrix = atten_sparse, dim=0) # softmax of attention
            features = torch.spmm(atten_sparse, features)

        return features
        # a1 = atten_sparse.to_dense()


    def __repr__(self):
        return self.__class__.__name__ \
        +'('+str(self.nfeatures_in) \
        +' -> '+str(self.nfeatures_out)+')'







