import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import graphpool

from einops import rearrange, repeat, einsum
import numpy as np

from typing import Union

############################## SNN ##############################
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    def __init__(self, state_num, action_num, device, batch_window=5, hidden1=620):
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.pseudo_spike = PseudoSpikeRect.apply
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.action_num, bias=True)


    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x, batch_size):
        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc1_sumspike = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, input_spike, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc1_sumspike += fc1_s
            fc2_sumspike += fc2_s
        out1 = fc1_sumspike / self.batch_window
        out2 = fc2_sumspike / self.batch_window
        return out1, out2

class PoissonEncoder:
    def __init__(self, spike_state_num, batch_window):
        self.spike_state_num = spike_state_num
        self.batch_window = batch_window

        # SNN
    def state_2_state_spikes(self, spike_state_value, batch_size):
        """
        Transform state to spikes of input neurons
        :param spike_state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: state_spikes
        """
        spike_state_value = spike_state_value.reshape((-1, self.spike_state_num, 1))
        state_spikes = np.random.rand(batch_size, self.spike_state_num, self.batch_window) < spike_state_value
        state_spikes = state_spikes.astype(float)
        return state_spikes

class LocalLayer(Module):
    def __init__(self, in_features, out_features, device):
        super(LocalLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.actor_net_spiking = ActorNetSpiking(self.in_features, self.out_features, self.device)

    def forward(self, input):
        batch_size = input.size(0)
        input_spread = input.view(batch_size, -1)
        input_spread_np = input_spread.cpu().numpy()
        encoder = PoissonEncoder(spike_state_num=310, batch_window=5)
        batch_size = input_spread.size(0)
        state_spikes_np = encoder.state_2_state_spikes(input_spread_np, batch_size)
        state_spikes = torch.tensor(state_spikes_np, dtype=torch.float32).to(self.device)

        output1, output2 = self.actor_net_spiking(state_spikes, batch_size)


        output_reshaped1 = output1.view(batch_size, 62, 10)
        output_reshaped2 = output2.view(batch_size, 62, 15)
        return output_reshaped1, output_reshaped2


    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features) }"
############################## SNN ##############################

############################## Global:  Time and space fusion  ##############################
class GlobalLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_feature):
        super(GlobalLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_feature
        self.num_heads = 8
        self.lrelu = nn.LeakyReLU(0.1)

        self.embed = nn.Linear(3, 40)
        self.get_qk = nn.Linear(self.in_features, self.in_features * 2)

        self.equ_weights = Parameter(torch.FloatTensor(self.num_heads))
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def forward(self, h, res_coor):
        h_with_embed = h + self.lrelu(self.embed(res_coor))

        attention_value = self.cal_att_matrix(h, h_with_embed)
        output = torch.matmul(attention_value, self.weight) + self.bias

        return output

    def cal_att_matrix(self, feature, feature_with_embed):
        out_feature = []
        batch_size, N = feature.size(0), feature.size(1)

        qk = rearrange(self.get_qk(feature_with_embed), "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys= qk[0], qk[1]
        values = feature

        dim_scale = (queries.size(-1)) ** -0.5
        dots = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * dim_scale

        attn = torch.einsum("b g i j -> b i j", dots)
        adj_matrix = self.dropout_80_percent(attn)
        # attn = self.attend(adj_matrix)
        attn = F.softmax(adj_matrix/0.3, dim=2)

        out_feature = torch.einsum('b i j, b j d -> b i d', attn, values)

        return out_feature


    def dropout_80_percent(self, attn):
        att_subview_, _ = attn.sort(2, descending=True)

        att_threshold = att_subview_[:, :, att_subview_.size(2) // 6]
        att_threshold = rearrange(att_threshold, 'b i -> b i 1')
        att_threshold = att_threshold.repeat(1, 1, attn.size()[2])
        attn[attn<att_threshold] = -1e-7
        return attn


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.equ_weights.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features)}"
############################## Global:  Time and space fusion  ##############################


############################## Meso ##############################
class MesoLayer(nn.Module):
    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super(MesoLayer, self).__init__()
        self.subgraph_num = subgraph_num
        self.coordinate = coordinate

        self.lrelu = nn.LeakyReLU(0.1)
        self.graph_list = self.sort_subgraph(subgraph_num)
        self.emb_size = 30

        self.softmax = nn.Softmax(dim=0)
        self.att_softmax = nn.Softmax(dim=1)

        self.trainable_vec = Parameter(torch.FloatTensor(trainable_vector))
        self.weight = Parameter(torch.FloatTensor(self.emb_size, 10))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.trainable_vec.size(0))
        self.trainable_vec.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        coarsen_x, coarsen_coor = self.att_coarsen(x)
        return coarsen_x, coarsen_coor

    def att_coarsen(self, features):
        features = graphpool.feature_trans(self.subgraph_num, features)
        coordinates = graphpool.location_trans(self.subgraph_num, self.coordinate)
        coarsen_feature, coarsen_coordinate = [], []

        idx_head = 0
        for index_length in self.graph_list:
            idx_tail = idx_head + index_length
            sub_feature = features[:, idx_head:idx_tail]
            sub_coordinate = coordinates[idx_head:idx_tail]

            feature_with_weight = torch.einsum('b j g, g h -> b j h', sub_feature, self.weight)
            feature_T = rearrange(feature_with_weight, 'b j h -> b h j')
            att_weight_matrix = torch.einsum('b j h, b h i -> b j i', feature_with_weight, feature_T)

            att_weight_vector = torch.sum(att_weight_matrix, dim=2)

            att_vec = self.att_softmax(att_weight_vector)

            sub_feature_ = torch.einsum('b j, b j g -> b g', att_vec, sub_feature)
            sub_coordinate_ = torch.einsum('b j, j g -> b g', att_vec,
                                           sub_coordinate)
            sub_coordinate_ = torch.mean(sub_coordinate_, dim=0)

            coarsen_feature.append(rearrange(sub_feature_, "b g -> b 1 g"))
            coarsen_coordinate.append(rearrange(sub_coordinate_, "g -> 1 g"))
            idx_head = idx_tail

        coarsen_features = torch.cat(tuple(coarsen_feature), 1)
        coarsen_coordinates = torch.cat(tuple(coarsen_coordinate), 0)
        return coarsen_features, coarsen_coordinates

    def sort_subgraph(self, subgraph_num):
        subgraph_7 = [5, 9, 9, 25, 9, 9, 12]
        subgraph_4 = [6, 6, 4, 6]
        subgraph_2 = [27, 27]

        graph_list = None
        if subgraph_num == 7:
            graph_list = subgraph_7
        elif subgraph_num == 4:
            graph_list = subgraph_4
        elif subgraph_num == 2:
            graph_list = subgraph_2

        return graph_list
############################## Meso ##############################
