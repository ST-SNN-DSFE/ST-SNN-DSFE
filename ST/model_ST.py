import torch.nn as nn
from layer_ST import LocalLayer, MesoLayer, GlobalLayer
import torch
from utils import normalize_adj, CrossInteraction


class ST(nn.Module):
    def __init__(self, args, local_adj, coor):
        super(ST, self).__init__()

        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout
        self.l_relu = args.lr
        self.adj = local_adj
        self.coordinate = coor

        # SNN
        self.local_snn = LocalLayer(args.in_feature * 62, 930, self.args.device)


        # Meso
        self.meso_embed = nn.Linear(5, 30)
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6, coordinate = self.coordinate, trainable_vector=78)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6, coordinate = self.coordinate, trainable_vector=54)
        self.meso_dropout = nn.Dropout(0.2)
        self.meso_conv_embed = nn.Linear(30, 10)
        self.meso_subgraph_embed = nn.Linear(30, 40)
        self.cross_interaction = CrossInteraction(in_dim=30, out_dim=40)

        # conv
        self.meso_parallel_convs = nn.ModuleList([
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])

        self.global_layer_1 = GlobalLayer(40, 50)


        # mlp
        self.mlp0 = nn.Linear(71 * 90, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        # common
        # self.layer_norm = nn.LayerNorm([30])
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout)
        # self.att_dropout = nn.Dropout(0.9)


    def forward(self, x):

        # #############################################
        # # step1:SNN
        # #############################################
        lap_matrix = normalize_adj(self.adj)
        laplacian = lap_matrix

        local_x1, local_x2 = self.local_snn(x)
        res_local = torch.cat((x, local_x1, local_x2), 2)

        if "local" not in self.args.module:
            self.args.module += "local "

        ##########################################
        #### step2:mesoscopic scale
        ##########################################
        meso_input = self.meso_embed(x)
        coarsen_x1, coarsen_coor1 = self.meso_layer_1(meso_input)
        coarsen_x1 = self.meso_subgraph_embed(coarsen_x1)
        coarsen_x1 = self.lrelu(coarsen_x1)

        coarsen_x2, coarsen_coor2 = self.meso_layer_2(meso_input)
        coarsen_x2 = self.meso_subgraph_embed(coarsen_x2)
        coarsen_x2 = self.lrelu(coarsen_x2)

        # conv
        meso_conv_results = [
            self.lrelu(conv_layer(meso_input.transpose(1, 2)).transpose(1, 2))
            for conv_layer in self.meso_parallel_convs
        ]

        fused_conv = torch.mean(torch.stack(meso_conv_results, dim=0), dim=0)

        fused_features = self.cross_interaction(fused_conv, res_local)


        res_meso = torch.cat((fused_features, coarsen_x1, coarsen_x2), 1)
        res_coor = torch.cat((self.coordinate, coarsen_coor1, coarsen_coor2), 0)

        if "meso" not in self.args.module:
            self.args.module += "meso "

        #############################################
        # step3:global scale
        #############################################

        global_x1 = self.lrelu(self.global_layer_1(res_meso, res_coor))
        res_global = torch.cat((res_meso, global_x1), 2)

        if "global" not in self.args.module:
            self.args.module += "global"

        # ############################################
        # step4:emotion recognition
        # ############################################

        x = res_global.view(res_global.size(0), -1)

        x = self.lrelu(self.mlp0(x))
        x = self.dropout(x)
        # x = self.bn(x)
        x = self.lrelu(self.mlp1(x))
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.mlp2(x)

        return x, lap_matrix, ""










