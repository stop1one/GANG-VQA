from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from gqa_dataset_entry import GQATorchDataset


class gat(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, edge_in_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(gat, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        # layer for edge and instruction vectors:
        self.lin_e = Linear(edge_in_channels, heads * out_channels, bias=False)
        self.att_e = Parameter(torch.Tensor(1, heads, out_channels))


        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_e.weight) # for edge feature
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.att_e) # for edge feature
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None


        # for edge features:
        e = self.lin_e(edge_attr).view(-1, H, C)
        alpha_e = (e * self.att_e).sum(dim=-1)


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), alpha_e=alpha_e, size=size)


        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, alpha_e,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha += alpha_e # add edge features...

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # print()
        # print(x_j.shape)
        # print(alpha_j.shape)
        # print(alpha_i.shape)
        # print(edge_attr.shape)
        # print()
        # print(alpha_j)
        # for i in range(x_j.shape[0]):
        #     print(x_j[i])

        # print(x_j)
        # print(edge_attr)




        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


import math
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerQuestionEncoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerQuestionEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp) )
        self.ninp = ninp

    def forward(self, src):

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src   = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

"""
Our core module (dga) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !!
Extremely Rough version ...
"""
class dga(torch.nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(dga, self).__init__()

        TEXT = GQATorchDataset.TEXT
        text_vocab = GQATorchDataset.TEXT.vocab
        text_emb_dim = 300 # 300d glove
        text_pad_idx = text_vocab.stoi[TEXT.pad_token]
        text_vocab_size = len(text_vocab)
        self.text_vocab_embedding = torch.nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=text_pad_idx)
        self.question_hidden_dim = 128 # 256, 79% slower # 128 - 82% on short # 512, batch size
        self.question_encoder = TransformerQuestionEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=text_emb_dim, # embedding dimension = 300
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
        )

        self.num_queries = GQATorchDataset.MAX_EXECUTION_STEP
        self.query_embed = torch.nn.Embedding(self.num_queries, ninp)
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.coarse_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        
    def forward(self, questions, gt_scene_graphs, batch):
        x = gt_scene_graphs.x # [ num_nodes, MAX_OBJ_TOKEN_LEN ]
        nodes = x[batch]
        # print(questions.size(), batch.size()) # questions: [len, batch]
        match_num = 0
        use_flags = []
        for q in questions:
            for w in q:
                if w in nodes.view(-1):
                    match_num += 1
                else:
                    q = 1 # change to <pad>
            use_flags.append(True if match_num != 0 else False)

        true_batch_size = questions.size(1)
        question_encoded = self.question_encoder(questions)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Dim]
        guided_instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=question_encoded, tgt_mask=None)
        # print("giv", guided_instr_vectors.size())

        return guided_instr_vectors, use_flags


class gat_seq(torch.nn.Module):
    """
    excute a sequence of GAT conv, BN, ReLU, and dropout layers for each instruction vector ins
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, ins_dim, num_ins,
                 dropout=0.0, gat_heads=4, gat_negative_slope=0.2, gat_bias=True):

        super(gat_seq, self).__init__()

        # 5 layers of conv with  BN, ReLU, and Dropout in between
        self.convs = torch.nn.ModuleList([gat(in_channels=in_channels+ins_dim, out_channels=out_channels, # input is h and ins concat
                 edge_in_channels=edge_attr_dim+ins_dim, # edge feature is edge_attr and instruction concat
                 heads=gat_heads, concat=False, negative_slope=gat_negative_slope, dropout=dropout, bias=gat_bias) for _ in range(num_ins)])

        # for the last output, no batch norm
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(out_channels) for _ in range(num_ins-1)]) 

        # dga module
        self.question_hidden_dim = 128 # 256, 79% slower # 128 - 82% on short # 512, batch size
        self.dga = dga(ninp=self.question_hidden_dim,
                       nhead=8,
                       nhid=4*self.question_hidden_dim,
                       nlayers=3,
                       dropout=0.1,
                       )

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index, edge_attr, instr_vectors, batch, questions, sg):

        num_conv_layers = len(self.convs)

        h = x
        for i in range(num_conv_layers):
            """
            here!!
            """
          # concat the inputs:
            # print("instr_vectors", instr_vectors.size())
            ins = instr_vectors[i] # shape: batch_size X instruction_dim
            # print("ins", ins.size())
            # print(ins)
            edge_batch = batch[edge_index[0]] # find out which batch the edge belongs to
            repeated_ins_edge = torch.zeros((edge_index.shape[1], ins.shape[-1])) # shape: num_edges x instruction_dim
            repeated_ins_edge = ins[edge_batch] # pick correct batched instruction for each edge
            edge_cat = torch.cat((edge_attr, repeated_ins_edge.to(edge_attr.device)), dim=-1) # shape: num_edges X  encode_dim+instruction_dim

            # print("batch", batch.size(), batch)
            # print("ins[batch]", ins[batch].size())
            # print(ins, ins[batch])
            guided_ins, use_flags = self.dga(questions, sg, batch)
            ins_node = torch.ones_like(guided_ins)
            for j in range(len(use_flags)):
                if use_flags: ins_node[j] = guided_ins[i][batch][j] # If possible, use guided instr vector
                else: ins_node[j] = ins[batch][j] # pick correct batched instruction for each node

            x_cat = torch.cat((h, ins_node), dim=-1) # concat the previous layer node hidden rep with the instruction vector


            # feed into the GAT:
            conv_res = self.convs[i](x=x_cat, edge_index=edge_index, edge_attr=edge_cat)
            h = conv_res + h # skip connection

            # do BN, ReLU, Droupout in-between all conv layers
            if i != num_conv_layers-1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)


        return h # return the last layer's hidden rep.

