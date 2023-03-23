from typing import Optional, Any
import math
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from pre_train.promp_mask import geom_noise_mask_single, inter_prompt_learning_mask, iterable_prompt_mask

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output

class TSTransformerEncoder_Fed(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder_Fed, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output

    def get_state(self):
        return self.state_dict(), []

    def set_state(self, w_server, w_local):
        self.load_state_dict(w_server)

class TSTransformerEncoder_Fed_Pre(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, model_dict=None):
        super(TSTransformerEncoder_Fed_Pre, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.feat_dim = feat_dim
        self.model_dict = model_dict

        self.Transformer_backbone = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, pos_encoding=pos_encoding, activation=activation, norm=norm)

        if self.model_dict['whether_prompt'] == 'prompt_learning':
            self.prompting_transformer = TSTransformerEncoder(feat_dim, self.model_dict['ipt_len'], 
                                                            d_model // 2, n_heads // 2, num_layers // 2, dim_feedforward // 2, pos_encoding=pos_encoding, activation=activation, norm=norm)

            self.fake_prompting_transformer = TSTransformerEncoder(feat_dim, self.model_dict['fore_len']  - self.model_dict['ipt_len'], 
                                                            d_model // 2, n_heads // 2, num_layers // 2, dim_feedforward // 2, pos_encoding=pos_encoding, activation=activation, norm=norm)

        elif self.model_dict['whether_prompt'] == 'Prompt_learning_interact':
            # Normal_unsup we can name it as supervised learning without any pormpting actions
            self.Transformer_prompt_pre = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, pos_encoding=pos_encoding, activation=activation, norm=norm)
            # self.prompt_en = TSTransformerEncoder(feat_dim, self.model_dict['fore_len'] + self.model_dict['pro_len'], d_model, n_heads * 16, num_layers // 2, dim_feedforward // 2, pos_encoding=pos_encoding, activation=activation, norm=norm)
            self.prompt_parap = torch.nn.Parameter(torch.FloatTensor(model_dict['bs'], 32, model_dict['vars']), requires_grad=True)
            self.prompt_parap_inter = torch.nn.Parameter(torch.FloatTensor(model_dict['bs'], 27, model_dict['vars']), requires_grad=True)
            self.prompt_xxx = torch.nn.Parameter(torch.FloatTensor(model_dict['bs'], 6, model_dict['vars']), requires_grad=True)
            self.prompt_final = torch.nn.Parameter(torch.FloatTensor(model_dict['bs'], 15, model_dict['vars']), requires_grad=True)
            stdv = 1. / math.sqrt(self.prompt_parap.shape[1])
            stdv_f = 1./ math.sqrt(self.prompt_final.shape[1])
            self.prompt_parap.data.uniform_(-stdv, stdv)
            self.prompt_parap_inter.data.uniform_(-stdv, stdv)
            self.prompt_xxx.data.uniform_(-stdv, stdv)
            self.prompt_final.data.uniform_(-stdv, stdv)
        
        elif self.model_dict['whether_prompt'] == "Novel_Prompting":
            logger.info("{}".format(self.model_dict['whether_prompt']))
            self.Transformer_prompt_pre = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, pos_encoding=pos_encoding, activation=activation, norm=norm)
            
            self.prompt_token = torch.nn.Parameter(torch.empty(model_dict['bs'], 3, model_dict['vars']), requires_grad=True)
            self.prompt_token2 = torch.nn.Parameter(torch.empty(model_dict['bs'], 6, model_dict['vars']), requires_grad=True)
            self.prompt_token3 = torch.nn.Parameter(torch.empty(model_dict['bs'], 6, model_dict['vars']), requires_grad=True)
            self.prompt_token4 = torch.nn.Parameter(torch.empty(model_dict['bs'], model_dict['vars']), requires_grad=True)

            self.prompt_horzion = torch.nn.Parameter(torch.empty(model_dict['bs'], 15, 1), requires_grad=True)
            self.prompt_last = torch.nn.Parameter(torch.empty(model_dict['bs'], 15, 1), requires_grad=True)
            # print(self.prompt_last)

            stdv = 1. / math.sqrt(self.prompt_token.shape[1])
            stdv3 =  1. / math.sqrt(self.prompt_token3.shape[1])
            stdv4 = 1. / math.sqrt(self.prompt_token4.shape[1])
            self.drop = nn.Dropout(p=0.5)
            self.prompt_token.data.uniform_(-stdv, stdv)
            self.prompt_token2.data.uniform_(-stdv3, stdv3)
            self.prompt_token3.data.uniform_(-stdv3, stdv3)
            self.prompt_token3.data.uniform_(-stdv4, stdv4)
            self.prompt_last.data.uniform_(-stdv4, stdv4)

            self.prompt_horzion.data.uniform_(-stdv4, stdv4)

            self.pre_head = nn.LayerNorm((128, 15, 12))
            self.head = nn.Linear(feat_dim * (self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']),  (self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']))
            
            self.pro_hor_fake = torch.nn.Parameter(torch.zeros(128, 15, 1), requires_grad=False)
 
        elif self.model_dict['whether_prompt'] == "Non_Interaction_Prompt":
            self.prompt_en = TSTransformerEncoder(feat_dim, self.model_dict['fore_len'] + self.model_dict['pro_len'], d_model, n_heads * 16, num_layers // 2, dim_feedforward // 2, pos_encoding=pos_encoding, activation=activation, norm=norm)

        elif self.model_dict['whether_prompt'] == "Iterable_Prompt":
            self.prompt_en = TSTransformerEncoder(feat_dim, self.model_dict['fore_len'] + self.model_dict['pro_len'], d_model, n_heads * 16, num_layers // 2, dim_feedforward // 2, pos_encoding=pos_encoding, activation=activation, norm=norm)

        elif self.model_dict['whether_prompt'] == "fine_tuning_transformer":
            self.fine_tune = TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes = np.square(self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']), \
                                                                 pos_encoding=pos_encoding, activation=activation, norm=norm)

        elif self.model_dict['whether_prompt'] == "Normal_Prompting":
            self.Transformer_prompt_pre = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, pos_encoding=pos_encoding, activation=activation, norm=norm)
            self.prompt_para = torch.nn.Parameter(torch.empty(128, 15, 12), requires_grad=True)
            stdv = 1. / math.sqrt(self.prompt_para.shape[1])
            self.prompt_para.data.uniform_(-stdv, stdv)
            self.pre_head = nn.LayerNorm((128, 15, 12))
            self.head = nn.Linear(12 * (self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']),  (self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']))

        if self.model_dict['Pretrained_II'] == True:
            if self.model_dict['dataset'] == "Percipition":
                logger.info("HAVE BEEN LOADED")
                self.Transformer_backbone.load_state_dict(torch.load(model_dict['Per_modelpath'])['state_dict'], False)
                if self.model_dict['whether_prompt'] == 'Novel_Prompting':
                    self.Transformer_prompt_pre.load_state_dict(torch.load(model_dict['Per_modelpath'])['state_dict'], False)

            elif self.model_dict['dataset'] == "Surface_temp":
                logger.info("pretrain LOADDDD")
                self.Transformer_backbone.load_state_dict(torch.load(model_dict['Sur_modelpath'])['state_dict'], False)
                if self.model_dict['whether_prompt'] == 'Novel_Prompting' or self.model_dict['whether_prompt'] == "Normal_Prompting":
                    self.Transformer_prompt_pre.load_state_dict(torch.load(model_dict['Sur_modelpath'])['state_dict'], False)

            elif self.model_dict['dataset'] == "Upstream_flux":
                logger.info("pretrain LOADDDD")
                self.Transformer_backbone.load_state_dict(torch.load(model_dict['Up_modelpath'])['state_dict'], False)
                if self.model_dict['whether_prompt'] == 'Novel_Prompting' or self.model_dict['whether_prompt'] == "Normal_Prompting":
                    self.Transformer_prompt_pre.load_state_dict(torch.load(model_dict['Up_modelpath'])['state_dict'], False)

    def mask_process(self, prompt):
        prompt_mask = []
        for k in range(prompt.shape[0]):
            mask = inter_prompt_learning_mask(prompt[k, :, :].detach().cpu().numpy(),
                                                masking_ratio=0.15,
                                                lm=3,
                                                i_len=self.model_dict['ipt_len'],
                                                f_len=self.model_dict['fore_len'],
                                                prompt_len=self.model_dict['pro_len'],
                                                mode=self.model_dict['prompt_app'])

                # Problem of length
            mask = torch.from_numpy(mask).cuda()
            prompt_mask.append(mask)

        prompt_mask = torch.stack(prompt_mask)

        return prompt_mask

    def prompt_em(self, x):
        B = x.shape[0]
        po = LearnablePositionalEncoding(d_model=x.shape[2], max_len=x.shape[1]).cuda()
        x = x.permute(1, 0, 2)
        pos = po(x)
        return pos.permute(1, 0, 2)

    def forward(self, X, padding_masks):

        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
            
        """

        if self.model_dict['whether_prompt'] != 'Prompt_learning_interact' or self.model_dict['whether_prompt'] != 'Non_Interaction_Prompt' or self.model_dict['whether_prompt'] != 'Interable_Prompt':
            if self.model_dict['whether_prompt'] == "fine_tuning_transformer":
                opt = self.fine_tune(X, padding_masks)
            elif self.model_dict['whether_prompt'] == "Novel_Prompting":
                X = X
            else:
                opt = self.Transformer_backbone(X, padding_masks)
        else:
            opt = self.Transformer_prompt_pre(X, padding_masks)

        if self.model_dict['whether_prompt'] == 'Novel_Prompting':
            """
            Stage I
            """
            prompt_token_pre = X[:, self.model_dict['ipt_len'] - self.model_dict['pro_len'] : self.model_dict['ipt_len'], :]
            
            prompt_token1 = torch.cat([prompt_token_pre, self.prompt_token], dim=1)
            cls_token1 = X[:, :self.model_dict['ipt_len'], :]

            round_ipt = torch.cat([
                cls_token1,
                # self.drop(self.prompt_em(prompt_token1))
                self.prompt_token
            ], dim=1)

            round_opt = self.Transformer_backbone(round_ipt, padding_masks[:, :round_ipt.shape[1]])

            # 0 -15

            # Stage II
            cls_token2 = round_opt[:, 2 * self.model_dict['pro_len'] : self.model_dict["ipt_len"], :]
            prompt_token2 = torch.cat([self.prompt_em(prompt_token1), self.prompt_em(self.prompt_token2)], dim=1)


            round2_ipt = torch.cat([
                X[:, :2 * self.model_dict['pro_len']],
                cls_token2,
                prompt_token2
            ], dim=1)

            round2_opt = self.Transformer_backbone(round2_ipt, padding_masks[:, :round2_ipt.shape[1]])
            # 0 - 21

            # Stage III
            cls_token3 = round2_opt[:, self.model_dict['ipt_len']:self.model_dict['pro_len'] + self.model_dict['ipt_len'], :]
            cls_token3 = torch.mul(cls_token3, self.prompt_token)
            prompt_token3 = torch.cat([self.prompt_token2, self.prompt_token3], dim=1)
            # mask_ii = self.mask_process(prompt_token3)
            # prompt_token3 = mask_ii * prompt_token3

            round3_ipt = torch.cat([
                X[:, :self.model_dict['ipt_len'], :],
                cls_token3,
                prompt_token3
            ], dim=1)

            round3_opt = self.Transformer_backbone(round3_ipt, padding_masks[:, :round3_ipt.shape[1]])

            whole_prompt = torch.cat([self.prompt_token, self.prompt_token2, self.prompt_token3], dim=1)


            dis = torch.cat([round_opt[:, 12:15] + round2_opt[:, 12:15] + round3_opt[:, 12:15], 
                            round2_opt[:, 15:21] + round3_opt[:, 15:21], round3_opt[:, 21:27]], dim=1)
            whole_prompt = torch.mul(whole_prompt, torch.tanh(dis))
            whole_mask = self.mask_process(whole_prompt)
            cls_to = torch.cat([
                X[:, :self.model_dict['ipt_len'], :],
                whole_prompt * whole_mask
            ], dim=1)

            opt = self.Transformer_backbone(cls_to, padding_masks[:, :cls_to.shape[1]])

            prompt_horzion = self.prompt_horzion
 
            for k in range(self.feat_dim):
                if k == 0:

                    prompt_horzion = torch.cat([prompt_horzion, self.pro_hor_fake.expand(-1, -1, self.feat_dim - 1)], dim=2)
                    hi = torch.cat([
                        X[:, :self.model_dict['ipt_len']],
                        prompt_horzion,
                    ], dim=1)

                    hi_opt = self.Transformer_backbone(hi, padding_masks[:, :hi.shape[1]])
                    prompt_horzion = torch.mul(hi_opt[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], 0].unsqueeze(2), prompt_horzion)
            
                else:
                    if k == 11:
                        prompt_horzion = torch.cat([prompt_horzion, self.prompt_last], dim=2)
                    elif k == 10:
                        prompt_horzion = torch.cat([prompt_horzion[:, :, :k + 1], self.prompt_last, self.pro_hor_fake], dim=2)
                    else:
                        # print(k)
                        prompt_horzion = torch.cat([prompt_horzion[:, :, :k + 1], self.prompt_last, self.pro_hor_fake.expand(-1, -1, self.feat_dim - (prompt_horzion[:, :, :k + 1].shape[2] + 1))], dim=2)
                   
                    # print(prompt_horzion.shape)
                    hi = torch.cat([
                        X[:, :self.model_dict['ipt_len']],
                        prompt_horzion
                    ], dim=1)

                    hi_opt = self.Transformer_backbone(hi, padding_masks[:, :hi.shape[1]])

            opt_1 = torch.tanh(opt[:, 12:]) * (1 - torch.sigmoid(prompt_horzion))
            opt = opt_1.reshape(opt.shape[0], -1)
            opt = self.head(opt)
            opt = opt.reshape(opt.shape[0], (self.model_dict['fore_len'] + self.model_dict['pro_len'] - self.model_dict['ipt_len']), -1)
            opt = opt.squeeze()

        elif self.model_dict['whether_prompt'] == "Prompt_learning_interact":

            unprompt_ipt = X[:, :self.model_dict['ipt_len'], :]
           
            # opt = torch.mul(opt, self.prompt_parap)
            inter_promoting_ipt = X[:, self.model_dict['ipt_len']: (self.model_dict['fore_len'] + self.model_dict['pro_len']) - 6, :]
            inter_promoting_ipt = torch.cat([opt, inter_promoting_ipt], dim=1)
            print(inter_promoting_ipt)

            prompt_mask = []
            for k in range(inter_promoting_ipt.shape[0]):
                mask = inter_prompt_learning_mask(inter_promoting_ipt[k, :, :].detach().cpu().numpy(),
                                                masking_ratio=0.35,
                                                lm=3,
                                                i_len=self.model_dict['ipt_len'],
                                                f_len=self.model_dict['fore_len'],
                                                prompt_len=self.model_dict['pro_len'],
                                                mode=self.model_dict['prompt_app'])

                # Problem of length
                mask = torch.from_numpy(mask).cuda()
                prompt_mask.append(mask)

            prompt_mask = torch.stack(prompt_mask)
            prompt_ii = inter_promoting_ipt * prompt_mask
            # print(prompt_ii)
            prompt_ii = torch.mul(prompt_ii, self.prompt_final)
      

            Prompt_1st = prompt_ii[:, (self.model_dict['fore_len'] - self.model_dict['ipt_len']):(self.model_dict['fore_len'] - self.model_dict['ipt_len']) + self.model_dict['pro_len'], :]
            prompt_2sd = prompt_ii[:, :self.model_dict['fore_len'] - self.model_dict['ipt_len'], :]

            Prompt_I = torch.cat([unprompt_ipt, opt[:, self.model_dict['ipt_len']: self.model_dict['fore_len'], :], Prompt_1st], dim=1)
            Prompt_II = torch.cat([unprompt_ipt, prompt_2sd, opt[:, self.model_dict['fore_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], :]], dim=1)
            Prompt_Inter_I = torch.mul(Prompt_I, self.prompt_parap_inter)
            Prompt_Inter_II = torch.mul(Prompt_II, self.prompt_parap_inter)



            final_proj = torch.cat([unprompt_ipt, 
                                    Prompt_Inter_II[:, :self.model_dict['fore_len'] - self.model_dict['ipt_len'], :], 
                                    Prompt_Inter_I[:, (self.model_dict['fore_len'] - self.model_dict['ipt_len']):(self.model_dict['fore_len'] - self.model_dict['ipt_len']) + self.model_dict['pro_len'], :]], dim=1)

            final_proj_mid = final_proj[:, self.model_dict['ipt_len'] : (self.model_dict['fore_len'] + self.model_dict['pro_len']), :] * ~prompt_mask
            final_proj_mid = torch.cat([unprompt_ipt, final_proj_mid], dim=1)

            prompt_proj = self.Transformer_backbone(final_proj, padding_masks[:, :final_proj.shape[1]])
            prompt_proj_mid = self.Transformer_backbone(final_proj_mid, padding_masks[:, :final_proj.shape[1]])

            prompt_proj = torch.tanh(prompt_proj + prompt_proj_mid) * (1 - torch.sigmoid(prompt_proj_mid))

            # Work
            prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] = \
                            torch.sigmoid(opt[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1]) * \
                            torch.tanh(prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

            opt = prompt_proj

        elif self.model_dict['whether_prompt'] == "Normal_FL_Pretrain":
            opt = opt

        elif self.model_dict['whether_prompt'] == "Non_Interaction_Prompt":
            unprompt_ipt = X[:, :self.model_dict['ipt_len'], :]

            inter_promoting_ipt = opt[:, self.model_dict['ipt_len'] : (self.model_dict['fore_len'] + self.model_dict['pro_len']), :]

            prompt_mask = []
            for k in range(inter_promoting_ipt.shape[0]):
                mask = inter_prompt_learning_mask(inter_promoting_ipt[k, :, :].detach().cpu().numpy(),
                                                masking_ratio=0.35,
                                                lm=3,
                                                i_len=self.model_dict['ipt_len'],
                                                f_len=self.model_dict['fore_len'],
                                                prompt_len=self.model_dict['pro_len'],
                                                mode=self.model_dict['prompt_app'])

                # Problem of length
                mask = torch.from_numpy(mask).cuda()
                prompt_mask.append(mask)
            
            prompt_mask = torch.stack(prompt_mask)
            prompt_ii = inter_promoting_ipt * prompt_mask

            pro_mid = torch.cat([unprompt_ipt, prompt_ii], dim=1)
            prompt = self.prompt_en(pro_mid, padding_masks[:, :pro_mid.shape[1]])

            # Fianl Proj
            prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] = \
                prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] *\
                    torch.zeros_like(prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], dtype=bool)

            prompt_proj = self.Transformer_backbone(prompt, padding_masks[:, :prompt.shape[1]])

            prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] = \
                            torch.sigmoid(opt[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1]) * \
                            torch.tanh(prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
            opt = prompt_proj

        elif self.model_dict['whether_prompt'] == "Iterable_Prompt":

            unprompt_ipt = X[:, :self.model_dict['ipt_len'], :]
            inter_promoting_ipt = opt[:, self.model_dict['ipt_len'] : (self.model_dict['fore_len'] + self.model_dict['pro_len']), :]

           
            total_prompt_mask = []
            for k in range(inter_promoting_ipt.shape[0]):
                prompt_mask = []
                for j in range(inter_promoting_ipt.shape[2]):
                    mask = iterable_prompt_mask(inter_promoting_ipt[k, :, :], masking_ratio=0.15, lm=3, feature_idx=j)

                    # mask = torch.from_numpy(mask).cuda()
                    prompt_mask.append(mask)
                    # prompt_mask = torch.from_numpy(prompt_mask)
                
                # print(torch.from_numpy(np.array(prompt_mask)).shape)
                total_prompt_mask.append(torch.from_numpy(np.array(prompt_mask)).cuda())
                # prompt_mask = torch.cat([prompt_mask, prompt_mask], 0)
            
            total_prompt_mask = torch.stack(total_prompt_mask)

            for tx_idx in range(inter_promoting_ipt.shape[2]):
                prompt_ii = inter_promoting_ipt * total_prompt_mask[:, tx_idx, :, :]
                pro_mid = torch.cat([unprompt_ipt, prompt_ii], dim=1)
                prompt = self.prompt_en(pro_mid, padding_masks[:, :pro_mid.shape[1]])

                pro_mid = prompt

            # Fianl Proj
            prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] = \
                prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] *\
                    torch.zeros_like(prompt[:, self.model_dict['ipt_len']:self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], dtype=bool)

            prompt_proj = self.Transformer_backbone(prompt, padding_masks[:, :prompt.shape[1]])

            prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1] = \
                            torch.sigmoid(opt[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1]) * \
                            torch.tanh(prompt_proj[:, self.model_dict['ipt_len']: self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
            opt = prompt_proj

        return opt

    def get_state(self):
        return self.state_dict(), []

    def set_state(self, w_server, w_local):
        self.load_state_dict(w_server)

def freezex(layer_name, model):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = False

