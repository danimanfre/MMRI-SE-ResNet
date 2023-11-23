# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=1):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            print(grid_size)
            print(img_size[0])
            patch_size = (img_size[0] // 10 // grid_size[0], img_size[1] // 10 // grid_size[1])
            print(patch_size)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 1)
            print(patch_size_real[0])
            print(patch_size_real[1])
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.avgPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            # print("CNN input x", x.type())#(B,C,W,H)=(16,3,384,384)
            x, features = self.hybrid_model(x)
            # print("CNN output x : ",x.size())#(B,C,W,H)=(16,1024,24,24)
            # features len() = 3
            # print("skip-features[0] : ",features[0].size())#(B,C,W,H)=(16,512,48,48)
            # print("skip-features[1] : ",features[1].size())#(B,C,W,H)=(16,256,96,96)
            # print("skip-features[2] : ",features[2].size())#(B,C,W,H)=(16,64,192,192)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print("embeddings output x : ",x.size())#(B(16),768,24,24)
        # x = x.flatten(2)
        # print("flatten output x : ",x.size())#(B(16),768,576(24*24))
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)=(16,576,768)
        # embeddings = x + self.position_embeddings
        # print("embeddings output: ",self.position_embeddings.size())
        # print("embeddings output: ",embeddings.size())# (B, n_patches, hidden)=(16,576,768)

        # averaging pool here
        avg_output = self.avgPooling(x)
        avg_output = avg_output.view(avg_output.size(0), -1)
        embeddings = self.dropout(avg_output)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):#CNN+embedding後(Vision Transformer)
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        # print("hidden_states : ",hidden_states.size())#(B(16),576(24*24),768)
        for layer_block in self.layer:#12層？
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
            # print("attn_weights : ",weights)#output  None なんだろう？
        encoded = self.encoder_norm(hidden_states)
        # print("attn_weights : ",attn_weights)
        # print("attn_weights : ",len(attn_weights))
        # print("encoed hidden_states : ",hidden_states.size())#(B(16),576(24*24),768)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size+1,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states,y,features=None):
        
        # print("hidden_states : ",hidden_states.size())#(B(16),576(24*24),768)
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # print("hidden_states : ",hidden_states.size())#(B(16),576(24*24),768)
        x = hidden_states.permute(0, 2, 1)
        # print("x : ",x.size())#(B(16),768,576)
        x = x.contiguous().view(B, hidden, h, w)
        # print("x : ",x.size())#(B(16),768,24,24)
        # print("y : ",y.size())
        #====================候補1=======================
        x = torch.cat([x, y], dim=1)
        # print("z : ",x.size())#(B(16),768,24,24)
        x = self.conv_more(x)
        # print("z : ",x.size())#(B(16),768,24,24)
        # print("x : ",x.size())#(B(16),512,24,24)
        #====================候補2=======================
        #アップサンプリング開始
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #====================候補3=======================
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    
    def __init__(self, config, img_size=384, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        # we want to add a CNN that outputs a probability distribution over the 5 classes 
        self.config = config

    def forward(self,x,y):
        # print("~~~~~~~~~~~~~~~~~~~")
        # print("x = ",x.size())
        # print("y = ",y.size())
        
        
        
        if x.size()[1] == 1:#3チャンネルRGBでも1チャンネル白黒でも可にしたいだけ？
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        
        x = self.decoder(x,y, features)
        
        logits = self.segmentation_head(x)
        # the model return a tensor with size [1, 5, 384, 384] so there are 5 slices (one for each class) with dimension 384x384
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

class CNN(nn.Module):


    def __init__(self, out_channels, in_channels=5, kernel_size=3, padding=1, stride=1, use_batchnorm='True', image_size=384):
        
        super(Conv2dReLU, self).__init__()

        # Remember this formula: Output dimension = ((input dimension + 2 * padding - kernel size) / stride) + 1
        # so Let's consider an example where you apply a convolution with a kernel size of 3x3, stride of 1, 
        # and padding of 1 (assuming equal padding on both sides). Using the formula mentioned above, the output 
        # dimension will be:
        # Output dimension = ((384 + 2 * 1 - 3) / 1) + 1
        # Output dimension = (384 + 2 - 3) + 1
        # Output dimension = 384

        # If I consider a 384*384 image and 5 channels (one for each segmented part) also in_channels must be 5.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding,)
        
        # ReLU (Rectified Linear Unit) is an element-wise activation function 
        # that sets negative values to zero and leaves positive values unchanged.
        # It does not alter the shape or dimensionality of the input data.

        # When performing ReLU (Rectified Linear Unit) "in-place," 
        # it means that the activation function is applied directly
        # on the input data without creating a separate copy. 
        # In other words, the operation modifies the original input values.
        self.relu1 = nn.ReLU(inplace='True')


        

        # Batch normalization (BatchNorm) is applied independently to each channel of the input tensor. 
        # It normalizes the activations along the batch dimension by subtracting the batch mean and 
        # dividing by the batch standard deviation. This normalization process does not alter the spatial dimensions of the input.
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # The output dimension after a nn.MaxPool2d layer depends on these parameters. 
        # It can be calculated using the following formula:

        # output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding)
        self.relu2 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_channels*image_size*image_size/4, out_features=5)
		# initialize our softmax classifier for computing a probability distribution over 5 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        output = self.softmax(x)

        # A probability distribution over 5 classes
        return output

class EmbeddingClassification(nn.Module):
    def __init__(self, configs, img_size):
        super(EmbeddingClassification, self).__init__()
        self.embeddings = Embeddings(configs, img_size=img_size)
        #self.norm = nn.LayerNorm((576*768))
        # Let understand how to define the input features. It will be 576*768
        self.fc1 = nn.Linear(in_features= configs.hidden_size,out_features=5)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        flc1_output = self.fc1(embedding_output[0])

        return flc1_output
    
class MultipleEmbeddingClassification(nn.Module):
    def __init__(self, configs, img_size, embedding_dim):
        super(MultipleEmbeddingClassification, self).__init__()
        # torch.nn.Embeddingn(num_embeddings: int, embedding_dim: int)
        self.emb = torch.nn.Embedding(5, embedding_dim)
        self.embeddings = Embeddings(configs, img_size=img_size)

        # Let understand how to define the input features. It will be 576*768
        self.fc1 = nn.Linear(in_features= configs.hidden_size,out_features=5)

    def forward(self, image_raw, image_segmented):
        # shape (batch_size, 384, 384, D)
        embedding_representation = self.emb(image_segmented)
        # shape (batch_size,D, 384, 384)
        embedding_representation = embedding_representation.permute(0, 3, 1, 2)
        # from (batch_size, 384, 384) to (batch_size, 1, 384, 384)
        image_raw = torch.unsqueeze(image_raw, dim=1) 
        # create an input (batch_size,1+D, 384, 384) so stack on channel dimension (dim=1)
        input_ids = torch.cat([image_raw, embedding_representation], dim=1)
        embedding_output = self.embeddings(input_ids)
        flc1_output = self.fc1(embedding_output[0])

        return flc1_output
    
class EmbeddingRegressionSigmoid(nn.Module):
    def __init__(self, configs, img_size):
        super(EmbeddingRegressionSigmoid, self).__init__()
        self.embeddings = Embeddings(configs, img_size=img_size)
        self.flatten = nn.Flatten()
        # self.norm = nn.LayerNorm((576*768))
        # Let understand how to define the input features. It will be 576*768
        self.fc1 = nn.Linear(in_features= 576*768,out_features=1024)

        # self.dropout1 = nn.Dropout(p=0.1)
        
        self.relu1 = nn.LeakyReLU()
        #self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        
        # self.dropout2 = nn.Dropout(p=0.1)

        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=1)
        # A regression value between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        # maybe I have to flatten not starting from the first element but from the second [batch_size, channels=1?, height * width]
        # flatten_output = self.flatten(embedding_output[0])
        #flatten_output_norm = self.norm(flatten_output)
        flc1_output = self.fc1(embedding_output[0])

        # dropout1_output = self.dropout1(flc1_output)

        relu_output = self.relu1(flc1_output)
        #batch_norm_output = self.batch_norm(relu_output)
        flc2_output = self.fc2(relu_output)

        # dropout2_output = self.dropout2(flc2_output)

        relu2_output = self.relu2(flc2_output)
        flc3_output = self.fc3(relu2_output)
        output = self.sigmoid(flc3_output)
        scaled_output = output * 4
        return scaled_output
    
class EmbeddingRegression(nn.Module):
    def __init__(self, configs, img_size):
        super(EmbeddingRegression, self).__init__()
        self.embeddings = Embeddings(configs, img_size=img_size)
        self.flatten = nn.Flatten()
        # self.norm = nn.LayerNorm((576*768))
        # Let understand how to define the input features. It will be 576*768
        self.fc1 = nn.Linear(in_features= 576*768,out_features=1024)

        # self.dropout1 = nn.Dropout(p=0.1)
        
        self.relu1 = nn.LeakyReLU()
        #self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        
        # self.dropout2 = nn.Dropout(p=0.1)

        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        # maybe I have to flatten not starting from the first element but from the second [batch_size, channels=1?, height * width]
        flatten_output = self.flatten(embedding_output[0])
        #flatten_output_norm = self.norm(flatten_output)
        flc1_output = self.fc1(flatten_output)

        # dropout1_output = self.dropout1(flc1_output)

        relu_output = self.relu1(flc1_output)
        #batch_norm_output = self.batch_norm(relu_output)
        flc2_output = self.fc2(relu_output)

        # dropout2_output = self.dropout2(flc2_output)

        relu2_output = self.relu2(flc2_output)
        output = self.fc3(relu2_output)
        # no scaling of this value
        return output
    
class MyTransformer(nn.Module):


    def __init__(self, configs, img_size=384, vis=False):
        super(MyTransformer, self).__init__()
        self.transformer = Transformer(configs, img_size, vis)
        self.flatten = nn.Flatten()
        # Let understand how to define the input features. It will be 576*768
        
        self.fc1 = nn.Linear(in_features= 768,out_features=512)
        
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=5)
        # A probability distribution over 5 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        x, attn_weights, features = self.transformer(input_ids)
        # maybe I have to flatten not starting from the first element but from the second [batch_size, channels=1?, height * width]
        
        flatten_output = self.flatten(x[0])
        
        flc1_output = self.fc1(flatten_output)
        relu_output = self.relu1(flc1_output)
        flc2_output = self.fc2(relu_output)
        relu2_output = self.relu2(flc2_output)
        flc3_output = self.fc3(relu2_output)
        output = self.softmax(flc3_output)
        return output

###############################################

# Define the network with which we want to compare

# ConvolutionalBlock or IdentityBlock -> they differ only for the skip connection part
# The last parameter is used to select the type of block
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_identity=False, is_last=False):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
       
        # Shortcut connection using nn.Identity()

        # case ConvolutionalBlock
        if (not is_identity):
            self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        # case IdentityBlock
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply shortcut connection
        out += self.shortcut(x)
        out = self.relu1(out)

        return out
    

class Model3D_CNN(nn.Module):
    def __init__(self, in_channels, residual_channel, stride=1, is_binary_problem=False):
        super(Model3D_CNN, self).__init__()
        self.is_binary_problem = is_binary_problem
        # Stage 1
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # the spatial dimensions will be reduced by half in each dimension due to the stride of 2.
        self.maxPool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        # Stage 2
        self.convBlock1 = ConvolutionalBlock(in_channels=32, out_channels=64, is_identity=False)
        self.identityBlock2 = ConvolutionalBlock(in_channels=64, out_channels=64, is_identity=True)
        self.identityBlock3 = ConvolutionalBlock(in_channels=64, out_channels=64, is_identity=True)
        self.convBlock4 = ConvolutionalBlock(in_channels=64, out_channels=128, is_identity=False)
        self.identityBlock5 = ConvolutionalBlock(in_channels=128, out_channels=128, is_identity=True)
        self.identityBlock6 = ConvolutionalBlock(in_channels=128, out_channels=128, is_identity=True)
        self.dropout2 = nn.Dropout(p=0.5)

        # Stage 3
        # the spatial dimensions will be reduced by half in each dimension due to the stride of 2.
        self.globalMaxPool3 = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 1024)
        self.relu3 = nn.ReLU()
        if(not is_binary_problem):
            self.fc2 = nn.Linear(1024, 5)
        else:
            self.fc2 = nn.Linear(1024, 1)
            self.sigmoid = nn.Sigmoid()
        self.dropout3 = nn.Dropout(p=0.5)


    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Stage 1
        # (batch_size, channel, height, width)
        # (15, 1, 120, 160, 160)
        out = self.conv1(x)
        # (15, 32, 57, 77, 77)
        out = self.bn1(out)
        out = self.maxPool1(out)
        # (15, 32, 28, 38, 38)
        out = self.dropout1(out)

        # Stage 2
        out = self.convBlock1(out)
        out = self.identityBlock2(out)
        out = self.identityBlock3(out)
        out = self.convBlock4(out)
        out = self.identityBlock5(out)
        out = self.identityBlock6(out)
        out = self.dropout2(out)

        # Stage 3
        out = self.globalMaxPool3(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        if (self.is_binary_problem):
            out = self.sigmoid(out)
        return out


# LeNet architecture for basic CNN
class LeNetCNN(nn.Module):    
    def __init__(self, numChannels, classes):
		# call the parent constructor
        super(LeNetCNN, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
	    # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=50, out_features=100)
        self.relu3 = nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = Linear(in_features=100, out_features=classes)
        #self.softmax = nn.Softmax(dim=5)

    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
        x = self.fc2(x)
        #output = self.softmax(x)
		# return the output predictions
        return x

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


# Structure of the model:
#                                          VisionTransformer
#       Transformer           +            DecoderCup        +      SegmentationHead
# Embedding    +     Encoder               DecoderBlock             Conv2d  +  Upsampling
#  CNN               Blocks             Conv2d  +  Upsampling
#                 TransformationLayer