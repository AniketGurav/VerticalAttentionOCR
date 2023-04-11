#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import sys
import random
import os
import time
import torch
import torch.nn as nn
from torch import tanh, log_softmax, softmax, relu
import torch.nn.functional as F

from torch.nn import Conv1d, Conv2d, Dropout,  Linear, AdaptiveMaxPool2d, InstanceNorm1d, AdaptiveMaxPool1d
from torch.nn import Flatten, LSTM, Embedding
from basic.models import DepthSepConv2D


class VerticalAttention(nn.Module):
    def __init__(self, params):
        
        #print("\n\t verticle attention reloaded!!!")
        
        super(VerticalAttention, self).__init__()
        self.att_fc_size = params["att_fc_size"] # 256
        self.features_size = params["features_size"] # 256
        self.use_location = params["use_location"] 
        self.use_coverage_vector = params["use_coverage_vector"]
        self.coverage_mode = params["coverage_mode"]
        self.use_hidden = params["use_hidden"]
        self.min_height = params["min_height_feat"] # 15
        self.min_width = params["min_width_feat"] # 100
        self.stop_mode = params["stop_mode"] 

        self.ada_pool = AdaptiveMaxPool2d((None, self.min_width)) # 100
        self.dense_width = Linear(self.min_width, 1) # 100

        self.dense_enc = Linear(self.features_size, self.att_fc_size) # 256 -> 256
        self.dense_align = Linear(self.att_fc_size, 1) # 256 -> 1

        if self.stop_mode == "learned":
            self.ada_pool_height = AdaptiveMaxPool1d(self.min_height)
            self.conv_decision = Conv1d(self.att_fc_size, self.att_fc_size, kernel_size=5, padding=2) #
            self.dense_height = Linear(self.min_height, 1)
            if self.use_hidden:
                self.dense_decision = Linear(params["hidden_size"]+self.att_fc_size, 2)
            else:
                self.dense_decision = Linear(self.att_fc_size, 2)
        in_ = 0
        if self.use_location:
            in_ += 1
        if self.use_coverage_vector:
            in_ += 1

        self.norm = InstanceNorm1d(in_, track_running_stats=False)
        self.conv_block = Conv1d(in_, 16, kernel_size=15, padding=7)
        self.dense_conv_block = Linear(16, self.att_fc_size)

        if self.use_hidden:
            self.hidden_size = params["hidden_size"]
            self.dense_hidden = Linear(self.hidden_size, self.att_fc_size)

        self.dropout = Dropout(params["att_dropout"])

        self.h_features = None

    def forward(self, features, prev_attn_weights, coverage_vector=None, hidden=None, status="init"):
        """
        features (B, C, H, W)
        h_features (B, C, H)
        coverage_vector (B, H)
        hidden (num_layers, B, hidden_size)
        prev_att_weights (B, H)
        returns context_vector (B, C, W), att_weights (B, H)
        """

        print("\n\t inside verticle attention!!!!")
        print("\n\t 1. features.size():",features.size(),"\t status:",status) # features.size(): torch.Size([8, 256, 25, 138])
        
        """
        Since we want to normalize the attention
        weights over the vertical axis, we must collapse the horizontal axis
        of the features. The idea is that we only need to know if there is
        at least one character in a features row to decide to process it as a
        text line. Thus, we can collapse this horizontal dimension without
        information loss. This collapse is carried out in two steps: we first
        get back to a fixed width of 100 (since inputs are of variable sizes)
        through AdaptiveMaxPooling. Then, a densely connected layer
        pushes the horizontal dimension to collapse. The remaining vertical
        representation is called f
        0 ∈ R
        Hf ×Cf
        """
        
        
        if status == "reset":
            self.h_features = self.h_features.detach()
        if status in ["init", "reset", ]:
            self.h_features = self.ada_pool(features) # AdaptiveMaxPooling 
            self.h_features = self.dense_width(self.h_features).squeeze(3) # densely connected layer pushes the horizontal dimension to collapse

        print("\n\t 2. features.size():",features.size(),"\t self.h_features.shape:",self.h_features.shape) # features.size(): torch.Size([8, 256, 25, 138])
        # self.h_features.permute(0, 2, 1): torch.Size([8, 25, 256])
        b, c, h, w = features.size()
        device = features.device
        sum = torch.zeros((b, h, self.att_fc_size), dtype=features.dtype, device=device) # torch.Size([8, 25, 256])
        cat = list()

        print("\n\t self.att_fc_size:",self.att_fc_size,"\t sum:",sum.shape," \t prev_attn_weights.shape:",prev_attn_weights.shape)
        # self.att_fc_size: 256 	 sum: torch.Size([8, 25, 256])  	 prev_attn_weights.shape: torch.Size([8, 25])
        
        """
        cat keeps tracks of all previous attention weights
        """
        if self.use_location:
            cat.append(prev_attn_weights)
        if self.use_coverage_vector:
            if self.coverage_mode == "clamp":
                cat.append(torch.clamp(coverage_vector, 0, 1))
            else:
                cat.append(coverage_vector) 
        # here cat is now attention and contex vector iₜ ???????
        print("\n\t self.h_features.permute(0, 2, 1):",self.h_features.permute(0, 2, 1).shape)
        # self.h_features.permute(0, 2, 1): torch.Size([8, 25, 256])
        
        temp = self.dropout(self.dense_enc(self.h_features.permute(0, 2, 1)))
        
        print("\n\t temp.shape:",temp.shape) #  temp.shape: torch.Size([8, 25, 256])
        sum += temp

        cat = torch.cat([c.unsqueeze(1) for c in cat], dim=1)
        cat = self.norm(cat)
        
        print("\n\t cat befor  =",cat.shape)
        cat = self.conv_block(cat)
        catPer= cat.permute(0, 2, 1).shape
        print("\n\t cat[0]:",cat[0].shape,"\t permute:",catPer)
        # 	 cat[0]: torch.Size([16, 25]) 	 permute: torch.Size([8, 25, 16])
        
        catOut = self.dropout(self.dense_conv_block(cat.permute(0, 2, 1))) 
        sum += self.dropout(self.dense_conv_block(cat.permute(0, 2, 1)))
        print("\n\t 1.sum:",sum.shape,"\t catOut.shape:",catOut.shape) # 1.sum: torch.Size([8, 25, 256])
        # 	 1.sum: torch.Size([8, 25, 256]) 	 catOut.shape: torch.Size([8, 25, 256])

        
        
        if self.use_hidden:
            
            try:
                print("\n\t hidden.shape:",hidden.shape)
            except Exception as e:
                pass
            
            try:
                print("\t hidden[0]:",hidden[0].shape)
            except Exception as e:
                pass
            
            temp1 = self.dropout(self.dense_hidden(hidden[0]).permute(1, 0, 2))
            
            print("\n\t temp1.shape =",temp1.shape) # temp1.shape = torch.Size([8, 1, 256])
            sum += temp1 

        sum = tanh(sum)
        print("\n\t 2.sum:",sum.shape) #	 2.sum: torch.Size([8, 25, 256]) it is Sₜ


        align_score = self.dense_align(sum) # channel collapse
        
        print("\n\t align_score.shape:",align_score.shape) # align_score.shape: torch.Size([8, 25, 1])  
        
        attn_weights = softmax(align_score, dim=1)  # it is αₜ
        
        f1 = features.permute(0, 1, 3, 2)
        a1 = attn_weights.unsqueeze(1)
        print("\n\t inside matmul shapes f1.shape:",f1.shape,"\t a1.shape:",a1.shape)
        # 	 inside matmul shapes f1.shape: torch.Size([1, 256, 138, 25]) 	 a1.shape: torch.Size([1, 1, 25, 1])

        
        context_vector = torch.matmul(features.permute(0, 1, 3, 2), attn_weights.unsqueeze(1)).squeeze(3) # Cₜ or Lₜ ???  

        print("\n\t cv1.shape:",context_vector.shape) # cv1.shape: torch.Size([8, 256, 138])
        m1 = features.permute(0, 1, 3, 2)
        m2 = attn_weights.unsqueeze(1)
        attSqz = attn_weights.squeeze(2)
        print("\n\t m1.shape:",m1.shape,"\t m2.shape:",m2.shape) # 	 m1.shape: torch.Size([8, 256, 138, 25]) 	 m2.shape: torch.Size([8, 1, 25, 1])
        print("\n\t attSqz =",attSqz.shape) # 	 attSqz = torch.Size([8, 25])
        decision = None
        
        if self.stop_mode == "learned":
            sum = relu(self.conv_decision(sum.permute(0, 2, 1)))
            decision = relu(self.dense_height(self.ada_pool_height(sum))).squeeze(2)
            if self.use_hidden:
                decision = torch.cat([hidden[0].squeeze(0), decision], dim=1)
            decision = self.dropout(decision)
            decision = self.dense_decision(decision)

        return context_vector, attn_weights.squeeze(2), decision


class LineDecoderCTC(nn.Module):
    def __init__(self, params):
        super(LineDecoderCTC, self).__init__()

        self.params = params
        self.use_hidden = params["use_hidden"]
        self.input_size = params["features_size"]
        self.vocab_size = params["vocab_size"]

        
        if self.use_hidden:
            self.hidden_size = params["hidden_size"]
            self.lstm = LSTM(self.input_size, self.hidden_size, num_layers=1)
            self.end_conv = Conv2d(in_channels=self.hidden_size, out_channels=self.vocab_size + 1, kernel_size=1)
        else:
            self.end_conv = Conv2d(in_channels=self.input_size, out_channels=self.vocab_size + 1, kernel_size=1)

    def forward(self, x, h=None):
        
        
        print("\n\t input to decoder shape!!",x.shape)
        
        print("\n\t inside LineDecoderCTC !!!")
        
        print("\n\t 1.",self.use_hidden) # 1. True
        print("\n\t 2--<",self.input_size) # 
        print("\n\t 3--<",self.vocab_size)
        
        print("\n\t 4--<",self.params["hidden_size"])
        #print("\n\t 5 self.hidden_size=",self.hidden_size)
        print("\n\t 6 self.vocab_size=",self.vocab_size)
        
        """

    	 1. True

	     2--< 256

    	 3--< 79

    	 4--< 256

    	 6 self.vocab_size= 79
        
        """
        
        
        """
        x (B, C, W)
        """
        if self.use_hidden:
            x, h = self.lstm(x.permute(2, 0, 1), h)
            x = x.permute(1, 2, 0)

        out = self.end_conv(x.unsqueeze(3)).squeeze(3)
        out = torch.squeeze(out, dim=2)
        out = log_softmax(out, dim=1)
        return out, h

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, drop=0.3):
        super().__init__()

        self.num_layers = num_layers
        self.rnn = nn.GRU(embed_size, embed_size, num_layers)
        if self.num_layers > 1: self.rnn.dropout = drop

    def forward(self, hidden, context):
        _, h = self.rnn(context.unsqueeze(0), hidden.expand(self.num_layers, -1, -1).contiguous())

        return h[-1]

class DeepOutputLayer(nn.Module):
    def __init__(self, embed_size, vocab_size, drop=0.3):
        super().__init__()
        
        self.l1 = nn.Linear(embed_size*2, embed_size)
        self.l2 = nn.Linear(embed_size, vocab_size)
        self.drop = nn.Dropout(drop)
        
    def forward(self, hidden, context):
        # this is called once for each timestep
        #(30,256)
        out = self.l1(torch.cat([hidden,context], -1))
        out = self.l2(self.drop(F.leaky_relu(out)))
        return out

class Embedding(nn.Module):
    def __init__(self, vocab, d_model, drop=0.2):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        emb = self.lut(x) * math.sqrt(self.d_model)
        return self.drop(emb)
    

class LineDecoderCTC1(torch.nn.Module):
    def __init__(self, params):
        super(LineDecoderCTC1, self).__init__()

        self.params = params
        self.hidden_size = 256 #params["hidden_size"]
        self.batch_size =  params["training_params"]["batch_size"] # 2
        self.max_line_len = 10 # assuming there will be max 10 words in a line

        self.hidden = torch.zeros(self.batch_size, self.hidden_size).to("cuda")

        self.use_hidden = True #params["use_hidden"]
        self.input_size = 256 #params["features_size"]
        self.vocab_size = 79 #params["vocab_size"]
        self.attention = BahdanauAttention1(self.hidden_size).to("cuda:0")
        self.word_context_vectors = []
        
        self.context_vector2 = None
        self.context_vector = None 
        self.attention_weights = None
        
        self.wrdCon2Dcd = Linear(256,self.input_size) # this will take input word context vector and
                                                             # converts it to the shape that LSTM can process 

        self.decoder = RNNDecoder(self.hidden_size,1).to("cuda:0")
        self.output  = DeepOutputLayer(self.hidden_size, self.vocab_size).to("cuda:0")
        
        #############################################################
        # BAHANDNAU ATTENTION
        
        self.W1 = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2 = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.V =  Linear(self.hidden_size, 1, bias=False)

        #############################################################
        
        
        if self.use_hidden:
            
            self.lstm = LSTM(self.input_size, self.hidden_size, num_layers=1)
            self.end_conv = Conv2d(in_channels=self.hidden_size, out_channels=self.vocab_size + 1, kernel_size=1)
            
            """
            self.lstm1 = LSTM(self.input_size, self.hidden_size, num_layers=1)
            self.end_conv1 = Conv2d(in_channels=self.hidden_size, out_channels=self.vocab_size + 1, kernel_size=1)
            """
        else:
            self.end_conv = Conv2d(in_channels=self.input_size, out_channels=self.vocab_size + 1, kernel_size=1)
        

        self.linear1 = Linear(256, 256).to("cuda:0")
            

    def forward(self, x, h=None):
                        
                
        """
        x (B, C, W)
        """
        
        
        res,attns = [],[]
        
        x1 = x.clone() # torch.Size([1, 256, 116])
        x1 = x1.to("cuda:0")
        x1 = x1.permute(0,2,1)
        
        print("\n\t x1.shape:",x1.shape)
        print("\n\t 1:",x1.device," \t x.device:",x.device)
        
        hidden_rep = self.linear1(x1)

        #print("\n\t 2")
        
        print("\n\t 1.x.shape:",x1.shape,"\t x.shape:",x.shape,"\t hidden_rep.shape:",hidden_rep.shape)        
        #           1.x1.shape: torch.Size([2, 116, 256]) 	 x.shape: torch.Size([2, 256, 116]) 	 hidden_rep.shape: torch.Size([2, 116, 256])

        print("\n\t self.hidden:",self.hidden.shape) # torch.Size([2, 256])
        for i in range(self.max_line_len):
            print("\n\t i:",i)

            self.context_vector, self.context_vector2, self.attention_weights = self.attention(self.hidden, hidden_rep)
            
            encoder_outputs = hidden_rep.clone()
            
            print("\n\t ii:",self.context_vector.shape, self.context_vector2.shape, self.attention_weights.shape,encoder_outputs.shape)
            
            print("\n\t word context_vector:",self.context_vector.permute(2, 0, 1).shape," \t attention_weights.shape:",self.attention_weights.shape)
            print("\n\t word context_vector2:",self.context_vector2.shape) #  torch.Size([batch_size, 256])

            #print("\n\t self.hidden.shape before:",self.hidden)
            self.hidden = self.decoder(self.hidden,self.context_vector2) 
            #print("\n\t self.hidden.shape after:",self.hidden)
            
            charOut = self.output(self.hidden, self.context_vector2)
            print("\n\t charOut.shape =",charOut.shape)
            
            res.append(charOut)
            attns.append(self.attention_weights)
            dec_inp = charOut.data.max(1)[1]

            #############################################################################################################
        
            
            
            #############################################################################################################

            #########################################################################################################


            
            
            # 	 word context_vector: torch.Size([2, 256])  	 attention_weights.shape: torch.Size([2, 116, 1]) (old)
            #  word context_vector: torch.Size([2, 256, 116])  	 attention_weights.shape: torch.Size([2, 116])     (new)
            
                        
            xOut, hOut = self.lstm(self.context_vector.permute(2, 0, 1), h) 
            xOut = xOut.permute(1, 2, 0)

            print("\n\t xOut:",xOut.shape,"\t hOut.shape =",hOut[0].shape)
	        #  	 word context_vector: torch.Size([2, 256, 116])  	 attention_weights.shape: torch.Size([2, 116])    
            self.word_context_vectors.append(self.context_vector)

            temp3 = xOut
            out2 = self.end_conv(temp3.unsqueeze(3)).squeeze(3)
            print("\n\t out2.shape:",out2.shape)
            
            #########################################################################################################

            #########################################################################################################
            

            
            #########################################################################################################

        res = torch.stack(res)
        attns = torch.stack(attns)
        
        if 1:#self.use_hidden:
            
            print("\n\t 1.1.0.x.shape:",x.permute(2, 0, 1).shape) #  1.1.0.x.shape: torch.Size([116, 2, 256])         

            x, h = self.lstm(x.permute(2, 0, 1), h) 
            print("\n\t 1.1.1. x.shape:",x.shape," \t h.shape:",h[0].shape)    
            #  1.1.1. x.shape: torch.Size([116, 2, 256]), h.shape: torch.Size([1, 2, 256]) middle dim 2 is batch size in both vectors    

            x = x.permute(1, 2, 0)

        temp2 = x.unsqueeze(3)
        print("\n\t 2.x.shape:",x.shape," temp2.shape:",temp2.shape)        
        # 2.x.shape: torch.Size([2, 256, 116])  temp2.shape: torch.Size([2, 256, 116, 1])
        
        out = self.end_conv(x.unsqueeze(3)).squeeze(3)
        print("\n\t out1.shape:",out.shape,"\t out2.shape:",out2.shape)
        
        out = torch.squeeze(out, dim=2)
        out = log_softmax(out, dim=1)
        return out, h, res, attns


class BahdanauAttention1(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention1, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = Linear(hidden_size, hidden_size, bias=False)
        self.W2 = Linear(hidden_size, hidden_size, bias=False)
        self.V = Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, max_line_len, hidden_size)

        # Compute attention scores
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        W1_hidden = self.W1(hidden)  # (batch_size, 1, hidden_size)
        W2_encoder = self.W2(encoder_outputs)  # (batch_size, max_line_len, hidden_size)
        scores = self.V(torch.tanh(W1_hidden + W2_encoder))  # (batch_size, max_line_len, 1)

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, max_line_len, 1)
        print("\n\t attention_weights =",attention_weights.shape)
        
        #attention_weights = attention_weights.squeeze(2)
        
        # Compute context vector
        context_vector = (attention_weights * encoder_outputs).sum(dim=1)  # (batch_size, hidden_size)
        
        context_vector2 = context_vector.clone()
        
        context_vector = context_vector.unsqueeze(0).transpose(1, 2).repeat(1, encoder_outputs.shape[1], 1, 1)
        context_vector = context_vector.squeeze(0)
        context_vector = context_vector.permute(2,1,0) # torch.Size([3, 256, 116])
        
        
        # context_vector shape: (1, max_line_len, batch_size, hidden_size)

        return context_vector,context_vector2, attention_weights.squeeze(2)
