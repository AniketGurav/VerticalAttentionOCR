# -*- coding: utf-8 -*-
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


from basic.generic_training_manager import GenericTrainingManager
from torch.nn import CrossEntropyLoss, CTCLoss
import torch
from basic.utils import edit_wer_from_list, nb_chars_from_list, nb_words_from_list, LM_ind_to_str
import numpy as np
import editdistance
import re
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
import torch

# https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
import sys

#sys.path.insert(0, '/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/VerticalAttentionOCR/')
sys.path.append('/global/D1/projects/ZeroShot_Word_Recognition/E2E/VerticalAttentionOCR/')

from OCR.document_OCR.v_attention.models_pg_va import VerticalAttention, LineDecoderCTC , LineDecoderCTC1
#from OCR.document_OCR.v_attention.parameters import params as params1

#from parameters import params as params1

dataset_name = "IAM"  # ["RIMES", "IAM", "READ_2016"]


def seq2seq_loss(input, target):
    target = target.permute(1,0).contiguous()
    tsl = target.size(0)
    sl,bs,nc = input.size()
    
    #print("\n\t input.size():",input.size())
    ##print("\n\t target:",target.shape)
    #print("\n\t sl:",sl,"\t tsl:",tsl)
    
    if sl>tsl: target = F.pad(target, (0,0,0,sl-tsl))
    if tsl>sl: target = target[:sl]   # clip target to match input seq_len
        
    targ = target.view(-1)
    pred = input.view(-1, nc)

    # combination of LogSoftmax and NLLLoss
    return F.cross_entropy(pred, targ.long(), reduction='sum')/bs



class Manager(GenericTrainingManager):

    
    def __init__(self, params):
        
        #print("\n\t inside Manager !!!")

        super(Manager, self).__init__(params)

        """
            this is modified line decoding to handle the output at character level
        """
        #self.ldc1 = LineDecoderCTC1(params).to("cuda:0")


    def get_init_hidden(self, batch_size):
        num_layers = self.params["model_params"]["nb_layers_decoder"]
        hidden_size = self.params["model_params"]["hidden_size"]
        return torch.zeros((num_layers, batch_size, hidden_size), device=self.device), torch.zeros((num_layers, batch_size, hidden_size), device=self.device)

    
    import torch.autograd.profiler as profiler

    # Enable profiler
    #profiler.profile(enabled=True, use_cuda=True)

    # Run the forward and backward pass
    # Print the profiler output

    
    
    def train_batch(self, batch_data, metric_names):
        
        #print("\n\t yyyy:",self.dataset.tokens["blank"])
        
        loss_ctc_func = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        loss_ce_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])
        global_loss = 0
        total_loss_ctc = 0
        total_loss_ce = 0
        self.optimizer.zero_grad()

        x = batch_data["imgs"].to(self.device)
        y = [l.to(self.device) for l in batch_data["line_labels"]]
        y_len = batch_data["line_labels_len"]
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        batch_size = y[0].size()[0]

        mode = self.params["training_params"]["stop_mode"]
        max_nb_lines = self.params["training_params"]["max_pred_lines"] if mode == "fixed" else len(y)
        num_iter = max_nb_lines if mode == "fixed" else max_nb_lines+1
        for i in range(len(y), num_iter):
            y.append(torch.ones((batch_size, 1), dtype=torch.long , device=self.device)*self.dataset.tokens["pad"])
            y_len.append([0 for _ in range(batch_size)])

        status = "init"
        features = self.models["encoder"](x)
        batch_size, c, h, w = features.size()
        attention_weights = torch.zeros((batch_size, h), dtype=torch.float, device=self.device)
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None

        line_preds = [list() for _ in range(batch_size)]
        for i in range(num_iter):
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            status = "inprogress"
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            
            if mode in ["fixed", "early"] or i < max_nb_lines:
                

                #probs, hidden = self.models["decoder"](context_vector, hidden)

                
                #print("\n\t context_vector.shape:--->:",context_vector.shape,"\t hidden.shape:",hidden[0].shape)
                
                #res, attns, dec_inp = self.ldc1(context_vector, hidden)

                res, attns, dec_inp = self.models["decoder1"](context_vector, hidden)

                
                #randLen = probs.shape[2]
                
                #res = torch.rand([randLen,1,80]).to("cuda:0")
                #attns = torch.rand([randLen,1,randLen]).to("cuda:0")
                
                #res = res.permute(1,0,2)
                #print(" \t res.shape:",res.shape,"\t attns.shape:",attns.shape) #," \t dec_inp.shape:",dec_inp.shape)
                #print("\n\t ctc:",probs.shape)
                
                #print("\n\t y_len:",y_len)
                
                """
                ctc: torch.Size([122, 1, 80])
             	res.shape: torch.Size([122, 1, 80]) 	 attns.shape: torch.Size([122, 1, 122])
                """
            
                #loss_ctc1 = loss_ctc_func(probs.permute(2, 0, 1), y[i], x_reduced_len, y_len[i])
                
                crossLoss = seq2seq_loss(dec_inp, y[i]) 

                loss_ctc1 = crossLoss #loss_ctc_func(res, y[i], x_reduced_len, y_len[i])
                
                #print("\n\t x_reduced_len.shape:",x_reduced_len," \t y_len[i].shape:",y_len[i])
                
                print("\n\t loss_ctc =",loss_ctc1.item()," loss_ctc1 =",loss_ctc1.item())
                
                #target = torch.zeros_like(res)

                # Compute the mean squared error (MSE) loss
                #loss_ctc1 = F.mse_loss(res, target)
                
                #print("\n\t loss_ctc ==",loss_ctc1.item())
                
                total_loss_ctc += loss_ctc1.item()
                global_loss += loss_ctc1
                
                
            
            if mode == "learned":
                gt_decision = torch.ones((batch_size, ), device=self.device, dtype=torch.long)
                for j in range(batch_size):
                    if y_len[i][j] == 0:
                        if i > 0 and y_len[i-1][j] == 0:
                            gt_decision[j] = self.dataset.tokens["pad"]
                        else:
                            gt_decision[j] = 0
                loss_ce = loss_ce_func(decision, gt_decision)
                total_loss_ce += loss_ce.item()
                global_loss += loss_ce
                
            res1 = res.permute(1,2,0)
            #print("\n\t probs1:",probs.shape," \t res1.shape:",res.shape)
            #line_pred1 = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if y_len[i][j] > 0 else None for j, lp in enumerate(probs)]
            line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if y_len[i][j] > 0 else None for j, lp in enumerate(res1)]

            """
            line_pred = [] # res.shape: torch.Size([122, 1, 80]) 
            for j, lp in enumerate(res1):
                if y_len[i][j] > 0:
                    line_pred_j = torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]]
                    line_pred.append(line_pred_j)
                else:
                    line_pred.append(None)
            """
            for i, lp in enumerate(line_pred):
                if lp is not None:
                    line_preds[i].append(lp)

        self.backward_loss(global_loss)
        self.optimizer.step()

        metrics = self.compute_metrics(line_preds, batch_data["raw_labels"], metric_names, from_line=True)
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = total_loss_ctc / metrics["nb_chars"]
        if "loss_ce" in metric_names:
            metrics["loss_ce"] = total_loss_ce
        return metrics

    def evaluate_batch(self, batch_data, metric_names,imgName):
        
        print("\n\t inside evaluate_batch!!")
        
        try:
            print("\n\t batch_data.keys():",batch_data.keys())
        except Exception as e:
            print("\n\t exception in batch keys!!!")
            pass
        
        def append_preds(pg_preds, line_preds):
            for i, lp in enumerate(line_preds):
                if lp is not None:
                    pg_preds[i].append(lp)
            return pg_preds

        x = batch_data["imgs"].to(self.device) 
        
        orgH,orgW = x.shape[2],x.shape[3]
        
        #print("\n\t input image shape x =",x.shape,"\t orgH:",orgH,"\t orgW:",orgW) # x = torch.Size([8, 3, 786, 1100]) 
        
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]] 

        #print("\n\t x_reduced_len =",len(x_reduced_len))
        
        status = "init" 
        mode = self.params["training_params"]["stop_mode"] 
        max_nb_lines = self.params["training_params"]["max_pred_lines"] # 
        
        features = self.models["encoder"](x)
        
        #print("\n\t features.shape:",features.shape) # features.shape: torch.Size([8, 256, 25, 138])

        batch_size, c, h, w = features.size() # 
        

        #print("\n\t h=",h,"\t w:" ,w,"\t torch.__version__:",torch.__version__) # h= 25 	 w: 138
        
        attention_weights = torch.zeros((batch_size, h), device=self.device, dtype=torch.float)
        #print("\n\t attention_weights =",attention_weights.shape)  # attention_weights = torch.Size([8, 25])  
        
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None
        preds = [list() for _ in range(batch_size)]
        end_pred = [None for _ in range(batch_size)]

        #print("\n\t max_nb_lines:",max_nb_lines) # max_nb_lines: 30 
        #print("\n\t coverage =",coverage.shape)  # coverage = torch.Size([8, 25])
        #print("\n\t hidden =",hidden[0].shape,"\t len:",len(hidden)) #  hidden = torch.Size([1, 8, 256])
        try:
            print("\n\t end_pred =",end_pred[0].shape,"\t end_pred :",len(end_pred)) 
        except Exception as e:
            pass
        
        currDocLine = dict()
        #allLines = []
        
        currDocLine["currDocLine"] = []
        currDocLine["imgName"] = imgName
        #currDocLine = []
        
        import os

        imgName = imgName[0] # /test/
        
        #print("\n\t imgName =",imgName)
        
        if "train" in imgName:
            imgName = imgName.split("train/")[1]

        elif "test" in imgName:
            
            #print("test imgName:",imgName)
            imgName = imgName.split("test/")[1]
        
        elif "valid" in imgName: 
            imgName = imgName.split("valid/")[1]

        imgName = imgName.split(".png")[0]
        #print("\n\t imgName =>>>",imgName)

        # imgName: the name of the image file
        if not os.path.exists('./attentionWeights1/'+imgName):
            os.makedirs('./attentionWeights1/'+imgName)
            
        for i in range(max_nb_lines):
            
            #print("\n\t calling attention!!:",i)
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            
            #  1. context_vector.shape: torch.Size([8, 256, 138]) 	 attention_weights.shape: torch.Size([8, 25])  	 decision: torch.Size([8, 2])

            #print("\n\t 1. context_vector.shape:",context_vector.shape,"\t attention_weights.shape:",attention_weights.shape," \t decision:",decision.shape)
            
            
            #att_weights_np = att_weights.detach().cpu().numpy()
            
            #attention_weights1 = attention_weights.view(batch_size, orgH, orgW)
            #attention_weights1 = F.interpolate(attention_weights.unsqueeze(0), size=(orgH, orgW), mode='bicubic').squeeze(0)

            #bs = 8
            #temp = torch.zeros((batch_size, 25), device="cuda:0", dtype=torch.float)

            # Resize the tensor to size (bs, 50) using bicubic interpolation
            #attention_weights1 = F.interpolate(attention_weights.unsqueeze(0).unsqueeze(0), size=(batch_size, orgH), mode='bicubic').squeeze() *255
            
            attention_weights1 = F.interpolate(attention_weights.unsqueeze(0).unsqueeze(0), size=(batch_size, orgH), mode='bicubic').squeeze() *255

            #print("\n\t 0.attention_weights1 =",attention_weights1.shape,"\t orgW:",orgW,"\t orgH:",orgH)
            
            attention_weights1 = torch.cat([attention_weights1.unsqueeze(0).unsqueeze(2)]*orgW, dim=2)

            attention_weights1 = attention_weights1 #+ x

            #print("\n\t 00.attention_weights1 =",attention_weights1.shape,"\t orgW:",orgW,"\t orgH:",orgH)
            #attention_weights1 = attention_weights1.transpose(1, 2)  # transpose the tensor
            #print("\n\t 000.attention_weights1 =",attention_weights1.shape,"\t orgW:",orgW,"\t orgH:",orgH)
            
            #attention_weights1 = torch.cat([attention_weights1]*3, dim=1)
            attention_weights1 = attention_weights1.cpu().detach().numpy().transpose(0, 2, 1) 
            
            # Rescale attention weights to [0, 255]
            attention_weights1 = (attention_weights1 - attention_weights1.min()) / (
                attention_weights1.max() - attention_weights1.min()) * 255
            
            
            #attention_weights1 = cv2.applyColorMap(attention_weights1.astype(np.uint8), cv2.COLORMAP_JET).astype(np.uint8)


            #print("\n\t 1.attention_weights1_numpy.shape =",attention_weights1.shape)
            
            attention_weights2 = torch.from_numpy(attention_weights1)
            attention_weights2 = attention_weights2.unsqueeze(1)
            #print("\n\t 2.attention_weights1_numpy.shape =",attention_weights1.shape)
            attention_weights2 = attention_weights2.detach().cpu().numpy().transpose(0,3,2,1)
            attention_weights2 = np.concatenate((attention_weights2,attention_weights2,attention_weights2),axis=3)

            #print("\n\t 22. attention_weights1 =",attention_weights2.shape," \t image.shape:",x.shape)

            x2 = x.cpu().numpy().transpose(0,2,3,1) 
            x2 = x2 + attention_weights2
            x2 = x2.transpose(0,2,1,3)
            #print("\n\t 33. attention_weights1 =",attention_weights2.shape," \t image.shape:",x2.shape)

        
            
            """ 
            attention_weights1 = attention_weights1.reshape(batch_size,orgH,orgW)
            #attention_weights1 = torch.cat([attention_weights1]*3, dim=1)
            
            print("\n\t 2.attention_weights1 =",attention_weights1.shape,"\t orgW:",orgW," orgH:",orgH," \t x:",x.shape)
            
            #attention_weights1 = F.interpolate(attention_weights.unsqueeze(0).unsqueeze(-1), size=( orgH,orgW), mode='bicubic').squeeze() *255
            #attention_weights1 = torch.cat([attention_weights1]*orgW, dim=1)

            attention_weights1 =  x #attention_weights1 #+
                        
            
            #attention_weights1 = attention_weights1.cpu().detach().numpy() 
             
            #print("\n\t 3.attention_weights1 =",attention_weights1.shape,"\t orgW:",orgW," orgH:",orgH," \t x:",x.shape)

            #attention_weights1 = Image.fromarray(attention_weights1.astype('uint8'), mode='L')
            """
            
            # create the directory if it does not exist
            save_dir = './attentionWeights1/' + imgName+"//"
            
            # save the attention weights as an image
            #save_path = os.path.join(save_dir, f'{i}.png')
            save_path = os.path.join(save_dir, str(i)+".png")

            #attention_weights1.save(save_path)
            

            """
            for i in range(attention_weights1.shape[0]):
                # Extract the i-th image from the tensor
                image_i = attention_weights1[i]
                #print("\n\t image_i =",image_i.shape)    
                # Save the image as a PNG file
                save_image(image_i, save_dir + f'image_{i}.png')
            """



            #for ii in range(attention_weights1.shape[0]):
            for ii in range(x2.shape[0]):

                #aw = cv2.transpose(attention_weights1[ii]) 
                aw = cv2.transpose(x2[ii]) 

                cv2.imwrite(save_dir+f"attention_weights1_{i}_{ii}.png",aw )
                #cv2.imwrite(save_dir+attention_weights1+"_"+str(i)+"_"+str(ii)+".png",aw )


            #cv2.imwrite(save_path, attention_weights1)  
            
            #plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #plt.close()            
    
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            # 1. coverage = torch.Size([8, 25])

            #print("\n\t 1. coverage inside =",coverage.shape)
            
            #print("\n\t 2.context_vector.shape:",context_vector.shape,"\t hidden.shape:",hidden[0].shape,"\t len:",len(hidden),"\t input to decoder")
            #print("\n\t 2.context_vector.shape:",context_vector.shape,"\t hidden.shape:",hidden[1].shape,"\t len:",len(hidden),"\t input to decoder")

            
            probs, hidden = self.models["decoder"](context_vector, hidden)
            
            #  1.probs.shape: torch.Size([80, 138]) 	 hidden.shape: torch.Size([1, 8, 256]) 	 mode: learned

            #print("\n\t 1.probs.shape:",probs[0].shape,"\t hidden.shape:",hidden[0].shape,"\t mode:",mode)

            """

            import sys
            sys.path.insert(0, '/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/VerticalAttentionOCR/')


            import sys
            sys.path.insert(0, '/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/VerticalAttentionOCR/')
            from OCR.document_OCR.v_attention.models_pg_va import VerticalAttention, LineDecoderCTC, LineDecoderCTC1
            
            try:
                from OCR.document_OCR.v_attention.trainer_pg_va import LineDecoderCTC1
            except Exception as e:
                from trainer_pg_va import LineDecoderCTC1
                
                pass
            """
            
            #print("\n\t context_vector:",context_vector.device, "\t hidden.device:",hidden[0].device)
            
            res, attns, dec_inp = self.ldc1(context_vector, hidden)

            #  1.probs.shape: torch.Size([80, 138]) 	 hidden.shape: torch.Size([1, 8, 256]) 	 mode: learned

            #print("\n\t 2.probs.shape:",probs1[0].shape,"\t hidden.shape:",hidden1[0].shape,"\t mode:",mode,
            #print(" \t res.shape:",res.shape,"\t attns.shape:",attns.shape," \t dec_inp.shape:",dec_inp)
            
            
            status = "inprogress"

            line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(probs)]
            
            
            line_pred1 = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in line_pred]

            
            #print("\n\t line_pred =:",line_pred1)
            
            currDocLine["currDocLine"].append(line_pred1[0])
            
            # 0. line_pred = 8
            
            #print("\n\t 0. line_pred =",len(line_pred))
            if mode == "learned":
                decision = [torch.argmax(d, dim=0) for d in decision]
                for k, d in enumerate(decision):
                    if d == 0 and end_pred[k] is None:
                        end_pred[k] = i

            if mode in ["learned", "early"]:
                for k, p in enumerate(line_pred):
                    if end_pred[k] is None and np.all(p == self.dataset.tokens["blank"]):
                        end_pred[k] = i
            line_pred = [l if end_pred[j] is None else None for j, l in enumerate(line_pred)]
            preds = append_preds(preds, line_pred)
            
            # 1. line_pred = 8
            #print("\n\t 1. line_pred =",len(line_pred))

            if np.all([end_pred[k] is not None for k in range(batch_size)]):
                break

        
        allText = currDocLine["currDocLine"]
        #print("\n\t allText1 =:",allText)        
        
        allText1 = "\n".join(allText)
        #print("\n\t name of the document:",currDocLine["imgName"])
        #print("\n\t allText =:",allText1)        
        
        metrics = self.compute_metrics(preds, batch_data["raw_labels"], metric_names, from_line=True)

        if "diff_len" in metric_names:
            end_pred = [end_pred[k] if end_pred[k] is not None else i for k in range(len(end_pred))]
            diff_len = np.array(end_pred)-np.array(batch_data["nb_lines"])
            metrics["diff_len"] = diff_len

        return metrics

    def evaluate_batch2(self, batch_data, metric_names,imgName):
        
        #print("\n\t inside evaluate_batch2!!")
        
        try:
            print("\n\t batch_data.keys():",batch_data.keys())
        except Exception as e:
            print("\n\t exception in batch keys!!!")
            pass
        
        def append_preds(pg_preds, line_preds):
            for i, lp in enumerate(line_preds):
                if lp is not None:
                    pg_preds[i].append(lp)
            return pg_preds

        x = batch_data["imgs"].to(self.device) 
        
        orgH,orgW = x.shape[2],x.shape[3]
        
        #print("\n\t input image shape x =",x.shape,"\t orgH:",orgH,"\t orgW:",orgW) # x = torch.Size([8, 3, 786, 1100]) 
        
        y_len = batch_data["line_labels_len"]
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]] 

        #print("\n\t x_reduced_len =",len(x_reduced_len))
        
        status = "init" 
        mode = self.params["training_params"]["stop_mode"] 
        max_nb_lines = self.params["training_params"]["max_pred_lines"] # 
        
        features = self.models["encoder"](x)
        
        #print("\n\t features.shape:",features.shape) # features.shape: torch.Size([8, 256, 25, 138])

        batch_size, c, h, w = features.size() # 
        

        #print("\n\t h=",h,"\t w:" ,w,"\t torch.__version__:",torch.__version__) # h= 25 	 w: 138
        
        attention_weights = torch.zeros((batch_size, h), device=self.device, dtype=torch.float)
        #print("\n\t attention_weights =",attention_weights.shape)  # attention_weights = torch.Size([8, 25])  
        
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None
        preds = [list() for _ in range(batch_size)]
        end_pred = [None for _ in range(batch_size)]

        #print("\n\t max_nb_lines:",max_nb_lines) # max_nb_lines: 30 
        #print("\n\t coverage =",coverage.shape)  # coverage = torch.Size([8, 25])
        #print("\n\t hidden =",hidden[0].shape,"\t len:",len(hidden)) #  hidden = torch.Size([1, 8, 256])
        try:
            print("\n\t end_pred =",end_pred[0].shape,"\t end_pred :",len(end_pred)) 
        except Exception as e:
            pass
        
        currDocLine = dict()
        #allLines = []
        
        currDocLine["currDocLine"] = []
        currDocLine["imgName"] = imgName
        #currDocLine = []
        
        import os

        imgName = imgName[0] # /test/
        
        #print("\n\t imgName =",imgName)
        
        if "train" in imgName:
            imgName = imgName.split("train/")[1]

        elif "test" in imgName:
            
            #print("test imgName:",imgName)
            imgName = imgName.split("test/")[1]
        
        elif "valid" in imgName: 
            imgName = imgName.split("valid/")[1]

        imgName = imgName.split(".png")[0]
        #print("\n\t imgName =>>>",imgName)

        # imgName: the name of the image file
        if not os.path.exists('./attentionWeights1/'+imgName):
            os.makedirs('./attentionWeights1/'+imgName)
            
        for i in range(max_nb_lines):
            
            #print("\n\t calling attention!!:",i)
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            

            
            res, attns, dec_inp  = self.models["decoder1"](context_vector, hidden)
                        
            
            status = "inprogress"

            res1 = res.permute(1,2,0)
            #line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if y_len[i][j] > 0 else None for j, lp in enumerate(res1)]

            
            #print("\n\t y_len:",y_len)
            
            line_pred = []
            for j, lp in enumerate(res1):
                if 1:
                    pred = torch.argmax(lp, dim=0).detach().cpu().numpy() #[:x_reduced_len[j]]
                    line_pred.append(pred)
                else:
                    line_pred.append(None)



            #line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(probs)]
                        
            line_pred1 = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in line_pred]

            
            print("\n\t line_pred =:",line_pred1)
            
            currDocLine["currDocLine"].append(line_pred1[0])
            
            # 0. line_pred = 8
            
            #print("\n\t 0. line_pred =",len(line_pred))
            if mode == "learned":
                decision = [torch.argmax(d, dim=0) for d in decision]
                for k, d in enumerate(decision):
                    if d == 0 and end_pred[k] is None:
                        end_pred[k] = i

            if mode in ["learned", "early"]:
                for k, p in enumerate(line_pred):
                    if end_pred[k] is None and np.all(p == self.dataset.tokens["blank"]):
                        end_pred[k] = i
            line_pred = [l if end_pred[j] is None else None for j, l in enumerate(line_pred)]
            preds = append_preds(preds, line_pred)
            
            # 1. line_pred = 8
            #print("\n\t 1. line_pred =",len(line_pred))

            if np.all([end_pred[k] is not None for k in range(batch_size)]):
                break

        
        allText = currDocLine["currDocLine"]
        #print("\n\t allText1 =:",allText)        
        
        allText1 = "\n".join(allText)
        #print("\n\t name of the document:",currDocLine["imgName"])
        #print("\n\t allText =:",allText1)        
        
        metrics = self.compute_metrics(preds, batch_data["raw_labels"], metric_names, from_line=True)

        if "diff_len" in metric_names:
            end_pred = [end_pred[k] if end_pred[k] is not None else i for k in range(len(end_pred))]
            diff_len = np.array(end_pred)-np.array(batch_data["nb_lines"])
            metrics["diff_len"] = diff_len

        return metrics



    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def compute_metrics(self, ind_x, str_y,  metric_names=list(), from_line=False):
        if from_line:
            str_x = list()
            for lines_token in ind_x:
                list_str = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in lines_token]
                str_x.append(re.sub("( )+", ' ', " ".join(list_str).strip(" ")))
        else:
            str_x = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in ind_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u, v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
        metrics["nb_samples"] = len(str_x)
        return metrics
