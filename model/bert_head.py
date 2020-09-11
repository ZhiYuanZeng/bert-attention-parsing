import torch.nn as nn
import torch
from typing import List
import numpy as np
import random

# from sklearn.metrics import f1_score

class BertHead(nn.Module):
    def __init__(self, bert_model, hidden_size, head_num, layer_nb, dropout_rate=0.1, loss_function='mle',
            rm_bpe_from_a=False, pred_label=0):
        assert hidden_size%head_num==0
        proj_hidden_size=hidden_size//head_num
        super(BertHead,self).__init__()
        self.bert=bert_model

        if isinstance(layer_nb,int):
            layer_nb=[layer_nb,]
        self.query_proj=nn.ModuleList([
            nn.Linear(hidden_size, proj_hidden_size) for i in range(len(layer_nb))
        ])
        self.key_proj=nn.ModuleList([
            nn.Linear(hidden_size, proj_hidden_size) for i in range(len(layer_nb))
        ])
        self.layer_nb=[int(layer) for layer in layer_nb]
        self.init_head(head_num)
        self.dropout=nn.Dropout(dropout_rate)
        self.rm_bpe_from_a=rm_bpe_from_a
        if pred_label!=0:
            self.label_predictor=nn.Sequential(
                nn.Linear(4*hidden_size,hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                # nn.Dropout(dropout_rate),
                nn.Linear(hidden_size,int(pred_label)+1),
                nn.Softmax()
            )
        self.pred_label=pred_label
        self.loss_function=loss_function
        
    def _span_embbedding(self,hiddens,i,j):
        hidden_size=hiddens.shape[-1]
        assert hidden_size%2==0
        # forward_hiddens=hiddens[:,:hidden_size//2]
        # backward_hiddens=hiddens[:,hidden_size//2:]
        return torch.cat((hiddens[i],hiddens[j],hiddens[i]*hiddens[j],hiddens[i]-hiddens[j]),dim=-1).unsqueeze(0)

    def _layer_forward_(self, hiddens , query_proj, key_proj, attention_mask, bpe_ids, labels, inner_only):
        """ 
        labels: List[List[Tuple[int,int,int]]]
        """
        hiddens=self.dropout(hiddens)
        query=query_proj(hiddens) # (bsz,seq,h//head)
        key=key_proj(hiddens) # (bsz,seq,h//head)
        hidden_size=query.shape[-1]
        if self.rm_bpe_from_a:
            attention=torch.matmul(query,key.transpose(-1,-2)).masked_fill_(~attention_mask.bool().unsqueeze(1), float('-inf'))
            attention=attention/(hidden_size**0.5)
            attention=torch.softmax(attention,dim=-1)
            attention=[self.remove_bpe_from_attention(b, a) for a,b in zip(attention,bpe_ids)]
        
        loss=torch.tensor(0.).to(hiddens.device)
        label_loss_scale=2
        preds=[]
        label_preds, label_tgts=[],[]
        for bidx in range(len(hiddens)):
            seq_len=len(attention_mask[bidx].nonzero())-2-len(bpe_ids[bidx])
            if seq_len!=len(labels[bidx])+1:
                print('error at bert_head.py: length of words is not equal to labels')
                continue
            if self.rm_bpe_from_a:
                a=attention[bidx]
                assert len(a[0].nonzero())-2==seq_len
                # assert a.sum()==len(a)
                a=a[1:1+seq_len,1:1+seq_len] # remove cls token
            else:
                k,q=key[bidx],query[bidx]
                k=self.remove_bpe_from_hiddens(bpe_ids[bidx],k)
                q=self.remove_bpe_from_hiddens(bpe_ids[bidx],q)
                k,q=k[1:1+seq_len],q[1:1+seq_len] # remove cls, sep, pad
                a=torch.matmul(q,k.transpose(-1,-2))/(hidden_size**0.5)
                a=torch.softmax(a,dim=-1)
                # assert a.sum().item()==len(a)
            # assert (a>0).all()
            a=torch.log(a+1e-6)
            span_list=labels[bidx]
            span_len=0
            # # TODO: sample split point from all split points
            # category={}
            # tau=8
            # for span in span_list:
            #     start_idx, end_idx,split_idx=span[1]
            #     if end_idx-start_idx<2: continue
            #     if category.get(split_idx-start_idx) is None:
            #         category[split_idx-start_idx]=[]
            #     category[split_idx-start_idx].append(span)
            # sample_prob=np.array([len(category[k]) for k in sorted(category.keys())])
            # sample_prob=np.exp(sample_prob/tau)
            # sample_prob=sample_prob/sample_prob.sum()
            # for i in range(len(span_list)):
            #     c=np.random.choice(list(category.keys()),1,p=sample_prob)
            #     l, r,split_pos=random.choice(category[c.item()])[1]

            for span in span_list:
                label,span=span
                l,r,split_pos=span
                if self.pred_label>0:
                    span_embed=self._span_embbedding(hiddens[bidx], l, r) # hi;hj
                    probs=self.label_predictor(span_embed).squeeze()
                    likelihood=probs[label] # null: 1
                    loss+=-torch.log(likelihood)
                    label_preds.append(probs.argmax(dim=-1).item())
                    label_tgts.append(int(label))
                if (r-l)==1: continue
                span_len+=1
                scores=self.get_predictions(l,r,a,inner_only)
                assert len(scores)==scores.shape[-1]
                if self.loss_function=='mle':
                    logp=torch.log_softmax(scores, dim=-1)
                    assert not torch.isnan(logp).all()
                    preds.append((logp.argmax(dim=-1)==(split_pos-l)).item())
                    loss+=(-logp[split_pos-l])
                else: # hinge loss
                    indices=[i for i in range(len(scores)) if i!=(split_pos-l)]
                    s_hat=scores[indices].max()
                    loss+=max(0,1+s_hat-scores[split_pos-l])
                    preds.append((scores.argmax(dim=-1)==(split_pos-l)).item())
        return loss/len(hiddens),sum(preds)/max(len(preds),1),0.
    
    def forward(self, inputs , attention_mask, bpe_ids, labels, inner_only):
        """ return loss, average acc,f1 of multiple layers """
        _,_,all_hiddens,_=self.bert(inputs, attention_mask)
        loss, acc, f1=0.,0.,0.
        for i,layer in enumerate(self.layer_nb):
            l,a,f=self._layer_forward_(all_hiddens[layer] , self.query_proj[i], self.key_proj[i], attention_mask,
                                    bpe_ids, labels, inner_only)
            loss+=l
            acc+=a
            f1+=f
        return loss, acc/len(self.layer_nb), f1/len(self.layer_nb)

    def parse(self, inputs , attention_mask, bpe_ids, sents, 
            rm_bpe_from_a=True, decoding='cky',inner_only=False):
        _,_,all_hiddens,all_attentions=self.bert(inputs, attention_mask)
        all_attens,all_keys,all_querys=[],[],[]
        trees=[]
        for bidx in range(len(inputs)):
            attens_of_layers=[]
            for i,layer in enumerate(self.layer_nb):
                hiddens=all_hiddens[layer] # (bsz,seq,h)
                query=self.query_proj[i](hiddens) # (bsz,seq,h//head)
                key=self.key_proj[i](hiddens) # (bsz,seq,h//head)
                hidden_size=query.shape[-1]

                seq_len=len(attention_mask[bidx].nonzero())-2-len(bpe_ids[bidx])
                assert seq_len==len(sents[bidx])
                k,q=key[bidx],query[bidx]
                if self.rm_bpe_from_a:
                    a=torch.matmul(q,k.transpose(-1,-2)).masked_fill_(~attention_mask[bidx].bool().unsqueeze(0), float('-inf'))
                    a=a/(hidden_size**0.5)
                    a=torch.softmax(a,dim=-1)
                    a=self.remove_bpe_from_attention(bpe_ids[bidx],a)
                    a=a[1:1+seq_len,1:1+seq_len] # remove cls token
                else:
                    k=self.remove_bpe_from_hiddens(bpe_ids[bidx],k)
                    q=self.remove_bpe_from_hiddens(bpe_ids[bidx],q)
                    k,q=k[1:1+seq_len],q[1:1+seq_len] # remove cls, sep, pad
                    a=torch.matmul(q,k.transpose(-1,-2))/(hidden_size**0.5)
                    a=torch.softmax(a,dim=-1)
                    all_keys.append(k.cpu().numpy())
                    all_querys.append(q.cpu().numpy())
                assert (a>0).all()
                a=torch.log(a)
                attens_of_layers.append(a.cpu().numpy())

            if decoding=='cky': 
                tree=self.cky_pase(attens_of_layers, all_hiddens[self.layer_nb[0]][bidx] ,
                                sents[bidx], inner_only=inner_only,norm=(self.loss_function=='mle'))
            else: tree=self.greedy_parse(a, list(range(seq_len)), inner_only=inner_only)
            if self.pred_label>0: tree=tree[1]
            all_attens.extend(attens_of_layers)
            attens_of_layers.clear()
            trees.append(tree)
        return trees, all_attens, all_keys, all_querys
    
    def greedy_parse(self,attention, sent, inner_only=False):
        def parse(l,r):
            nonlocal attention, sent
            if (l+1)==r:
                return [sent[l],sent[r]]
            if l==r:
                return sent[l]
            logp=self.get_predictions(l,r,attention, inner_only)
            split_idx=torch.argmax(logp, dim=-1)
            ltree=parse(l,split_idx+l)
            rtree=parse(split_idx+1+l, r)
            return [ltree, rtree]
        assert len(attention)==len(sent)
        tree=parse(0, len(attention)-1)
        return tree

    def cky_pase(self,attens_of_layers, hiddens, sent, inner_only=False, norm=True):
        """ 
        emsemble tree distribution of different layer
        param:
            attens_of_layers: List, attention of one sample of different layers
        """
        def parse_structure(split_score: torch.tensor,sent: List[str],i,j):
            if i==j:
                return sent[i]
            if (i+1)==j:
                return [sent[i],sent[j]]
            tree=[]
            k=int(split_score[i,j].item())
            tree.append(parse_structure(split_score,sent,i,k))
            tree.append(parse_structure(split_score,sent,k+1,j))
            return tree
        
        def parse_label(split_score: torch.tensor,sent: List[str],i,j):
            if i==j:
                return '',sent[i]
            span_embed=self._span_embbedding(hiddens, i, j)
            label=torch.argmax(self.label_predictor(span_embed).squeeze(),dim=-1)
            label=en_idx2label[label.item()]

            if (i+1)==j:
                return label, [sent[i], sent[j]]
            k=int(split_score[i,j].item())
            llabel, ltree=parse_label(split_score,sent,i,k)
            rlabel, rtree=parse_label(split_score,sent,k+1,j)
            subtrees=[]
            if llabel=='empty': 
                subtrees+=ltree
            else: subtrees.append(ltree)
            if rlabel=='empty':
                subtrees+=rtree
            else: subtrees.append(rtree)
            return label, [*subtrees]
        
        m=len(sent)                                                                                                                                                                                                                                                                                                                                     
        split_score = torch.zeros(m, m, m)
        for a in attens_of_layers:
            assert len(a)==len(sent)
            m = a.shape[-1]
            a_T = a.transpose(-1, -2)
            sum_of_attention = torch.zeros(m, m)
            for i in range(m):
                for j in range(i,m):                                                                                                                                                                                
                    if i <= j:
                        if i>0:
                            left = a[i:(j+1), 0:i].sum()+a_T[i:(j+1), 0:i].sum()
                        else:
                            left = 0

                        if j+1 < m:
                            right = (a[i:(j+1), (j+1):m].sum()+a_T[i:(j+1), (j+1):m].sum())
                        else:
                            right = 0
                        
                        scale1 = (j-i+1)**2
                        scale2 = 2*(m-(j-i+1))*(j-i+1) if m != (j-i+1) else 1
                        sum_of_attention[i, j] = a[i:(j+1), i:(j+1)].sum()/scale1
                        if not inner_only:
                            sum_of_attention[i,j]=sum_of_attention[i,j]-(left+right)/scale2
            for i in range(m):
                for j in range(i+1, m):
                    split_score_of_span=torch.zeros(j-i)
                    for k in range(i, j):
                        split_score_of_span[k-i]=sum_of_attention[i,k]+sum_of_attention[k+1,j]
                    if norm:
                        split_score_of_span=torch.softmax(split_score_of_span,dim=-1)
                    split_score[i, j, i:j]+=split_score_of_span
                    del split_score_of_span
        split_score/=len(attens_of_layers)
        if norm: split_score=torch.log(split_score+1e-6)
        L=len(sent)
        splits=torch.zeros([L,L])
        tree_score=torch.zeros([L,L])

        for w in range(2,L+1): # [3,L]
            for i in range(L):
                j=i+w-1
                if j>=L: continue
                maxv=float('-inf')
                for k in range(i,j): # [i,j)
                    s=split_score[i,j,k].item()
                    # assert s!=0
                    s=s+tree_score[i,k]+tree_score[k+1,j]
                    if maxv<s:
                        splits[i,j]=k
                        maxv=s
                tree_score[i,j]=maxv
        if self.pred_label:
            tree=parse_label(splits,sent,0,L-1) 
        else:
            tree=parse_structure(splits,sent,0,L-1)

        return tree

    def remove_bpe_from_hiddens(self, bpe_ids, hiddens):
        """
        remove bpe id of one sentence
        args
            bpe_ids: List
            hiddens/attention: Tensor (seq,h)
        """
        seq_len=hiddens.shape[0]
        device=hiddens.device
        non_bpe_ids=[]
        no_bpe_hiddens=hiddens.clone()
        if isinstance(bpe_ids,tuple): bpe_ids=bpe_ids[0]
        cnt=1
        for i in reversed(range(seq_len)):
            if i not in bpe_ids:
                non_bpe_ids.append(i)
                cnt=1
            else:
                cnt+=1
                no_bpe_hiddens[i-1]=no_bpe_hiddens[i-1]/cnt+no_bpe_hiddens[i]*(cnt-1)/cnt
        non_bpe_ids.reverse()
        no_bpe_hiddens=torch.index_select(no_bpe_hiddens,dim=0,index=torch.tensor(non_bpe_ids).to(device))
        return no_bpe_hiddens
    
    def remove_bpe_from_attention(self, bpe_ids, attention):
        """
        remove bpe id of one sentence
        args
            bpe_ids: List
            attention: Tensor (seq,seq)
        """
        seq_len=attention.shape[0]
        device=attention.device
        no_bpe_attention=attention.clone()
        non_bpe_ids=[]
        cnt=1
        if isinstance(bpe_ids,tuple):  bpe_ids=bpe_ids[0]
        for i in reversed(range(seq_len)):
            if i not in bpe_ids:
                cnt=1
                non_bpe_ids.append(i)
            else:
                cnt+=1
                no_bpe_attention[i-1]=no_bpe_attention[i-1]/cnt+no_bpe_attention[i]*(cnt-1)/cnt
                no_bpe_attention[:,i-1]=(no_bpe_attention[:,i]+no_bpe_attention[:,i-1])
        non_bpe_ids.reverse()
        assert len(non_bpe_ids)==(seq_len-len(bpe_ids))
        no_bpe_attention=torch.index_select(no_bpe_attention,dim=0,index=torch.tensor(non_bpe_ids).to(device))
        no_bpe_attention=torch.index_select(no_bpe_attention,dim=1,index=torch.tensor(non_bpe_ids).to(device))
        return no_bpe_attention

    def get_predictions(self, l, r, attention, inner_only=False):
        def get_span_scores(i,j,a,seq_len):
            a_T=a.t()
            m=seq_len
            if i>0:
                left = a[i:(j+1), 0:i].sum(dim=(-1, -2))+a_T[i:(j+1), 0:i].sum(dim=(-1, -2))
            else:
                left = 0

            if j+1 < m:
                right = (a[i:(j+1), (j+1):m].sum(dim=(-1, -2))+a_T[i:(j+1), (j+1):m].sum(dim=(-1, -2)))
            else:
                right = 0
            
            scale1 = (j-i+1)**2
            scale2 = 2*(m-(j-i+1))*(j-i+1) if m != (j-i+1) else 1
            res=a[i:(j+1), i:(j+1)].sum(dim=(-1, -2))/scale1
            if not inner_only:
                res=res-(left+right)/scale2
            assert not (res==float('inf')).any()
            return res
        
        seq_len=len(attention[0])
        prediction=torch.empty(r-l).to(attention.device)
        for i in range(l,r):
            left_score=get_span_scores(l,i,attention,seq_len)
            right_score=get_span_scores(i+1,r,attention,seq_len)
            prediction[i-l]=left_score+right_score
        return prediction

    def init_head(self, head_num):
        def _avg_on_head(weight):
            hidden_size=weight.shape[-1]
            if len(weight.shape)==2: # weight
                return weight.view(head_num, hidden_size//head_num, hidden_size).mean(dim=0)
            else:
                return weight.view(head_num, hidden_size//head_num).mean(dim=0)
        for i,layer in enumerate(self.layer_nb):
            if layer>=11: layer=10 # layer11 is the last layer
            key_w=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{layer+1}.attention.self.key.weight'])
            key_b=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{layer+1}.attention.self.key.bias'])
            query_w=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{layer+1}.attention.self.query.weight'])
            query_b=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{layer+1}.attention.self.query.bias'])
            self.key_proj[i].weight.data.copy_(key_w.data)
            self.key_proj[i].bias.data.copy_(key_b.data)
            self.query_proj[i].weight.data.copy_(query_w.data)
            self.query_proj[i].bias.data.copy_(query_b.data)