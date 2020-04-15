import torch.nn as nn
import torch
from typing import List

class BertHead(nn.Module):
    def __init__(self, bert_model, hidden_size, head_num, layer_nb, dropout_rate=0.1, rm_bpe_from_a=False):
        assert hidden_size%head_num==0
        proj_hidden_size=hidden_size//head_num
        super(BertHead,self).__init__()
        self.bert=bert_model
        self.query_proj=nn.Linear(hidden_size, proj_hidden_size)
        self.key_proj=nn.Linear(hidden_size, proj_hidden_size)
        self.layer_nb=layer_nb
        self.init_head(head_num)
        self.dropout=nn.Dropout(dropout_rate)
        self.rm_bpe_from_a=rm_bpe_from_a
        # self.max_seq_len=max_seq_len
    
    def forward(self, inputs , attention_mask, bpe_ids, labels, method):
        """ 
        labels: List[List[Tuple[int,int,int]]]
        """
        _,_,all_hiddens,_=self.bert(inputs, attention_mask)
        hiddens=all_hiddens[self.layer_nb] # (bsz,seq,h)
        hiddens=self.dropout(hiddens)
        query=self.query_proj(hiddens) # (bsz,seq,h//head)
        key=self.key_proj(hiddens) # (bsz,seq,h//head)
        hidden_size=query.shape[-1]
        if self.rm_bpe_from_a:
            attention=torch.matmul(query,key.transpose(-1,-2)).masked_fill_(~attention_mask.bool().unsqueeze(1), float('-inf'))
            attention=attention/(hidden_size**0.5)
            attention=torch.softmax(attention,dim=-1)
            attention=[self.remove_bpe_from_attention(b, a) for a,b in zip(attention,bpe_ids)]
        
        loss=torch.tensor(0.).to(inputs.device)
        preds=[]
        for bidx in range(len(hiddens)):
            seq_len=len(attention_mask[bidx].nonzero())-2-len(bpe_ids[bidx])
            if seq_len!=len(labels[bidx])+1:
                print('error example')
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
            for span in span_list:
                l,r,split_pos=span
                if (r-l)==1:
                    continue
                span_len+=1
                logp=self.get_predictions(l,r,a,method)
                assert len(logp)==logp.shape[-1]
                assert not torch.isnan(logp).all()
                preds.append((logp.argmax(dim=-1)==(split_pos-l)).item())
                loss+=(-logp[split_pos-l])/span_len
        
        return loss/len(hiddens),sum(preds)/max(len(preds),1)
    
    def parse(self, inputs , attention_mask, bpe_ids, sents, 
            rm_bpe_from_a=True, decoding='cky',method='inner',use_bert_head=False):
        _,_,all_hiddens,all_attentions=self.bert(inputs, attention_mask)
        hiddens=all_hiddens[self.layer_nb] # (bsz,seq,h)
        query=self.query_proj(hiddens) # (bsz,seq,h//head)
        key=self.key_proj(hiddens) # (bsz,seq,h//head)
        all_attens,all_keys,all_querys=[],[],[]
        hidden_size=query.shape[-1]
        if use_bert_head>0:
            attention=all_attentions[self.layer_nb][:,use_bert_head]
        elif self.rm_bpe_from_a:
            attention=torch.matmul(query,key.transpose(-1,-2)).masked_fill_(~attention_mask.bool().unsqueeze(1), float('-inf'))
            attention=attention/(hidden_size**0.5)
            attention=torch.softmax(attention,dim=-1)
        trees=[]
        for bidx in range(len(hiddens)):
            seq_len=len(attention_mask[bidx].nonzero())-2-len(bpe_ids[bidx])
            k,q=key[bidx],query[bidx]
            k=self.remove_bpe_from_hiddens(bpe_ids[bidx],k)
            q=self.remove_bpe_from_hiddens(bpe_ids[bidx],q)
            k,q=k[1:1+seq_len],q[1:1+seq_len] # remove cls, sep, pad
            if self.rm_bpe_from_a or use_bert_head>0:
                a=attention[bidx]
                a=self.remove_bpe_from_attention(bpe_ids[bidx],a)
                a=a[1:1+seq_len,1:1+seq_len] # remove cls token
            else:
                a=torch.matmul(q,k.transpose(-1,-2))/(hidden_size**0.5)
                a=torch.softmax(a,dim=-1)
            assert (a>0).all()
            all_attens.append(a.cpu().numpy())
            all_keys.append(k.cpu().numpy())
            all_querys.append(q.cpu().numpy())
            a=torch.log(a)
            # print(a)
            if decoding=='cky': tree=self.cyk_parse(a, sents[bidx], method,noram=(use_bert_head<0))
            else: tree=self.greedy_parse(a, sents[bidx], method)
            trees.append(tree)
        return trees, all_attens, all_keys, all_querys

    def greedy_parse(self,attention, sent, method):
        def parse(l,r):
            nonlocal attention, sent
            if (l+1)==r:
                return [sent[l],sent[r]]
            if l==r:
                return sent[l]
            logp=self.get_predictions(l,r,attention, method)
            split_idx=torch.argmax(logp, dim=-1)
            ltree=parse(l,split_idx+l)
            rtree=parse(split_idx+1+l, r)
            return [ltree, rtree]
        assert len(attention)==len(sent)
        tree=parse(0, len(attention)-1)
        return tree

    def cyk_parse(self,a,sent, method='inner', noram=True):
        def parse(split_score: torch.tensor,sent: List[str],i,j):
            if i==j:
                return sent[i]
            if (i+1)==j:
                return [sent[i],sent[j]]
            tree=[]
            k=int(split_score[i,j].item())
            tree.append(parse(split_score,sent,i,k))
            tree.append(parse(split_score,sent,k+1,j))
            return tree

        assert len(a)==len(sent)
        m = a.shape[-1]
        a_T = a.transpose(-1, -2)
        sum_of_attention = torch.zeros(m, m)
        split_score = torch.zeros(m, m, m)
        if method=='inner':
            for i in range(m):
                for j in range(i,m):
                    if i <= j:
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
                        sum_of_attention[i, j] = a[i:(j+1), i:(j+1)].sum(dim=(-1, -2))/scale1-(left+right)/scale2
        else:
            for i in range(m):
                for j in range(i, m):
                    sum_of_attention[i,j] = a[i:(j+1), i:(j+1)].sum()
        # print(sum_of_attention)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(i, j):
                    if method=='inner':
                        split_score[i, j, k] = sum_of_attention[i,k]+sum_of_attention[k+1,j]
                    else:
                        split_score[i, j, k]=(sum_of_attention[i, j]-sum_of_attention[i, k]-sum_of_attention[k+1, j]) / (2*(j-k)*(k-i+1))
                if noram:
                    split_score[i, j, i:j]=torch.log_softmax(split_score[i, j, i:j],dim=-1)
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
        tree=parse(splits,sent,0,L-1) 

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
        for i in reversed(range(seq_len)):
            if i not in bpe_ids:
                non_bpe_ids.append(i)
            else:
                no_bpe_hiddens[i-1]=(no_bpe_hiddens[i]+no_bpe_hiddens[i-1])/2
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
        if isinstance(bpe_ids,tuple):  bpe_ids=bpe_ids[0]
        for i in reversed(range(seq_len)):
            if i not in bpe_ids:
                non_bpe_ids.append(i)
            else:
                no_bpe_attention[i-1]=(no_bpe_attention[i]+no_bpe_attention[i-1])/2
                no_bpe_attention[:,i-1]=(no_bpe_attention[:,i]+no_bpe_attention[:,i-1])
        non_bpe_ids.reverse()
        assert len(non_bpe_ids)==(seq_len-len(bpe_ids))
        no_bpe_attention=torch.index_select(no_bpe_attention,dim=0,index=torch.tensor(non_bpe_ids).to(device))
        no_bpe_attention=torch.index_select(no_bpe_attention,dim=1,index=torch.tensor(non_bpe_ids).to(device))
        return no_bpe_attention

    def get_predictions(self, l, r, attention, method='inner'):
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
            res=a[i:(j+1), i:(j+1)].sum(dim=(-1, -2))/scale1-(left+right)/scale2
            assert not (res==float('inf')).any()
            return res
        
        seq_len=len(attention[0])
        prediction=torch.empty(r-l).to(attention.device)
        for i in range(l,r):
            if method=='inner':
                left_score=get_span_scores(l,i,attention,seq_len)
                right_score=get_span_scores(i+1,r,attention,seq_len)
                prediction[i-l]=left_score+right_score
            else:
                prediction[i-l]=(attention[l:(i+1), (i+1):(r+1)].sum()+
                                attention[(i+1):(r+1), l:(i+1)].sum())/(2*(r-i)*(i-l+1))
        prediction=torch.log_softmax(prediction, dim=-1)
        return prediction

    def init_head(self, head_num):
        def _avg_on_head(weight):
            hidden_size=weight.shape[-1]
            if len(weight.shape)==2: # weight
                return weight.view(head_num, hidden_size//head_num, hidden_size).mean(dim=0)
            else:
                return weight.view(head_num, hidden_size//head_num).mean(dim=0)
        key_w=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{self.layer_nb+1}.attention.self.key.weight'])
        key_b=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{self.layer_nb+1}.attention.self.key.bias'])
        query_w=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{self.layer_nb+1}.attention.self.query.weight'])
        query_b=_avg_on_head(self.state_dict()[f'bert.encoder.layer.{self.layer_nb+1}.attention.self.query.bias'])
        self.key_proj.weight.data.copy_(key_w.data)
        self.key_proj.bias.data.copy_(key_b.data)
        self.query_proj.weight.data.copy_(query_w.data)
        self.query_proj.bias.data.copy_(query_b.data)