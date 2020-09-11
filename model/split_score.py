import torch
import copy
from torch import Tensor


class SplitScore:
    @staticmethod
    def inner_score(attention, bpe_ids, positional_mask, layer_nb: int, head_nb: int = -1) -> Tensor:
        NotImplemented

    @staticmethod
    def cross_score(attention, bpe_ids, positional_mask, layer_nb: int, head_nb: int = -1) -> Tensor:
        NotImplemented

    @staticmethod
    def _reduce_matrix(bpe_ids, attention):
        NotImplemented

def relevance_score(hiddens,attens,rev_score_type):
    """ 
    get relevance scores according to rev_score_type\\
    params
        hiddens:Tensor (seq,h)
        attens: Tensor (seq,seq)
        rev_score_type: str, choices=['attention','L2']
    return span scores: Tensor (seq,seq)"""
    if rev_score_type=='attention':
        return torch.log(attens+1e-6)
    elif rev_score_type=='L2':
        seq_len,hsz=hiddens.shape
        distance=hiddens.unsqueeze(0)-hiddens.unsqueeze(1)
        distance=(distance**2).sum(dim=-1)/hsz
        distance=-torch.log(torch.softmax(distance**0.5,dim=-1))
        return distance

def split_score(hiddens, attens, bpe_mask, rev_score_type='attention' ,norm=False, inner_only=False):
    """ 
    compute split scores, the hiddens and attens should not contain bpe\\
    @params
        hiddens:Tensor (seq,h)
        attens: Tensor (seq,seq)
        rev_score_type: str, choices=['attention','L2']
    return split_scores: Tensor (seq,seq,seq) """
    m = attens.shape[-1]
    rev_scores=relevance_score(hiddens,attens,rev_score_type)
    rev_scores_T = rev_scores.transpose(-1, -2)
    sum_of_attention = torch.zeros(m, m)
    split_score = torch.zeros(m, m, m).to(attens.device)
    for i in range(m):
        for j in range(i,m):
            if i <= j:
                if i>0:
                    left = rev_scores[i:(j+1), 0:i].sum(dim=(-1, -2))+rev_scores_T[i:(j+1), 0:i].sum(dim=(-1, -2))
                else:
                    left = 0
                if j+1 < m:
                    right = (rev_scores[i:(j+1), (j+1):m].sum(dim=(-1, -2))+rev_scores_T[i:(j+1), (j+1):m].sum(dim=(-1, -2)))
                else:
                    right = 0
                scale1 = (j-i+1)**2
                scale2 = 2*(m-(j-i+1))*(j-i+1) if m != (j-i+1) else 1
                sum_of_attention[i, j] = rev_scores[i:(j+1), i:(j+1)].sum(dim=(-1, -2))/scale1  
                if not inner_only:
                    sum_of_attention[i, j] = sum_of_attention[i,j]-(left+right)/scale2
    for i in range(m):
        for j in range(i+1, m):
            for k in range(i, j):
                split_score[i, j, k] = sum_of_attention[i,k]+sum_of_attention[k+1,j]
            if norm:
                split_score[i, j, i:j]=torch.log_softmax(split_score[i, j, i:j],dim=-1)
            split_score[i, j, i:j]+=split_score[i,j,i:j].masked_fill_(bpe_mask[i+1:j+1],float('-inf'))
            # split_score=torch.log(split_score[i, j, i:j])
    return split_score

def remove_bpe_from_hiddens(bpe_ids, hiddens):
    """
    remove bpe id of one sentence
    args
        bpe_ids: List
        hiddens/attention: Tensor (seq,h)
    """
    if len(hiddens)==0: return hiddens
    seq_len=hiddens.shape[0]
    device=hiddens.device
    non_bpe_ids=[]
    cnt=1
    no_bpe_hiddens=hiddens.clone()
    if isinstance(bpe_ids,tuple): bpe_ids=bpe_ids[0]
    for i in reversed(range(seq_len)):
        if i not in bpe_ids:
            non_bpe_ids.append(i)
            cnt=1
        else:
            cnt+=1
            no_bpe_hiddens[i-1]=no_bpe_hiddens[i]*(cnt-1)/cnt+no_bpe_hiddens[i-1]/cnt
    non_bpe_ids.reverse()
    no_bpe_hiddens=torch.index_select(no_bpe_hiddens,dim=0,index=torch.tensor(non_bpe_ids).to(device))
    return no_bpe_hiddens

def remove_bpe_from_attention(bpe_ids, attention):
    """
    remove bpe id of one sentence
    args
        bpe_ids: List
        attention: Tensor (seq,seq)
    """
    if len(bpe_ids)==0: return attention
    seq_len=attention.shape[0]
    device=attention.device
    no_bpe_attention=attention.clone()
    non_bpe_ids=[]
    cnt=1
    if isinstance(bpe_ids,tuple):  bpe_ids=bpe_ids[0]
    for i in reversed(range(seq_len)):
        if i not in bpe_ids:
            non_bpe_ids.append(i)
            cnt=1
        else:
            cnt+=1
            no_bpe_attention[i-1]=no_bpe_attention[i]*(cnt-1)/cnt+no_bpe_attention[i-1]/cnt
            no_bpe_attention[:,i-1]=(no_bpe_attention[:,i]+no_bpe_attention[:,i-1])
    non_bpe_ids.reverse()
    assert len(non_bpe_ids)==(seq_len-len(bpe_ids))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=0,index=torch.tensor(non_bpe_ids).to(device))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=1,index=torch.tensor(non_bpe_ids).to(device))
    return no_bpe_attention
