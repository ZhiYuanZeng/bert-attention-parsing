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
        return torch.log(attens)
    elif rev_score_type=='L2':
        seq_len,hsz=hiddens.shape
        distance=hiddens.unsqueeze(0)-hiddens.unsqueeze(1)
        distance=(distance**2).sum(dim=-1)/hsz
        distance=-torch.log(torch.softmax(distance**0.5,dim=-1))
        return distance

def split_score(hiddens, attens, bpe_ids, rev_score_type='attention' ,norm=False, inner_only=False):
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
    split_score = torch.zeros(m, m, m)
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
            no_bpe_hiddens[i-1]=no_bpe_hiddens[i]/cnt+no_bpe_hiddens[i-1]*(cnt-1)/cnt
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
            no_bpe_attention[i-1]=no_bpe_attention[i]/cnt+no_bpe_attention[i-1]*(cnt-1)/cnt
            no_bpe_attention[:,i-1]=(no_bpe_attention[:,i]+no_bpe_attention[:,i-1])
    non_bpe_ids.reverse()
    assert len(non_bpe_ids)==(seq_len-len(bpe_ids))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=0,index=torch.tensor(non_bpe_ids).to(device))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=1,index=torch.tensor(non_bpe_ids).to(device))
    return no_bpe_attention

class AttentionScore(SplitScore):
    """ get split score from attention matrix """
    @staticmethod
    def inner_score(attention:Tensor, bpe_ids:list, positional_mask:Tensor=None):
        """ 
        split score is the product of attention score of two spans,
        divided by the product of cross attention
        * attention: (bsz,head,m,m)
        * bpe_ids: list of id of bpe token
        * pos_mask: (m,m)
        * head_nb: head number
        """
        attention=copy.deepcopy(attention).to(attention.device)
        split_score_list = []

        for a, ids in zip(attention, bpe_ids):  # iterate on batch,
            a = AttentionScore. _reduce_matrix(ids, a)
            m = a.shape[-1]
            if positional_mask is not None:
                local_pos_mask = positional_mask[:m, :m]
                a = a*local_pos_mask
            a = torch.log(a)
            a_T = a.transpose(-1, -2)

            sum_of_attention = torch.zeros(m, m)
            split_score = torch.zeros(m, m, m)

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
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(i, j):
                        split_score[i, j, k] = sum_of_attention[i,k]+sum_of_attention[k+1,j]
                    # split_score[i, j]=torch.sotmax(split_score[i, j],dim=-1)

            # visual_matrix(a[0],'figures/score.png',dpi=600)
            split_score_list.append(split_score)

        return split_score_list

    @staticmethod
    def cross_score(attention, bpe_ids, head_nb=-1, positional_mask=None):
        """ 
        split score is the the product of cross attention between spans.
        * attention:(bsz,head,m,m)
        * bpe_ids: list of id of bpe token
        * pos_mask: (m,m)
        * layer_nb: layer number
        * head_nb: head number
        """
        attention=copy.deepcopy(attention).to(attention.device)
        if head_nb != -1:
            attention = attention[:, head_nb, :, :]
        else:
            attention = attention.mean(dim=1)
        split_score_list = []

        for a, ids in zip(attention, bpe_ids):  # iterate on batch,
            a = AttentionScore._reduce_matrix(ids, a)
            m = a.shape[-1]
            if positional_mask is not None:
                local_pos_mask = positional_mask[:m, :m]
                a = a*local_pos_mask
            a = torch.log(a)

            split_score = torch.zeros(m, m, m)
            span_scores = torch.zeros(m, m)
            for i in range(m):
                for j in range(i, m):
                    sum_of_attention = a[i:(j+1), i:(j+1)].sum()
                    span_scores[i, j] = sum_of_attention

            for i in range(m):
                for j in range(i+1, m):
                    for k in range(i, j):
                        # inner_attention = span_scores[i, k] / ((k-i+1)**2) + span_scores[k+1, j]/((j-k)**2)
                        cross_attention = (
                            span_scores[i, j]-span_scores[i, k]-span_scores[k+1, j]) / (2*(j-k)*(k-i+1))
                        split_score[i, j, k] = cross_attention
                    split_score[i,j]=split_score[i,j]/split_score[i,j].abs().sum(dim=-1) # log prob
            split_score_list.append(split_score)

        return split_score_list

    @staticmethod
    def inner_and_cross_score(attention, bpe_ids, head_nb=-1, positional_mask=None):
        _inner_score=AttentionScore.inner_score(attention, bpe_ids, head_nb=-1)
        _cross_score=AttentionScore.cross_score(attention, bpe_ids, head_nb=-1)
        scores=[]
        for i,c in zip(_inner_score,_cross_score):
            scores.append(i+c)
        return scores

    @staticmethod
    def emsemble_cross_heads(attention, bpe_ids, head_nb=-1, positional_mask=None):
        assert isinstance(attention, list)
        

    @staticmethod
    def _reduce_matrix(bpe_ids: list, attention: Tensor):
        """ 
        del word piece and pad index from attention matrix
        attention size:(head, m,m)
        """
        non_zero_count = len(attention[0].nonzero())
        # reduced_a=attention[:non_zero_count,:non_zero_count] # remove pad
        ids = list(range(non_zero_count))
        for i in sorted(bpe_ids, reverse=True):
            if i > 0:
                attention[i-1, :] = (attention[i-1, :]+attention[i, :])/2
                attention[:, i-1] += attention[:, i]
            del ids[i]
        attention = attention[torch.LongTensor(ids), :]
        attention = attention[:, torch.LongTensor(ids)]
        attention = attention[1:-1, 1:-1]  # remove cls,sep
        return attention        


class HiddenScore(SplitScore):
    """ get split score from hidden states """
    @staticmethod
    def inner_score(hiddens, bpe_ids, attention_mask, positional_mask=None,) -> Tensor:
        """ 
        split score is the product of L2 distance of words in the span
        """
        batch_size, seq_len, hidden_size = hiddens.shape  # hiddens: [bsz,len,h],
        pad_embed = torch.zeros(hidden_size) # what pad embedding is does not effect parsing

        hiddens, sequence_lengths = HiddenScore._reduce_matrix(
            bpe_ids, pad_embed, hiddens, attention_mask)

        expand_shape=(batch_size,seq_len,seq_len,hidden_size)
        distance = torch.dist(
            hiddens.unsqueeze(1).expand(expand_shape),hiddens.unsqueeze(2).expand(expand_shape))/hidden_size
        
        split_scores_list=[]

        if positional_mask is not None:
            local_pos_mask = positional_mask[:seq_len, :seq_len]
            distance = distance*local_pos_mask

        distance = -torch.log(distance+1e-6)

        sum_of_distance = torch.zeros(batch_size, seq_len, seq_len)

        for i in range(seq_len):
            for j in range(i, seq_len):
                scale1 = (j-i+1)**2
                sum_of_distance[:, i, j] = distance[:, i:j+1, i:j+1].sum(dim=(-1, -2))/scale1

        for sample_idx, length in enumerate(sequence_lengths):
            split_scores = torch.zeros(length, length, length)
            for i in range(length):
                for j in range(i+1, length):
                    for k in range(i, j):
                        split_scores[i, j, k] = (sum_of_distance[sample_idx,i, k] + \
                            sum_of_distance[sample_idx, k+1, j]) / ((j-k)**2 + (k-i+1)**2)
            split_scores_list.append(split_scores)
        return split_scores_list

    @staticmethod
    def cross_score(hiddens: Tensor, bpe_ids: list, attention_mask: Tensor) -> Tensor:
        """ split score is the distance of two span hiddens  
            hiddens: tensor, [bsz,len,h]
            bpe_ids: [[]], list of list of bpe id
            attention_mask: tensor, [bsz,len]
        """
        batch_size, seq_len, hidden_size = hiddens.shape
        # what pad embedding is does not effect parsing
        pad_embed = torch.zeros(hidden_size)
        hiddens, sequence_lengths = HiddenScore._reduce_matrix(
            bpe_ids, pad_embed, hiddens, attention_mask)

        span_hiddens = torch.zeros(  # span_hiddens: [bsz,len,len,h]
            batch_size, seq_len, seq_len, 4*hidden_size).to(hiddens.device)

        for i in range(seq_len):
            for j in range(i, seq_len):
                span_hiddens[:, i, j] = SpanEmbedding.difference_of_head_tail(
                    hiddens, i, j)  # compute span embedding

        split_scores = torch.zeros(batch_size, seq_len, seq_len, seq_len)

        for i in range(seq_len):
            for j in range(i+1, seq_len):
                if j >= max(sequence_lengths):
                    continue
                for k in range(i, j):
                    # split_scores[:,i,j,k]=(span_hiddens[:,i,k]*span_hiddens[:,k+1,j]).sum(dim=-1) # inner product
                    split_scores[:, i, j, k] = -torch.dist(  # squared L2 distance
                        span_hiddens[:, i, k], span_hiddens[:, k+1, j])

        return split_scores/(4*hidden_size)

    @staticmethod
    def _reduce_matrix(bpe_ids: list, pad_embed: Tensor, hiddens: Tensor, attention_mask: Tensor) -> Tensor:
        """ 
        remove word piece and pad index from hidden states
        - bpe_ids: [[]], list of list of word piece token id
        - pad_embed: int, pad token id
        - hiddens: tensor, (bsz,len,h)
        - attention_mask: Tensor, (bsz,len)
        """
        non_zero_count = len(hiddens[0].nonzero())
        bsz, seq_len, hidden_size = hiddens.shape
        device = hiddens.device
        hiddens = copy.deepcopy(hiddens).to(device)
        reduced_hiddens = pad_embed.repeat(bsz, seq_len, 1).to(device)
        sequence_lengths = [len(mask.nonzero()) for mask in attention_mask]
        for batch_no, ids in enumerate(bpe_ids):
            indices = list(range(sequence_lengths[batch_no]))
            last_id, continous_count = -1, 1
            for i in sorted(ids, reverse=True):
                if i <= 0:
                    continue

                hiddens[batch_no, i-1] = hiddens[batch_no, i-1] + \
                    hiddens[batch_no, i]
                if last_id-i == 1:  # continous
                    continous_count += 1
                elif last_id != -1:  # note that the rightest index is particular
                    hiddens[batch_no, last_id-1] = hiddens[batch_no,
                                                           last_id-1]/(continous_count+1)
                    continous_count = 1
                last_id = i
                del indices[i]
            hiddens[batch_no, last_id-1] = hiddens[batch_no,
                                                   last_id-1]/(continous_count+1)

            reduced_hiddens[batch_no, :len(
                indices)] = hiddens[batch_no][indices]
            sequence_lengths[batch_no] -= len(ids)
            assert (reduced_hiddens[batch_no,
                                    sequence_lengths[batch_no]].sum()) == 0
        return reduced_hiddens, sequence_lengths


class SpanEmbedding:
    """ 
    compute embedding of span 
    * input:
        hiddens:Tensor, [bsz,seq,h]
        beg:int, begin index of span 
        end:int, end index of span
    * output:Tensor, [bsz,h]
    """
    @staticmethod
    def max_pool(hiddens, beg, end):
        return torch.max(hiddens[:, beg:(end+1)])

    @staticmethod
    def difference_of_head_tail(hiddens, beg, end):
        """ [h[beg];h[end];h[beg] x h[end];h[beg]-h[end]] """
        return torch.cat([
            hiddens[:, beg] * hiddens[:, end], hiddens[:, beg]-hiddens[:, end]
        ], dim=1)

    @staticmethod
    def avg_pool(hiddens, beg, end):
        return torch.means(hiddens[:, beg:(end+1)])

    @staticmethod
    def funcname(parameter_list):
        pass


if __name__ == "__main__":
    bpe_ids = [[1, 3, 4], [1, 2, 5]]
    hiddens = torch.ones(2, 6, 6)
    pad_embed = torch.zeros(6)
    reduced_hiddens = HiddenScore._reduce_matrix(bpe_ids, pad_embed, hiddens)
    print(reduced_hiddens)
