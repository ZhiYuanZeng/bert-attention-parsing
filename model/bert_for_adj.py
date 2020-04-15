import torch
from model.bert_head import BertHead
from random import randint

class BertForAdj(BertHead):
    def __init__(self, bert, hidden_size, head_num, layer_nb, negative_sample, dropout_rate, rm_bpe_from_a):
        super(BertForAdj,self).__init__(bert, hidden_size, head_num, layer_nb, dropout_rate, rm_bpe_from_a)
        self.negative_sample=negative_sample
    
    def forward(self, x, attention_mask, epsilon=1e-6):
        _,_,all_hiddens,_=self.bert(x,attention_mask)
        hiddens=all_hiddens[self.layer_nb]
        # hiddens=self.dropout(hiddens)
        query=self.query_proj(hiddens) # (bsz,seq,h//head)
        key=self.key_proj(hiddens) # (bsz,seq,h//head)
        bsz, hidden_size=hiddens.shape[0],hiddens.shape[-1]
        attention=torch.matmul(query,key.transpose(-1,-2)).masked_fill_(~attention_mask.bool().unsqueeze(1), float('-inf'))
        attention=attention/(hidden_size**0.5)
        attention=torch.softmax(attention,dim=-1)
        attention=torch.log(attention+epsilon)
        loss, sample_count=0,0
        acc=[]
        for i in range(bsz):
            seq_len=len(attention_mask[i].nonzero())
            if seq_len<5: continue
            for j in range(seq_len):
                l=randint(2,seq_len-3)
                r=randint(l,seq_len-3)
                left_span=(randint(0,l-2),l-1)
                assert left_span[1]-left_span[0]>0
                assert left_span[1]==l-1
                distance1=self._get_distance(attention[i], left_span, (l,r))
                loss+=torch.log(torch.sigmoid(distance1))
                negative_sample=max(self.negative_sample, left_span[1]-left_span[0])
                for k in range(negative_sample):
                    negative_span=(left_span[0],randint(left_span[0],left_span[1]-1))
                    distance2=self._get_distance(attention[i], negative_span, (l,r))
                    loss+=torch.log(torch.sigmoid(-distance2))/negative_sample
                    acc.append(distance1>distance2)

                right_span=(r+1,randint(r+2, seq_len-1))
                assert right_span[1]-right_span[0]>0
                distance1=self._get_distance(attention[i], (l,r), right_span)
                loss+=torch.log(torch.sigmoid(distance1))
                negative_sample=max(self.negative_sample, right_span[1]-right_span[0])
                for k in range(negative_sample):
                    negative_span=(randint(right_span[0]+1,right_span[1]),right_span[1])
                    distance2=self._get_distance(attention[i], (l,r), negative_span)
                    loss+=torch.log(torch.sigmoid(-distance2))/negative_sample
                    acc.append(distance1>distance2)
                sample_count+=2
        assert sample_count != 0
        return -loss/sample_count, sum(acc).item()/len(acc)
    
    def _get_distance(self, attention, span1, span2):
        l1,r1=span1
        l2,r2=span2
        assert l1<=r1 and l2<=r2 and r1<=l2
        return (attention[l1:r1+1,l2:r2+1].sum()+attention[l2:r2+1,l1:r1+1].sum())/(2*(r1-l1+1)*(r2-l2+1))