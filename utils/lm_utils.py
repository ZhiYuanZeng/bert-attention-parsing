import torch
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def log_scores(attention, epsilon=10**(-5)):
    """ 
    parse tree from attention matrix
    * attention: tensor(bsz,head,m,m)
    """
    # scores = torch.zeros_like(attention)

    a=attention.mean(dim=0) # mean on head, to save computation
    m = len(torch.nonzero(a[0]))
    a=a[1:m-1,1:m-1] # remove special tokens(cls,sep,pad)
    m=len(a)
    scores=torch.zeros_like(a)

    a = torch.log(a)
    a_T = a.transpose(-1, -2)

    for i in range(m):
        for j in range(m):
            if i <= j:
                left = a[i:(j+1), 0:i].sum(dim=(-1, -2)) + \
                    a_T[i:(j+1), 0:i].sum(dim=(-1, -2)) if i > 0 else 0
                right = (a[i:(j+1), (j+1):m].sum(dim=(-1, -2))+a_T[i:(j+1), (j+1):m].sum(dim=(-1, -2))) \
                    if j+1 < m else 0
                scale1 = (j-i+1)**2
                scale2 = 2*(m-(j-i+1))*(j-i+1) if m != (j-i+1) else 1
                scores[i, j] = a[i:(
                    j+1), i:(j+1)].sum(dim=(-1, -2))/scale1-(left+right)/scale2
            else:
                scores[i, j] = scores[j, i]

    return scores

def parse_cyk(score,sent):
        """ 
        parse tree using cyk
        * scores:(bsz,m,m)

        """
        def build_tree(splits,sent,gram_dis,i,j):
            if i==j:
                return sent[i]
            tree=[]
            k=int(splits[i,j])
            gram_dis[i:(k+1),i:(k+1)]+=1 # [i,k]
            gram_dis[(k+1):(j+1),(k+1):(j+1)]+=1 # [k+1,j]

            tree.append(build_tree(splits,sent,gram_dis,i,k))
            tree.append(build_tree(splits,sent,gram_dis,k+1,j))
            return tree


        assert len(score)==len(sent)
        L=len(score)
        splits=np.zeros([L,L])
        for w in range(2,L+1): # [2,L]
            for i in range(L):
                j=i+w-1
                if j>=L:
                    continue
                maxv=float('-inf')
                for k in range(i,j): # [i,j)
                    if maxv<score[i,k]+score[k+1,j]:
                        splits[i,j]=k
                        maxv=score[i,k]+score[k+1,j]
                score[i,j]+=maxv
        gram_dis=torch.zeros(L,L)
        tree=build_tree(splits,sent,gram_dis,0,L-1) 
        return tree,gram_dis