import numpy as np
import random
import torch

def _get_entropy(score):
    score=np.exp(score)
    dist=score/score.sum() # softmax
    entropy=-(dist*np.log(dist+1e-6)).sum() # -sum(plogp)
    return entropy

def parse_cyk(split_score,sent):
    """ 
    parse tree using cyk
    * split_score: Tensor, (m,m,m)
    * sents: list
    """
    def build_tree(splits,sent,i,j):
        if i==j:
            return sent[i]
        if torch.all(split_score[i,j,i:j]==float('-inf')):
            return ' '.join(sent[i:j+1]).replace(' ##','')
        tree=[]
        k=int(splits[i,j])
        assert k!=-1
        tree.append(build_tree(splits,sent,i,k))
        tree.append(build_tree(splits,sent,k+1,j))
        return tree

    L=len(sent)
    splits=np.zeros([L,L])-1
    tree_score=np.zeros([L,L])
    for w in range(2,L+1): # [2,L]
        for i in range(L):
            j=i+w-1
            if j>=L: continue
            maxv=float('-inf')

            if i!=0 and split_score[i-1,i,i-1].item()==float('-inf'): 
                continue
            if j!=L-1 and split_score[j,j+1,j].item()==float('-inf'): 
                continue

            if torch.all(split_score[i,j,i:j]==float('-inf')):
                tree_score[i,j]=0
            else:
                for k in range(i,j): # [i,j)
                    if split_score[i,j,k].item()==float('-inf') or tree_score[i,k].item()==float('-inf') or \
                        tree_score[k+1,j].item()==float('-inf'): continue
                    s=split_score[i,j,k].item()
                    s+=tree_score[i,k].item()+tree_score[k+1,j].item()
                    
                    if maxv<s:
                        splits[i,j]=k
                        maxv=s
                tree_score[i,j]=maxv
    tree=build_tree(splits,sent,0,L-1) 
    return tree
        

def parse_greedy(split_score,sent):
    """ 
    parse tree using greedy
    * split_score:(bsz,m,m)

    """
    def parse(i,j):
        if i==j:
            return sent[i]
        if torch.all(split_score[i,j,i:j]==float('-inf')):
            return ' '.join(sent[i:j+1]).replace(' ##','')
        else:
            tree=[]
            max_ix=torch.argmax(split_score[i,j,i:j])+i
            tree.append(parse(i,max_ix))
            tree.append(parse(max_ix+1,j))
        return tree
    return parse(0,len(sent)-1)


def random_tree(sent):
    if len(sent)==1: return sent[0]
    split_idx=random.choice(range(len(sent)-1))
    tree=[]
    tree.append(random_tree(sent[:split_idx+1]))
    tree.append(random_tree(sent[split_idx+1:]))
    return tree


if __name__ == "__main__":
    from dataloader.data import tree2dis
    from utils.parse_utils import comp_tree

    sample_nb,seq_len=100,20
    sents = [list(range(seq_len))]*sample_nb
    trees = [random_tree(s) for s in sents]

    
    dis=np.zeros([sample_nb,seq_len,seq_len])
    for i,t in enumerate(trees):
        dis[i]=tree2dis(t)+1
    dis=np.log(dis)

    split_scores=np.zeros([sample_nb,seq_len,seq_len,seq_len])
    for i in range(seq_len):
        for j in range(i+1,seq_len):
            for k in range(i,j):
                inner_score=dis[:,i:k+1,i:k+1].sum(axis=(-1,-2))+dis[:,k+1:j+1,k+1:j+1].sum(axis=(-1,-2))
                # cross_score=dis[:,i:k+1,k+1:j+1].sum(axis=(-1,-2))+dis[:,k+1:j+1,i:k+1].sum(axis=(-1,-2))
                split_scores[:,i,j,k]=inner_score

    parsed_trees = parse_greedy(-split_scores,sents)

    prec_list,reca_list,f1_list=[],[],[]
    for pred_tree, tgt_tree in zip(parsed_trees,trees):
        prec,reca,f1=comp_tree(pred_tree,tgt_tree)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)
    
    mean_of_list=lambda _list: sum(_list)/len(_list)
    print(mean_of_list(prec_list),mean_of_list(reca_list),mean_of_list(f1_list))
    
