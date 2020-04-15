'''
visualize the f1 score of different experiments
'''

import matplotlib.pyplot as plt
import seaborn
import numpy as np

def visual_hiddens(querys, keys, texts):
    '''
    query: [seq,2], key: [seq,2]
    '''
    if len(keys)>8: keys,querys,texts=keys[:8],querys[:8],texts[:8],
    if len(keys)<4:
        row,col=1,len(keys)
    else:
        row,col=len(keys)//4,4
    fig, axs = plt.subplots(row,col, figsize=(10*col, 10*row))
    axs=axs.reshape(-1)
    for i in range(row*col):
        query,key,text,ax=querys[i],keys[i],texts[i],axs[i]
        assert len(query)==len(key)==len(text)
        ax.plot(query[:,0],query[:,1],'o',color='red',label='query')
        ax.plot(key[:,0],key[:,1],'s',color='blue',label='key')
        for i in range(len(query)):
            k,q,t=key[i],query[i],text[i]
            ax.annotate(t, (k[0], k[1]))
            ax.annotate(t, (q[0],q[1]))
        ax.legend(numpoints=1)
    # plt.show()
    plt.savefig('plot_hiddens.png')
    plt.close()

def visualize_distances(distances:np.ndarray):
    """
    visualize distances of layers and heads
        distances: [len,len]
    """
    # visualize distances of differnet layer
    layers=distances.mean(axis=1) 
    visual_vec(layers,'figures/distances/layers.png',x_labels='layer',y_labels='L2 distance')

    # visualize heads
    for layer_nb,layer in enumerate(distances):
        visual_vec(layer,f'figures/distances/layers{layer_nb}.png',x_labels='head',y_labels='L2 distance')

def visual_matrix(data,figure_name='unnamed-matrix-graph',x_labels=None,y_labels=None,**plt_kwargs):
    if isinstance(data,torch.Tensor):
        data=data.cpu().numpy()
    fig,ax=plt.subplots()
    im=ax.imshow(data)
    if x_labels is not None and y_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(figure_name,**plt_kwargs)
    plt.close()

def visual_vec(data,figure_name='unnamed-vector-graph',x_labels=None,y_labels=None):
    if isinstance(data,torch.Tensor):
        data=data.cpu().numpy()
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.bar(range(len(data)),data)
    plt.savefig(figure_name)
    plt.close()

def visualize_f1_of_heads(f1_scores, layer_nb, method):
    """ f1_scores: numpy array, [head,f1] """
    visual_vec(f1_scores, f'figures/f1/{method}_{layer_nb}.png', x_labels='head', y_labels='f1')

def visual_attention(a, s):
    if len(a)<4:
        row,col=1,len(a)
    else:
        row,col=len(a)//4,4
    fig, axs = plt.subplots(row,col, figsize=(10*col, 10*row))
    axs=axs.reshape(-1)
    for i in range(row*col):
        seaborn.heatmap(a[i], 
                        xticklabels=s[i], square=True, yticklabels=s[i], vmin=0.0, vmax=1.0, 
                        cbar=False,ax=axs[i])
    plt.savefig('attention.png')
    plt.close()

if __name__ == "__main__":
    x=np.ones([10,10,10])
    s=[1]*10
    visual_attention(x,s)
