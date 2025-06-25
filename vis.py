import argparse
import torch
import os.path as osp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def normalize(feature):
    norm = np.sqrt((feature**2).sum(1, keepdims=True))
    return feature / (norm + 1e-12)


def main():
    
    #file = torch.load('./embed_alg.pt')
    file = torch.load('./embed.pt')

    embed = file['embed']
    domain = file['domain']
    labels = file['label']
    #dim = embed.shape[1] // 2
    #embed = embed[:, dim:]

    #domain = file['label']
    cnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    dnames = ['Art', 'Cartoon', 'Photo', 'Sketch']
    #print(domain)
    embed = normalize(embed)
    print('Loaded features with shape {}'.format(embed.shape))

    embed2d_path = osp.join('./', 'embed2d' + '.pt')

    
        
    print('Dimension reduction with t-SNE (dim=2) ...')
    #if osp.exists(embed2d_path):
    #    embed2d = torch.load(embed2d_path)
    #else:
    tsne = TSNE(
        n_components=2, metric='euclidean', verbose=1,
        perplexity=50, n_iter=1000, learning_rate=200.
    )
    embed2d = tsne.fit_transform(embed)

    #    torch.save(embed2d, embed2d_path)
    #    print('Saved embed2d to "{}"'.format(embed2d_path))



    avai_domains = list(set(domain.tolist()))
    avai_domains.sort()
    avai_categ = list(set(labels.tolist()))
    avai_categ.sort()
    print('Plotting ...')

    SIZE = 4
    COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    LEGEND_MS = 3

    fig, ax = plt.subplots()

    #for d in avai_domains:
    for cat in avai_categ:
        #d = int(d)
        cat = int(cat)
        #e = embed2d[domain == d]
        e = embed2d[labels == cat]

       
        #label = dnames[d]
        label = cnames[cat]
        

        ax.scatter(
            e[:, 0],
            e[:, 1],
            s=SIZE,
            c=COLORS[cat],  #d
            edgecolors='none',
            label=label,
            alpha=1,
            rasterized=False
        )

    #ax.legend(loc='upper left', fontsize=10, markerscale=LEGEND_MS)
    #ax.legend(fontsize=10, markerscale=LEGEND_MS)
    ax.set_xticks([])
    ax.set_yticks([])
    #LIM = 22
    #ax.set_xlim(-LIM, LIM)
    #ax.set_ylim(-LIM, LIM)

    figname = 'embed.pdf'
    fig.savefig(osp.join('./', figname), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()