import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib.ticker as tick


def produce_ratio_plot_P2(d, color_ = 'orange'):
    plt.plot(d['rt_list'], d['res_list'], color=color_)
    plt.xlabel('Ratio (w1/w2)',size = 13.5)
    plt.ylabel('Jaccard distance',size=13.5)
    plt.yticks([0, 0.2,0.4,0.6,0.8,1],size=12)
    plt.xticks(size=12)
    plt.title(r"$\bf{" +  "a)" + "}$", size = 14.8)
    
def produce_contour_P2(res_dc):
    
    p_tn_data = np.array(res_dc['jac_d']).reshape(res_dc['W1'].shape)
    cp = plt.contourf(res_dc['W1'], res_dc['W2'], p_tn_data,cmap=plt.cm.get_cmap('RdYlGn_r'))
    plt.pcolor(res_dc['W1'], res_dc['W2'], p_tn_data,shading='auto',cmap=plt.cm.get_cmap('RdYlGn_r'))
    v = np.linspace(min(p_tn_data.reshape(-1)), max(p_tn_data.reshape(-1)), len(set(p_tn_data.reshape(-1))), 
                        endpoint=True)
    plt.colorbar(cp, ticks = v).ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('w1',size=13.5)
    plt.ylabel('w2', size= 13.5)
    plt.title(r"$\bf{" + "b)" + "}$" + ' Jaccard distance',size=14.8)
    
def viz_grid(all_d, params_d ={'fig_size':(11.5,7.5),
                               'title_size':20}, save = False, dir2save = None):
    plt.figure(figsize=params_d['fig_size'])
    plt.subplot(2,2,1)
    produce_ratio_plot_P2(all_d['ratio'])
    plt.subplot(2,2,2)
    produce_contour_P2(all_d['contour'])
    ax = plt.subplot(2,1,2)
    d = {n:[] for n in list(set(all_d['box']['diff_n_comp']))}
    for i in range(len(all_d['box']['diff_n_comp'])):
        d[all_d['box']['diff_n_comp'][i]].append(all_d['box']['jac_d'][i])
    data = [d[k] for k in range(len(d))]
    bp = ax.boxplot(data,patch_artist=True,medianprops={"linewidth": 3.5})
    plt.xticks([1,2,3,4,5,6],range(len(d.keys())),size=12)
    plt.yticks(size=12)
    plt.title(r"$\bf{" + 'c)' + "}$", size=14.8)
    ax.set_xlabel('# of observations changed by attack',size=13.5)
    ax.set_ylabel('Jaccard distance', size=13.5)
    plt.suptitle("Poisoning Decoding" ,size = params_d['title_size'],fontweight='bold')
    if save == True:
        plt.savefig(dir2save)
    plt.show()
    
 