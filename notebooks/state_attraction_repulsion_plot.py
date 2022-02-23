import numpy as np
from hmm_utils import HMM
from smoothing_state_attacker import ss_attacker
from extra import monte_carlo_enumeration
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import warnings
import random
warnings.filterwarnings("ignore")


def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))

def ratio_plot(X, hmm, t_ , state_ , c_, k_value_, N_ite ,initial_ratio, 
               final_ratio, ratio_step , color_ = 'blue'):
    ## ratio plots
    Z_set = hmm.generate_z(len(X))
    f1_attacker = ss_attacker(w1 = 1 ,w2= 0 ,t = t_, state= state_ ,c=c_)
    f2_attacker = ss_attacker(w1 = 0 ,w2= 1 ,t = t_, state= state_ ,c=c_)
    f1_utilities = monte_carlo_enumeration(hmm, k_value = k_value_, attacker = f1_attacker, N= N_ite, x_vector =X, theta_prob_vec= np.ones(hmm.n_obs))[1]
    f2_utilities = monte_carlo_enumeration(hmm,k_value = k_value_,attacker= f2_attacker, N= N_ite, x_vector = X, theta_prob_vec = np.ones(hmm.n_obs))[1]
    result_list, ratio_list = [], []
    for r in range(int(final_ratio/initial_ratio)):
        utility_vector = initial_ratio * f1_utilities + f2_utilities
        max_indx = np.argmax(utility_vector)
        z_star = Z_set[max_indx]
        y_vec =  hmm.attack_X(X.reshape(len(X),1), np.ones([len(X),hmm.n_obs]),z_star).astype('int')
        compute_prob_att = ss_attacker(w1 = initial_ratio ,w2= 1,t = t_, state=state_ ,c=1)
        ratio_list.append(initial_ratio)
        p_tainted_data = compute_prob_att.state_attraction_repulsion_f1(alpha = hmm.alpha(y_vec), beta = hmm.beta(y_vec))
        result_list.append(p_tainted_data)
        initial_ratio += ratio_step
        
    p_untainted_data = compute_prob_att.state_attraction_repulsion_f1(alpha = hmm.alpha(X), beta = hmm.beta(X))
    plt.axhline(y = p_untainted_data, color = color_, linestyle = ':', label = 's= '+ str(state_)+' unt. data')
    plt.plot(ratio_list, result_list, label = 's= '+ str(state_) ,color=color_)
    plt.xlabel('Ratio (w1/w2)')
    plt.ylabel('Prob. tainted data')
    plt.title('c = ' + str(c_) + ',t = ' +  str(t_)  + ',N_ite=' + str(N_ite)  + ',k=' +  str(k_value_))


def compute_contour_w1_w2(hmm, X ,t_, state_, N_ite, k_value_, c_):
    
    Z_set = hmm.generate_z(len(X))
    f1_attacker = ss_attacker(w1 = 1 ,w2= 0 ,t = t_, state= state_ ,c=c_)
    f2_attacker = ss_attacker(w1 = 0 ,w2= 1 ,t = t_, state= state_ ,c=c_)
    f1_utilities = monte_carlo_enumeration(hmm, k_value = k_value_,
                                           attacker = f1_attacker, N= N_ite, 
                                           x_vector =X, theta_prob_vec = np.ones(hmm.n_obs))[1]
    f2_utilities = monte_carlo_enumeration(hmm,k_value = k_value_,
                                           attacker= f2_attacker, N= N_ite, x_vector = X, 
                                           theta_prob_vec = np.ones(hmm.n_obs))[1]
    result_list = []
    start, stop, n_values = 0, 10, 100
    w1_vals = np.linspace(start, stop, n_values)
    w2_vals = np.linspace(start, stop, n_values)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    W1_loop = W1.reshape(-1)
    W2_loop = W2.reshape(-1)
    
    for element in tqdm(range(len(W1_loop))):
            utility_vector = W1_loop[element] * f1_utilities + W2_loop[element] * f2_utilities
            max_indx = np.argmax(utility_vector)
            z_star = Z_set[max_indx]
            y_vec =  hmm.attack_X(X.reshape(len(X),1), np.ones([len(X), hmm.n_obs]), z_star).astype('int')
            compute_prob_att = ss_attacker(w1 = W1_loop[element] ,w2= W2_loop[element],t = t_, state=state_ ,c=1)
            p_tainted_data = compute_prob_att.state_attraction_repulsion_f1(alpha = hmm.alpha(y_vec), beta = hmm.beta(y_vec))
            result_list.append(p_tainted_data)
    
    y_title_str = 'Prob tainted data '        
    Z_l = np.array(result_list).reshape(W1.shape)
    
    if c_==1:
        cp = plt.contourf(W1, W2, Z_l,cmap=plt.cm.get_cmap('RdYlGn_r'))
        plt.pcolor(W1, W2, Z_l,shading='auto',cmap=plt.cm.get_cmap('RdYlGn_r'))
    elif c_==-1:
        cp = plt.contourf(W1, W2, Z_l,cmap=plt.cm.get_cmap('RdYlGn'))
        plt.pcolor(W1, W2, Z_l,shading='auto',cmap=plt.cm.get_cmap('RdYlGn'))
        
    compute_prob_att = ss_attacker(w1 = 10 ,w2= 0 ,t = t_, state=state_ ,c=1)
    p_untainted_data = compute_prob_att.state_attraction_repulsion_f1(alpha = hmm.alpha(X), beta = hmm.beta(X))
    print('Prob. untainted data-----------------', p_untainted_data)
    v = np.linspace(min(Z_l.reshape(-1)), max(Z_l.reshape(-1)), len(set(Z_l.reshape(-1))), endpoint=True)
    plt.colorbar(cp, ticks = v)
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('Contour,'+ y_title_str +', c= '+str(c_)+' state='+str(state_)+',t='+str(t_))
    plt.show()

