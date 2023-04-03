import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .Linear_Inverse_Model import LIM

def auto_corr(PCs: np.ndarray, tau_list: int, plot: bool = False, **plot_additional):
    corr_list = np.zeros(len(tau_list))
    for i, tau in enumerate(tau_list):
        if tau == 0:
            corr_list[i] = pearsonr(PCs[1], PCs[1])[0]
        else:
            corr_list[i] = pearsonr(PCs[1, :-tau], PCs[1, tau:])[0]
    if plot:
        plt.figure(figsize = (8, 6))
        plt.scatter(tau_list, corr_list, **plot_additional)
        plt.title("auto correlation", fontsize = 16)
        plt.show()
    else:
        return corr_list

def _find_G(PCs: np.ndarray, tau: int):
    LIM_MJO = LIM(PCs[:, 0], PCs, tau, 1)
    LIM_MJO.build()
    return LIM_MJO.G

def tau_test(PCs: np.ndarray, tau_list: np.ndarray, plot: bool = False, **plot_additional):
    G_list = np.zeros((len(tau_list), PCs.shape[0], PCs.shape[0]))
    for i, tau in enumerate(tau_list):
        G_list[i] = _find_G(PCs, tau)
    if plot:
        plt.figure(figsize = (8, 6))
        plt.scatter(tau_list, G_list[:, 0, 0], **plot_additional)
        plt.title("tau test", fontsize = 16)
        plt.show()
    else:
        return G_list
