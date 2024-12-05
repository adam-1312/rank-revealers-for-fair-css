from numpy import genfromtxt, full, vstack
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_over_k(q, k, label, ax=None, leg="q", line_style='.-'):
    if not ax:
        fig, ax = plt.subplots()
    
    ax.plot(k, q, line_style, label=leg)
    ax.set_xticks(range(1,37,2))
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(label)
    ax.legend(prop={'size': 10})
    

def visualize_mask(mask, dim=(28,28)):
    im = full((1, dim[0]*dim[1]), 255)
    im[0, mask] = 0
    plt.matshow(im.reshape(dim), cmap='gray_r')
    plt.show()

def joint_histogram(quantity1=None, quantity2=None, k=1, labels=['q1','q2'], y_bottom=1, y_top=None, ax=None, quantities=None):
    '''
    See     
    '''
    
    if (not quantities) and quantity1.any() and quantity2.any():
        quantities = [quantity1, quantity2]
    
    width = 0.3  # the width of the bars
    multiplier = 0
    
    if not ax:
        fig, ax = plt.subplots(layout='constrained')
    
    for i in range(len(quantities)):
        offset = width * multiplier
        rects = ax.bar(np.array(k)+offset, quantities[i], width=width, label=labels[i])
        multiplier += 1
    
    ax.set_xticks(np.array(k) + (width*len(quantities)-width)/2, k)
    
    if not y_top:
        y_top = max(map(max, quantities))*1.1
    
    ax.set_ylim(y_bottom, y_top)
    ax.legend()

def visualize_digit(im, mask=range(784), ax=None):
    im = im.reshape(28, 28)
    im = (np.full(im.shape, 255) - im).astype(np.uint8)
    rgb_image = np.stack([im]*3, axis=-1)/ 255.0

    for col in range(784):
        if col not in mask:
            row, col_index = divmod(col, 28)
            rgb_image[row, col_index] = [1, 0, 0]
    
    if not ax:
        given_ax = False
        fig, ax = plt.subplots()
    
    ax.matshow(rgb_image, cmap='gray_r')
    ax.axis('off')
    
    if not given_ax:
        plt.show()

def mnist_CSS_interpretation():
    from CSSlib import get_mnist, perform_CSS

    X_train, X_test, y_train, y_test = get_mnist()
    y_train2 = list(y_train)
    idx = [y_train2.index(i) for i in range(3,8)]

    cols,_ = perform_CSS(X_train, 32, alg='RRQR')

    fig, axs = plt.subplots(2, 5)
    for a,i in enumerate(idx):
        visualize_digit(X_train[i], ax=axs[0,a])
        visualize_digit(X_train[i], cols, ax=axs[1,a])

    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    # ~ fig.savefig("CSS_interpretation3.png")

def plot_css_results():
    RRQR_results = pd.read_csv('RRQRresults_one_group.csv', index_col=0)
    lowQR_results = pd.read_csv('LowQRresults_one_group.csv', index_col=0)
    fig, axs = plt.subplots(1, 2)
    # Reconstruction error
    plot_over_k(RRQR_results.loc['err'], k=RRQR_results.columns, label='Relative reconstruction error', ax=axs[0], leg="RRQR")
    plot_over_k(lowQR_results.loc['err'], k=lowQR_results.columns, label='Relative reconstruction error', ax=axs[0], leg="Low QR")
    # Classifier Accuracy
    plot_over_k(RRQR_results.loc['acc_noninverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs[1], leg="RRQR")
    plot_over_k(lowQR_results.loc['acc_noninverted'], k=lowQR_results.columns, label='Classifier Accuracy', ax=axs[1], leg="Low QR")
    axs[1].plot(lowQR_results.columns, full(lowQR_results.loc['acc_noninverted'].shape, 0.9713), label='Before CSS')
    axs[1].legend()
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    fig.set_figwidth(13)
    fig.savefig("CSS_results.png")
    plt.show()

def plot_sensitivity_results():
    RRQR_results = pd.read_csv('RRQRresults_one_group_nonrotated.csv', index_col=0) # nonrotated training
    rot_RRQR_results = pd.read_csv('RRQRresults_one_group_rotated.csv', index_col=0) # rotated training
    RRQR_50_50_results = pd.read_csv('RRQRresults_two_groups_50-50.csv', index_col=0)
    RRQR_75_25_results = pd.read_csv('RRQRresults_two_groups_75-25.csv', index_col=0) # 25% rotated
    fig, axs = plt.subplots(1, 1)
    
    # Classifier Accuracy
    plot_over_k(RRQR_results.loc['acc_noninverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="100:0 on original", line_style="x-")
    plot_over_k(RRQR_results.loc['acc_inverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="100:0 on rotated")
    plot_over_k(rot_RRQR_results.loc['acc_noninverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="0:100 on original")
    plot_over_k(rot_RRQR_results.loc['acc_inverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="0:100 on rotated")
    plot_over_k(RRQR_50_50_results.loc['acc_noninverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="50:50 on original")
    plot_over_k(RRQR_50_50_results.loc['acc_inverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="50:50 on rotated")
    plot_over_k(RRQR_75_25_results.loc['acc_noninverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="75:25 on original", line_style="--")
    plot_over_k(RRQR_75_25_results.loc['acc_inverted'], k=RRQR_results.columns, label='Classifier Accuracy', ax=axs, leg="75:25 on rotated", line_style="--")
    axs.legend()
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    fig.set_figwidth(7)
    # ~ fig.savefig("Sensitivity_results.png")
    plt.show()

def plot_css_unfairness():
    RRQR_results = pd.read_csv('RRQRresults_two_groups_75-25.csv', index_col=0).iloc[:,12:37]
    FairCPQR_results = pd.read_csv('FairCPQRresults_two_groups.csv', index_col=0).iloc[:,12:37]
    
    LowQR_results = pd.read_csv('LowQRresults_two_groups_75-25.csv', index_col=0).iloc[:,8:20]
    FairLowQR_results = pd.read_csv('FairLowQRresults_two_groups.csv', index_col=0).iloc[:,8:20]
       
    fig, axs = plt.subplots(1, 2)
    joint_histogram(RRQR_results.loc['err_noninverted'], RRQR_results.loc['err_inverted'], RRQR_results.columns.astype(int), labels=['original','rotated'], ax=axs[0], y_top=1.5)
    joint_histogram(FairCPQR_results.loc['err_noninverted_test'], FairCPQR_results.loc['err_inverted_test'], FairCPQR_results.columns.astype(int), labels=['original','rotated'], ax=axs[1], y_top=1.5)
    
    # ~ joint_histogram(
        # ~ quantities=[LowQR_results.loc['err_noninverted'], LowQR_results.loc['err_inverted'], FairLowQR_results.loc['err_noninverted_test'], FairLowQR_results.loc['err_inverted_test']], 
        # ~ k=LowQR_results.columns.astype(int), 
        # ~ labels=['Low QR original','Low QR rotated', 'Fair Low QR original','Fair Low QR rotated'], 
        # ~ ax=axs[0]
    # ~ )
    # ~ joint_histogram(
        # ~ quantities=[LowQR_results.loc['acc_noninverted'], LowQR_results.loc['acc_inverted'], FairLowQR_results.loc['acc_noninverted'], FairLowQR_results.loc['acc_inverted']], 
        # ~ k=LowQR_results.columns.astype(int), 
        # ~ labels=['Low QR original','Low QR rotated', 'Fair Low QR original','Fair Low QR rotated'], 
        # ~ ax=axs[1],
        # ~ y_bottom=0.5
    # ~ )
    
    axs[0].set_title("Reconstruction errors")
    axs[1].set_title("Fair CPQR")
    
    axs[0].set_xlabel("$k$")
    axs[1].set_xlabel("$k$")
    
    axs[0].set_ylabel("Relative group-wise reconstruction error")
    axs[1].set_ylabel("Relative group-wise reconstruction error")
    
    # ~ axs[0].set_ylabel("Relative group-wise reconstruction error")
    # ~ axs[1].set_ylabel("Classifier accuracy")
    
    plt.style.use('seaborn-v0_8-paper')
    plt.tight_layout()
    fig.set_figwidth(16)
    # ~ fig.savefig("FairCPQR-reconstruction-unfairness.png")
    plt.show()
