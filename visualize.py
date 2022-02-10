import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["legend.loc"] = 'lower right'

def train_plot_curves(iters, loss, path=''):
    plt.figure(figsize=[16,9])
    plt.plot(iters, loss)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Training Loss', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def eval_plot_curves_with_bar(loss_mean, loss_std, colors=['orange', 'royalblue'], 
    alpha_fill=0.3, ax=None, path=''):
    iters, loss_mean_emd, loss_mean_chamfer = loss_mean.T
    _, loss_std_emd, loss_std_chamfer = loss_std.T
    plt.figure(figsize=[16, 9])

    emd_min = loss_mean_emd - loss_std_emd
    emd_max = loss_mean_emd + loss_std_emd

    chamfer_min = loss_mean_chamfer - loss_std_chamfer
    chamfer_max = loss_mean_chamfer + loss_std_chamfer

    plt.plot(iters, loss_mean_emd, color=colors[0], linewidth=6, label='EMD')
    plt.fill_between(iters, emd_max, emd_min, color=colors[0], alpha=alpha_fill)

    plt.plot(iters, loss_mean_chamfer, color=colors[1], linewidth=6, label='Chamfer')
    plt.fill_between(iters, chamfer_max, chamfer_min, color=colors[1], alpha=alpha_fill)

    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Dyanmics Model Evaluation Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)


    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def eval_plot_curves(loss_list, path=''):
    iters, loss_emd, loss_chamfer, loss_uh = map(list, zip(*loss_list))
    plt.figure(figsize=[16, 9])
    plt.plot(iters, loss_emd, linewidth=6, label='EMD')
    plt.plot(iters, loss_chamfer, linewidth=6, label='Chamfer')
    plt.plot(iters, loss_uh, linewidth=6, color='r', label='Hausdorff')
    plt.xlabel('frames', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Test Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()