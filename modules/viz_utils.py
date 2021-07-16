import numpy as np

from scipy.stats import sem

from sklearn.calibration import calibration_curve

from umap import UMAP

import matplotlib.pyplot as plt

from .metrics_losses import prob_log_loss


def visulize_bernoulli_post(traces, y):
    """
    """
    loss = prob_log_loss(y, traces['p'])

    loss_mean = loss.mean(0)
    loss_sem = sem(loss, axis=0)

    y_hat_mean = traces['p'].mean(0)
    y_hat_sem = sem(traces['p'], axis=0)

    frac_pos, mean_pred = calibration_curve(
        y,
        y_hat_mean,
        n_bins=5
    )

    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    axs[0].scatter(
        x=[i for i in range(len(y))],
        y=y_hat_mean[np.argsort(y_hat_mean)],
        s=5,
        c='k'
    )
    axs[0].errorbar(
        x=[i for i in range(len(y))],
        y=y_hat_mean[np.argsort(y_hat_mean)],
        yerr=1.96*y_hat_sem[np.argsort(y_hat_mean)],
        c='r',
        elinewidth=1,
        ls='none'
    )
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('Estimated p')
    axs[0].set_title('Estimated Posterior p')

    axs[1].scatter(
        x=[i for i in range(len(y))],
        y=loss_mean[np.argsort(y_hat_mean)],
        s=5,
        c='k'
    )
    axs[1].errorbar(
        x=[i for i in range(len(y))],
        y=loss_mean[np.argsort(y_hat_mean)],
        yerr=1.96*loss_sem[np.argsort(y_hat_mean)],
        c='r',
        elinewidth=1,
        ls='none'
    )
    axs[1].set_xlabel('Sample Number')
    axs[1].set_ylabel('$logLoss(y, p)$')
    axs[1].set_title('Posterior Predictive Check')

    axs[2].plot(
        mean_pred,
        frac_pos,
        marker='o',
        linewidth=1,
        c='r',
        markersize=1
    )
    axs[2].plot(
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0.25, 0.5, 0.75, 1],
        linewidth=1,
        linestyle='--',
        c='k'
    )
    axs[2].set_xlabel('Mean Estimated p')
    axs[2].set_ylabel('Fraction Positives')
    axs[2].set_title('Posterior Predictive Check')

    plt.tight_layout()
    plt.show()
    return None


def visulize_categorical_post(X, p, index, max_labels=10, figsize=(10, 4)):
    """
    """
    X = X[index, :].reshape((8, 8))

    mean_p = p[:, index, :].mean(0)
    sem_p = sem(p[:, index, :], axis=0)

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(ncols=10, nrows=4)

    img_ax = fig.add_subplot(spec[0:4, 0:4])
    img_ax.imshow(X, cmap='binary')
    img_ax.set_yticks([])
    img_ax.set_xticks([])
    img_ax.set_title('Ground Truth Image')

    count_ax = fig.add_subplot(spec[0:4:, 4:])
    count_ax.bar(
        [label for label in range(max_labels)],
        [mean_p[label] for label in range(max_labels)],
        color='k'
    )
    count_ax.errorbar(
        [label for label in range(max_labels)],
        [mean_p[label] for label in range(max_labels)],
        yerr=sem_p,
        fmt='none',
        c='r'
    )
    count_ax.set_xticks(
        [label for label in range(max_labels)],
    )
    count_ax.set_xticklabels(
        [label for label in range(max_labels)],
    )
    count_ax.set_ylabel('Estimated p')
    count_ax.set_xlabel('Categories')
    count_ax.set_title('Estimated Posterior p')

    plt.tight_layout()
    plt.show()
    return None


def visualize_embedding(embedding, y, **kwargs):
    """
    """
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    sampled_embedding = np.random.choice(
                [i for i in range(embedding.shape[0])],
                25
    )
    for index, ax in zip(sampled_embedding, axs.flatten()):

        reduction = UMAP(
            n_components=2,
            **kwargs
        ).fit_transform(embedding[index, :, :])

        ax.scatter(
            reduction[:, 0],
            reduction[:, 1],
            s=5,
            c=y
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'Embedding Sample {index}')

    plt.tight_layout()
    plt.show()
    return None
