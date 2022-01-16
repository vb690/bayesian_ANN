import numpy as np

from scipy.stats import sem
from scipy.stats import median_abs_deviation as mad

from sklearn.calibration import calibration_curve

import umap
import umap.aligned_umap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

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
    axs[2].set_title('Calibration Curve')

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


def visualize_embedding(embedding, y, title, sampled_emb=25, **kwargs):
    """
    """
    if isinstance(sampled_emb, int):
        sampled_emb = np.random.choice(
                    [i for i in range(embedding.shape[0])],
                    sampled_emb
        )

    rows = int(len(sampled_emb) ** 0.5)
    fig, axs = plt.subplots(rows, rows, figsize=(15, 15))

    embeddings = [embedding[index, :, :] for index in sampled_emb]
    relationships = [
        {i: i for i in range(embedding.shape[1])}
        for relationship in range(len(embeddings) - 1)
    ]
    reductions = umap.AlignedUMAP(**kwargs).fit_transform(
        embeddings,
        relations=relationships,
        n_neighbours=embeddings[0].shape[1] // 4
    )
    for red, ax, sample in zip(reductions, axs.flatten(), sampled_emb):

        for y_unique in np.unique(y):

            y_idx = np.argwhere(y == y_unique).flatten()
            cmap = matplotlib.cm.get_cmap('tab10')

            ax.scatter(
                red[y_idx, 0],
                red[y_idx, 1],
                s=5,
                color=cmap(y_unique),
                label=f'Digit {y_unique}'
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'Embedding Sample {sample}')

    plt.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    axs.flatten()[-3].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=5
    )
    plt.show()
    return None


def visualize_kde_embedding(embedding, y, title, sampled_emb=25, **kwargs):
    """
    """
    if isinstance(sampled_emb, int):
        sampled_emb = np.random.choice(
                    [i for i in range(embedding.shape[0])],
                    sampled_emb
        )
    cmap = matplotlib.cm.get_cmap('tab10')
    fig = plt.figure(figsize=(10, 10))

    embeddings = [embedding[index, :, :] for index in sampled_emb]
    relationships = [
        {i: i for i in range(embedding.shape[1])}
        for relationship in range(len(embeddings) - 1)
    ]

    reductions = umap.AlignedUMAP(**kwargs).fit_transform(
        embeddings,
        relations=relationships,
        n_neighbours=embeddings[0].shape[1] // 4
    )
    for y_unique in np.unique(y):

        y_red = []

        for red, sample in zip(reductions, sampled_emb):

            y_idx = np.argwhere(y == y_unique).flatten()
            y_red.append(red[y_idx, :])

        y_red = np.vstack(y_red)
        sns.kdeplot(
            x=y_red[:, 0],
            y=y_red[:, 1],
            fill=True,
            color=cmap(y_unique),
            alpha=0.5,
            label=f'Digit {y_unique}'
        )

    custom_patches = [
        Circle([0], [0], color=cmap(digit)) for digit in range(10)
    ]

    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    plt.legend(
        custom_patches,
        [f'Digit {digit}' for digit in range(10)]
    )

    plt.show()
    return None


def visualize_residuals(y_trace, y_gt, figsize=(10, 10)):
    """
    """
    md_mae = round(np.median(abs(y_trace - y_gt)), 3)
    mad_mae = round(mad(abs(y_trace - y_gt), axis=None), 3)

    X = {
        'ŷ': np.median(y_trace, axis=0),
        'y': y_gt
    }

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharey=True)

    for ax_i, (label, x) in enumerate(X.items()):

        axs[ax_i].scatter(
            x,
            np.median(y_trace - y_gt, axis=0),
            s=10,
            c='k'
        )
        axs[ax_i].errorbar(
            x,
            np.median(y_trace - y_gt, axis=0),
            yerr=mad(y_trace - y_gt) * 1.96,
            c='k',
            linewidth=0.25,
            ls='none'
        )
        axs[ax_i].errorbar(
            x,
            np.median(y_trace - y_gt, axis=0),
            yerr=mad(y_trace - y_gt) * 1.645,
            c='k',
            linewidth=0.5,
            ls='none'
        )
        axs[ax_i].axhline(
            0,
            linestyle=':',
            c='r'
        )
        axs[ax_i].set_xlabel(
            label
        )

        axs[ax_i].set_ylabel(
            'Residuals'
        )

    plt.suptitle(
        f'MAE: {md_mae} +/- {1.96 * mad_mae}'
    )
    plt.tight_layout()
    plt.show()
    return None
