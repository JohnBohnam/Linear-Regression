import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_hist, val_loss_hist):
    eps = 1e-100
    loss_hist = [max(eps, x) for x in loss_hist]
    val_loss_hist = [max(eps, x) for x in val_loss_hist]
    plt.plot(np.log(loss_hist), label='train')
    plt.plot(np.log(val_loss_hist), label='validation')
    plt.legend()


def plot_corelations(train_data):
    correlations = []
    for feature in train_data.columns:
        if feature == 'y':
            continue
        corr = np.corrcoef(train_data[feature], train_data['y'])[0, 1]
        correlations.append((feature, corr))

    correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    # plot as heatmap

    correlations = np.array(correlations)
    correlations_values = correlations[:, 1].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.bar(correlations[:, 0], correlations_values, color='b')

    ax.set_xlabel('Features')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation of Features with Target')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_heatmap(data_3d):
    # data_3d is an array of tuples (x, y, z)
    data_3d = np.array(data_3d)
    x = data_3d[:, 0].astype(float)
    y = data_3d[:, 1].astype(float)
    z = data_3d[:, 2].astype(float)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# get the most important features used by the model
def plot_feature_importance(model, train_data):
    features = train_data.drop(columns='y').columns
    weights = model.weights
    features_weights = list(zip(features, weights))
    features_weights = sorted(features_weights, key=lambda x: abs(x[1]), reverse=True)
    print(features_weights[:10])

    # plot this
    features = [x[0] for x in features_weights]
    weights = [x[1] for x in features_weights]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.bar(features, weights, color='b')

    ax.set_xlabel('Features')
    ax.set_ylabel('Weights')
    ax.set_title('Weights of Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return features_weights


def plot_losses(train_losses, val_losses, alphas):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original losses
    axs[0].plot(alphas, train_losses, label='train')
    axs[0].plot(alphas, val_losses, label='validation')
    axs[0].set_title('Original Losses')
    axs[0].set_xlabel('Alphas')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot normalized losses
    train_losses -= np.mean(train_losses)
    train_losses /= np.std(train_losses)
    val_losses -= np.mean(val_losses)
    val_losses /= np.std(val_losses)

    axs[1].plot(alphas, train_losses, label='train')
    axs[1].plot(alphas, val_losses, label='validation')
    axs[1].set_title('Normalized Losses')
    axs[1].set_xlabel('Alphas')
    axs[1].set_ylabel('Normalized Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
