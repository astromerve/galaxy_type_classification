import numpy as np
import itertools
from matplotlib import pyplot as plt

from astropy.io import fits

def generate_features_targets(data_mining):
    # Open the FITS file
    fits_file = fits.open(r"galaxies.fits")

    # Read the data from the FITS file
    data = fits_file[1].data

    # Extract the required columns
    output_targets = np.empty(shape=(len(data)), dtype='<U20')
    output_targets[:] = data['class']

    input_features = np.empty(shape=(len(data), 12))
    input_features[:, 0] = data['u_g']
    input_features[:, 1] = data['g_r']
    input_features[:, 2] = data['r_i']
    input_features[:, 3] = data['i_z']
    input_features[:, 4] = data['mCr4_u']
    input_features[:, 5] = data['mCr4_g']
    input_features[:, 6] = data['mCr4_r']
    input_features[:, 7] = data['mCr4_i']
    input_features[:, 8] = data['mCr4_z']
    input_features[:, 9] = data['petroR50_u'] / data['petroR90_u']
    input_features[:, 10] = data['petroR50_r'] / data['petroR90_r']
    input_features[:, 11] = data['petroR50_z'] / data['petroR90_z']

    # Close the FITS file
    fits_file.close()

    return input_features, output_targets


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')