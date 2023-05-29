import pyedflib
import numpy as np


def read_edf_file(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs, signal_labels


if __name__ == '__main__':
    sigbufs, signal_labels = read_edf_file(
        'datasets/EDF-ND/29-余振琮-ND-EEG-2.edf')
    # Draw the first 20 seconds of the first channel:
    import matplotlib.pyplot as plt
    plt.plot(sigbufs[0, 0:2000])
    plt.show()
    # plt.plot(signal_labels[0, 0:2000])
    # plt.show()
    print(signal_labels)

    print(sigbufs.shape)

    # # Load the model
    # from models.CLAM import RNNAttentionModel
    # import torch

    # model = RNNAttentionModel(input_size=16, hid_size=128, rnn_type='lstm',
    #                           bidirectional=False, n_classes=2, kernel_size=5)

    # # Test the model
    # x = torch.randn(64, 16, 512)
    # y = model(x)
    # print(y.shape)
