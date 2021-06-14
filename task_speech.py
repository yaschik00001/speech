import glob
import os
import numpy as np
import scipy
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import argparse


SPEAKERS = ['S0767', 'S0901', 'S0903', 'S0905', 'S0916']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data', 
        type=str, 
        default='.\\TestTaskData\\data', 
        help='Path to data to perform evaluation on.'
    )
    parser.add_argument(
        '--path_to_norm_set', 
        type=str,
        default='.\\TestTaskData\\normalization_set.',
        help='Path to normalization set')
    parser.add_argument(
        '--path_to_embeddings', 
        type=str,
        default='.\\embeddings.npy',
        help='Path to embeddings of the evaluation data set.')
    parser.add_argument(
        '--path_to_norm_set_embeddings', 
        type=str,
        default='.\\norm_set_embeddings.npy',
        help='Path to normalization set embeddings.')
    parser.add_argument(
        '--path_to_onehot_classes', 
        type=str,
        default='.\\onehot_classes.npy',
        help='Path to onehot classes of the evaluation set.')
    args = parser.parse_args()

    # initialize classifier
    classifier = None

    # get evaluation set embeddings
    print("Extracting evaluation set embeddings...")
    if os.path.exists(args.path_to_embeddings):

        embeddings = np.load(args.path_to_embeddings, allow_pickle=True)
        onehot_classes = np.load(args.path_to_onehot_classes, allow_pickle=True)

    else:

        # get evaluation set utterance paths
        utterances = glob.glob(f'{args.path_to_data}\\*\\*')

        # get Speechbrain classifier 
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

        # compute evaluation set embeddings with Speechbrain model and normalize them
        onehot_classes = None
        onehots = np.identity(len(SPEAKERS))
        for utterance in tqdm(utterances):
            signal, fs =torchaudio.load(utterance)
            embedding = classifier.encode_batch(signal)
            utterance_speaker = utterance.split('\\')[-2]
            _class = SPEAKERS.index(utterance_speaker)
            if onehot_classes is None:
                onehot_classes = np.expand_dims(onehots[_class], 0)
                embeddings = embedding[0]/np.linalg.norm(embedding[0])
            else:
                onehot_classes = np.concatenate([onehot_classes, np.expand_dims(onehots[_class], 0)], axis=0)
                embeddings = np.concatenate([embeddings, embedding[0]/np.linalg.norm(embedding[0])], axis=0)

        # save embeddings and one-hot classes as npy
        np.save(args.path_to_embeddings, embeddings)
        np.save(args.path_to_onehot_classes, onehot_classes)

    # compute labels and scores between all pairs of evaluation set embeddings
    label_matrix = onehot_classes@np.transpose(onehot_classes)
    score_matrix = embeddings@np.transpose(embeddings)
    indices = np.triu_indices(label_matrix.shape[0], 1)
    labels = label_matrix[indices]
    scores = score_matrix[indices]

    # compute EER
    fpr, tpr, thrs = roc_curve(labels, scores)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print(f'EER for evaluation set without normalization: {eer}')

    # get normalization set embeddings
    print("Extracting normalization set embeddings...")
    if os.path.exists(args.path_to_norm_set_embeddings):

        embeddings_norm_set = np.load(args.path_to_norm_set_embeddings, allow_pickle=True)

    else:
    
        # get normalization set utterance paths
        utterances_norm_set = glob.glob(f'{args.path_to_norm_set}\\*')

        # get Speechbrain classifier 
        if classifier is None:
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

        # compute normalization set embeddings with Speechbrain model and normalize them
        embeddings_norm_set = None
        for utterance in tqdm(utterances_norm_set):
            signal, fs = torchaudio.load(utterance)
            embedding = classifier.encode_batch(signal)
            utterance_speaker = utterance.split('\\')[-2]
            if embeddings_norm_set is None:
                embeddings_norm_set = embedding[0]/np.linalg.norm(embedding[0])
            else:
                embeddings_norm_set = np.concatenate([embeddings_norm_set, embedding[0]/np.linalg.norm(embedding[0])], axis=0)

        # save normalization set embeddings as npy
        np.save(args.path_to_norm_set_embeddings, embeddings_norm_set)

    # compute EER with normalization for different normalization set sizes
    eers_norm = []
    lens_norm_set = []
    len_norm_set = embeddings_norm_set.shape[0]
    while len_norm_set != 1:
        print(f'Doing normalization set size {len_norm_set}')
        # compute cosine similarity between data set embeddings and normalization set embeddings
        embeddings_norm_set = embeddings_norm_set[:len_norm_set]
        score_matrix_for_norm = embeddings@np.transpose(embeddings_norm_set)

        # compute average and std over the normalization set axis
        means_for_norm = np.mean(score_matrix_for_norm, axis=1)
        stds_for_norm = np.std(score_matrix_for_norm, axis=1)

        # compute s-norm scores
        means_x = means_for_norm[indices[0]]
        means_y = means_for_norm[indices[1]]
        stds_x = stds_for_norm[indices[0]]
        stds_y = stds_for_norm[indices[1]]
        scores_norm = 1/2*((scores - means_x)/stds_x + (scores - means_y)/stds_y)

        # compute EER
        fpr_norm, tpr_norm, thrs_norm = roc_curve(labels, scores_norm)
        eer_norm = brentq(lambda x : 1. - x - interp1d(fpr_norm, tpr_norm)(x), 0., 1.)
        eers_norm.append(eer_norm)
        lens_norm_set.append(len_norm_set)
        len_norm_set //= 2

    print(f"EER's: {eers_norm}")
    print(f"Norm set sizes: {lens_norm_set}")


    plt.plot(lens_norm_set, eers_norm, marker='o', color='purple')
    plt.plot([0, 160], [eer, eer], '--', color='red')
    plt.xlabel('Length of norm set')
    plt.ylabel('EER')
    plt.legend(['with norm set', 'without norm set'])
    plt.grid()
    plt.savefig(f'eer_on_norm_set_length.png')
