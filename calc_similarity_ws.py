from src import WSEMB, SBEMB, EMBVisualizer
from src import KNNCluster
from dotenv import load_dotenv
import numpy as np
import argparse
import torch
import json
import os


def main(args):
    wsemb = WSEMB()
    ws_model = wsemb.load_model()
    knn_cluster = KNNCluster()
    emb_visualizer = EMBVisualizer()

    speaker_A = os.listdir('./dataset/audio/강성호팀장')
    speaker_B = os.listdir('./dataset/audio/외부업체A')
    speaker_C = os.listdir('./dataset/audio/김태완매니저')

    emb_a = wsemb.get_embeddings(ws_model, os.path.join(args.file_path, '강성호팀장') , speaker_A)
    emb_b = wsemb.get_embeddings(ws_model, os.path.join(args.file_path, '외부업체A'), speaker_B)
    emb_c = wsemb.get_embeddings(ws_model, os.path.join(args.file_path, '김태완매니저'), speaker_C)
    print(np.shape(emb_a))

    emb_array_a = np.vstack(emb_a)
    emb_array_b = np.vstack(emb_b)
    emb_array_c = np.vstack(emb_c)
    print(np.shape(emb_array_a))

    emb_array = np.vstack([emb_array_a, emb_array_b, emb_array_c])
    print("stacked ws_emb_array shape:", emb_array.shape)    # (2, 192)
    
    labels = ['A'] * emb_array_a.shape[0] + ['B'] * emb_array_b.shape[0] + ['C'] * emb_array_c.shape[0]
    emb_visualizer.pca_and_plot(emb_array, labels=labels, title="Wespeaker Embeddings (PCA 2D)")
    emb_visualizer.tsne_and_plot(emb_array, labels=labels, title="Wespeaker Embeddings (t-SNE)")
    
    new_labels = knn_cluster.relabel_by_knn(emb_array, labels)
    emb_visualizer.pca_and_plot(emb_array, labels=new_labels, title="Wespeaker new Embeddings (PCA 2D)")
    emb_visualizer.tsne_and_plot(emb_array, labels=new_labels, title="Wespeaker new Embeddings (t-SNE)")
    '''
    speaker_means = wsemb.calc_speaker_mean_embeddings(emb_array, np.array(labels))
    sim_matrix, speakers = wsemb.calc_mean_similarity_matrix(speaker_means)
    emb_visualizer.plot_similarity_heatmap(sim_matrix, speakers=speakers, title='Wespeaker Mean Similarity Heatmap')
    '''

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser() 
    cli_parser.add_argument('--file_path', type=str, default='./dataset/audio')
    cli_args = cli_parser.parse_args()
    main(cli_args)