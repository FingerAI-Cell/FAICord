from src import AudioFileProcessor
from src import WSEMB, SBEMB, EMBVisualizer
from src import PostProcessPipe
from dotenv import load_dotenv
import numpy as np
import argparse
import torch
import json
import os


def main(args):
    with open(os.path.join(args.model_config_path, args.model_config_file)) as f:
        model_config = json.load(f)

    speaker_emb = SBEMB(model_config)
    speaker_emb.set_classifier()
    speaker_emb.set_srmodel()
    
    emb_visualizer = EMBVisualizer()
    audio_file_p = AudioFileProcessor()
    postprocess_pipe = PostProcessPipe()

    speaker_A = os.listdir('./dataset/audio/강성호팀장')
    speaker_B = os.listdir('./dataset/audio/외부업체A')
    speaker_C = os.listdir('./dataset/audio/김태완매니저')

    emb_a = speaker_emb.get_embeddings(speaker_emb.classifier, os.path.join(args.file_path, '강성호팀장') , speaker_A)
    emb_b = speaker_emb.get_embeddings(speaker_emb.classifier, os.path.join(args.file_path, '외부업체A'), speaker_B)
    emb_c = speaker_emb.get_embeddings(speaker_emb.classifier, os.path.join(args.file_path, '김태완매니저'), speaker_C)

    emb_array_a = np.vstack(emb_a)
    emb_array_b = np.vstack(emb_b)
    emb_array_c = np.vstack(emb_c) 

    emb_array = np.vstack([emb_array_a, emb_array_b, emb_array_c])
    print("stacked sb_emb_array shape:", emb_array.shape)    # (2, 192)

    labels = ['A'] * emb_array_a.shape[0] + ['B'] * emb_array_b.shape[0] + ['C'] * emb_array_c.shape[0]
    emb_visualizer.pca_and_plot(emb_array, labels=labels, title="Speechbrain Embeddings (PCA 2D)")
    emb_visualizer.tsne_and_plot(emb_array, labels=labels, title="Speechbrain Embeddings (t-SNE)")

    speaker_means = speaker_emb.calc_speaker_mean_embeddings(emb_array, np.array(labels))
    sim_matrix, speakers = speaker_emb.calc_mean_similarity_matrix(speaker_means)
    emb_visualizer.plot_similarity_heatmap(sim_matrix, speakers=speakers, title='Speechbrain Mean Similarity Heatmap')


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser() 
    cli_parser.add_argument('--file_path', type=str, default='./dataset/audio')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--model_config_file', type=str, default='speechbrain_config.json')
    cli_args = cli_parser.parse_args()
    main(cli_args)