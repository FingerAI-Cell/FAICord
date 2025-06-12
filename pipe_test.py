from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe
from src import EMBVisualizer
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import os
import time

def main(args):
    '''
    Default Setting
    '''
    start = time.time()
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe()
    diar_pipe = DIARPipe()
    postprocess_pipe = PostProcessPipe()
    
    frontend_pipe.set_env()
    vad_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_vad_config.yaml'))
    diar_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml'))
    postprocess_pipe.set_env()
    emb_visualizer = EMBVisualizer()
    '''
    Cleanse audio, Get VAD Result, Get Diar Result, Process Diar Result 
    '''
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=args.chunk_length, deverve=True)
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    diar_result, _ = diar_pipe.get_diar(args.file_name, return_embeddings=False)   # emb 값 사용 x 
    processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)    # ok. 
    relabeled_diar = postprocess_pipe.prepare_nonoverlapped_labels(args.file_name, non_overlapped_diar)
    final_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, relabeled_diar)
    speaker_emb = postprocess_pipe.get_chunk_emb_array(args.file_name, final_diar)
    speaker_emb2 = postprocess_pipe.get_chunk_emb_array(args.file_name, diar_result)
    chunk_idx, emb_array, labels, segments = speaker_emb[2]  # unpack your data
    chunk_idx2, emb_array2, labels2, segments2 = speaker_emb2[2]  # unpack your data

    emb_visualizer.pca_and_plot(
        embeddings=emb_array,
        labels=labels,
        title=f"Chunk {chunk_idx} Speaker Embeddings",
        file_path="."  # 원하는 폴더 경로
    )
    emb_visualizer.pca_and_plot(
        embeddings=emb_array2,
        labels=labels,
        title=f"Chunk origin {chunk_idx} Speaker Embeddings",
        file_path="."  # 원하는 폴더 경로
    )
    # print(speaker_emb)
    diar_pipe.save_files(final_diar, file_name=args.file_name)
    # file_name = args.file_name.split('/')[-1].split('.')[0] 
    # overlapped_file = args.file_name.replace(file_name, 'non_overlapped_bl_' + file_name)
    # diar_pipe.save_files(non_overlapped_diar, file_name=overlapped_file)

    # overlapped_file = args.file_name.replace(file_name, 'non_overlapped_al_' + file_name)
    # diar_pipe.save_files(relabeled_diar, file_name=overlapped_file)
    # diar_pipe.save_files(diar_result, file_name='full_diar_alg3_' + file_name)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)