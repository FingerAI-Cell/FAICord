from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe
from dotenv import load_dotenv
import numpy as np
import argparse 
import json 
import os 

def main(args):
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe()
    diar_pipe = DIARPipe()
    
    frontend_pipe.set_env()
    vad_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_vad_config.yaml'))
    diar_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml'))
    
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=300, deverve=True)
    # frontend_pipe.save_audio(clean_audio, 'frontend-processed.wav')
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    diar_result, emb_result = diar_pipe.get_diar(args.file_name, chunk_length=300, return_embeddings=True)
    diar_result = diar_pipe.resegment_result(vad_result=vad_result, diar_result=diar_result, chunk_offset=300)
    diar_pipe.save_files(diar_result, emb_result, file_name=args.file_name)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)