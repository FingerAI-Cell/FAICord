from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe
import numpy as np
import argparse 
import os 

def main(args):
    frontend_pipe = FrontendPipe()
    frontend_pipe.set_env()
    diar_pipe = DIARPipe()
    diar_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml'))

    emb = diar_pipe.diar_model.load_emb_npy('./dataset/emb/chunk_20250211_1.npy')
    emb2 = diar_pipe.diar_model.load_emb_npy('./dataset/emb/chunk_20250211_0.npy')
    score = diar_pipe.calc_emb_similarity(emb[0], emb[1])
    score2 = diar_pipe.calc_emb_similarity(emb[0], emb2[0])
    score3 = diar_pipe.calc_emb_similarity(emb[0], emb2[1])
    score4 = diar_pipe.calc_emb_similarity(emb[0], emb2[2])
    score5 = diar_pipe.calc_emb_similarity(emb[0], emb2[3])

    score12 = diar_pipe.calc_emb_similarity(emb[1], emb2[0])
    score13 = diar_pipe.calc_emb_similarity(emb[1], emb2[1])
    score14 = diar_pipe.calc_emb_similarity(emb[1], emb2[2])
    score15 = diar_pipe.calc_emb_similarity(emb[1], emb2[3])

    score6 = diar_pipe.calc_emb_similarity(emb[0], emb[0])
    print(score, score2, score3, score4, score5, score6)
    print(score, score12, score13, score14, score15)
    # frontend_processor.save_audio(clean_audio, './dataset/denoised/denoised-pipe-test.wav')   


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_args = cli_parser.parse_args()
    main(cli_args)