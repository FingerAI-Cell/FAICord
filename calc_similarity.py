from src import AudioFileProcessor
from src import WeSPEAKEMB, SBEMB
from src import PostProcessPipe
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import os

def main(args):
    with open(os.path.join(args.model_config_path, args.model_config_file)) as f:
        model_config = json.load(f) 

    speaker_emb = SBEMB(model_config)
    speaker_emb.set_classifier()
    speaker_emb.set_srmodel()
    
    wsemb = WeSPEAKEMB()
    ws_model = wsemb.load_model()

    audio_file_p = AudioFileProcessor()
    postprocess_pipe = PostProcessPipe()

    chunk1 = audio_file_p.chunk_audio(args.file_name, time_s=4.013, time_e=10.868)
    chunk2 = audio_file_p.chunk_audio(args.file_name, time_s=115.6, time_e=120)

    audio_emb = speaker_emb.get_emb(speaker_emb.classifier, chunk1[0])
    audio_emb2 = speaker_emb.get_emb(speaker_emb.classifier, chunk2[0])
    print(f"Speechbrain emb shape: {np.shape(audio_emb)}")
    
    buf1 = audio_file_p.audioseg_to_bytesio(chunk1[0])
    buf2 = audio_file_p.audioseg_to_bytesio(chunk2[0])
    ws_emb = wsemb.get_embedding(ws_model, buf1)
    ws_emb2 = wsemb.get_embedding(ws_model, buf2)
    print(f"Wespeaker emb shape: {np.shape(ws_emb)}")
    
    sb_sim = postprocess_pipe.calc_emb_similarity(audio_emb, audio_emb2)
    wespeak_sim = postprocess_pipe.calc_emb_similarity(ws_emb, ws_emb2, model_type='wespeaker')
    print(f"SpeechBrain emb similarity: {sb_sim}")
    print(f"Wespeaker emb similarity: {wespeak_sim}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser() 
    argparser.add_argument('--model_config_path', type=str, default='./models')
    argparser.add_argument('--model_config_file', type=str, default='speechbrain_config.json')
    argparser.add_argument('--file_name', type=str, required=True)
    cli_args = argparser.parse_args()
    main(cli_args)