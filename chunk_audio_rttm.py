from src import DataProcessor, AudioFileProcessor
from src import PyannotDIAR
import argparse
import os


def main(args):
    audio_file = os.path.join('./dataset/audio/', args.file_name.split('/')[-1].split('.')[0].split('_')[-1] + '.wav')
    audio_p = AudioFileProcessor()
    pyannot_diar = PyannotDIAR()
    rttm_file = pyannot_diar.read_rttm(args.file_name)
    
    chunk_idx = int(args.file_name.split('/')[-1].split('_')[1])
    for idx, rttm in enumerate(rttm_file): 
        if rttm['speaker_id'] == args.speaker_info and rttm['duration'] > 1.0:
            time_s, time_e = rttm['start_time'] + chunk_idx * 300, rttm['end_time'] + chunk_idx * 300
            chunk = audio_p.chunk_audio(audio_file, time_s=time_s, time_e=time_e)[0]
            file_name = os.path.join(args.output_path, f"{args.file_name.split('/')[-1].split('.')[0]}_chunk_{idx}.wav")  
            audio_p.save_audio(chunk, file_name) 


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./dataset')
    cli_parser.add_argument('--output_path', type=str, required=True)
    cli_parser.add_argument('--speaker_info', type=str, required=True)
    cli_parser.add_argument('--file_name', type=str, required=True)   
    cli_args = cli_parser.parse_args()
    main(cli_args)