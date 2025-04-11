from src import DataProcessor, AudioFileProcessor
import argparse
import os

def main(args):
    file_name = args.file_name
    audio_p = AudioFileProcessor()
    chunk_list = audio_p.chunk_audio(args.file_name, chunk_length=args.chunk_length)
    for idx, chunk in enumerate(chunk_list):
        chunk_file_name = f"chunk_{idx}_{args.file_name.split('/')[-1].split('.')[0]}.wav"
        audio_p.save_audio(chunk, chunk_file_name)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./dataset')
    cli_parser.add_argument('--output_path', type=str, default='./dataset/audio')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)