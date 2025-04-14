from abc import abstractmethod
from pydub import AudioSegment
from openai import OpenAI
import tempfile
import json 
import os
import io 

class STTModule:
    def __init__(self, openai_api=None):
        self.openai_api = openai_api

    @abstractmethod
    def set_client(self):
        pass


class WhisperSTT(STTModule):
    def __init__(self, openai_api, generation_config):
        self.load_word_dictionary(os.path.join('./config', 'word_dict.json')) 
        self.set_client(openai_api)
        self.generation_config = generation_config 

    def set_client(self, openai_api):
        self.openai_client = OpenAI(api_key=openai_api)
    
    def load_word_dictionary(self, word_dict_path):
        with open(word_dict_path, mode='r', encoding='utf-8') as file:
            self.word_dict = json.load(file)    # JSON 데이터를 한번만 로드

    def apply_word_dictionary(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = stt_text.replace(incorrect_word, correct_word)
        return stt_text

    def prepare_whisper_audio(self, audio_input, sample_rate=16000):
        if isinstance(audio_input, AudioSegment):
            audio = audio_input
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            audio = AudioSegment.from_file(audio_input, format="wav")
        elif isinstance(audio_input, str):     # 파일 경로
            audio = AudioSegment.from_file(audio_input)
        else:
            raise TypeError("지원되지 않는 오디오 타입입니다.")
        whisper_audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        return whisper_audio

    def transcribe_text_api(self, audio_file):
        '''
        transcription.segments: segment.start, segment.end, segment.text, segment.no_speech_prob, segment.seek, segment.temperature, segment.avg_logprob
        '''
        whisper_audio = self.prepare_whisper_audio(audio_file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            whisper_audio.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language='ko',
                    response_format="verbose_json",
                    # timestamp_granularities=["segment"],
                    # prompt="The sentence may be cut off. do not make up words to fill in the rest of the sentence." 
                )
            os.remove(temp_audio_file.name)
        segments = transcription.segments
        for segment in segments:
            segment.text = segment.text.strip()
            segment.text = self.apply_word_dictionary(segment.text, self.word_dict) + " "
        return segment