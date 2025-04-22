from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import torchaudio
import wespeaker
import random
import torch
import os 

class BaseEMB:
    def __init__(self):
        self.set_seed()
        self.set_gpu()

    def set_seed(self, seed=42):
        """랜덤 시드 설정"""
        self.seed = seed
        random.seed(self.seed)  
        np.random.seed(self.seed)  
        torch.manual_seed(self.seed)  
        torch.cuda.manual_seed_all(self.seed)    # GPU 연산을 위한 시드 설정
        torch.backends.cudnn.deterministic = True   # 연산 재현성을 보장
        torch.backends.cudnn.benchmark = False    # 성능 최적화 옵션 비활성화

    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"


class WSEMB(BaseEMB):
    def __init__(self):
        super().__init__()
    
    def load_model(self, language='english'):
        model = wespeaker.load_model(language)
        return model

    def get_embedding(self, model, file_name):
        embedding = model.extract_embedding(file_name)
        return embedding 


class SBEMB(BaseEMB):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def set_classifier(self):
        self.classifier = EncoderClassifier.from_hparams(
            source=self.config['model_name'],
            run_opts={"device":self.device}
        )

    def set_srmodel(self):
        self.srmodel = SpeakerRecognition.from_hparams(
            source=self.config['model_name'], 
            savedir=self.config['model_path'], 
            run_opts={"device":self.device}
        )
                                            
    def get_emb(self, classifier, audio_file):
        if isinstance(audio_file, AudioSegment):
            buffer = BytesIO()
            audio_file.export(buffer, format="wav")
            buffer.seek(0)  # 반드시 처음으로 포인터 옮기기
            signal, fs = torchaudio.load(buffer)
        signal, fs = torchaudio.load(audio_file)
        embeddings = classifier.encode_batch(signal)
        return embeddings
    
    def calc_emb_similarity(self, srmodel, audio_file1, audio_file2):
        score, pred = srmodel.verify_files(audio_file1, audio_file2)
        return score, pred 