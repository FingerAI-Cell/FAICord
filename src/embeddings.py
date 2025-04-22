from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
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

    def prepare_embeddings(self, embeddings):
        """
        speechbrain_embs: list of torch.Tensor, shape (1, 1, 192)
        wespeaker_embs: list of torch.Tensor, shape (256,)
        """
        return torch.stack([emb.view(-1) for emb in embeddings]).cpu().numpy() 
    

class WSEMB(BaseEMB):
    def __init__(self):
        super().__init__()
    
    def load_model(self, language='english'):
        model = wespeaker.load_model(language)
        return model

    def get_embedding(self, model, file_name):
        embedding = model.extract_embedding(file_name)
        return embedding 

    def get_embeddings(self, model, file_path, file_list):
        '''
        여러 파일들을 입력으로 받아 각 파일별 임베딩 리스트 반환 
        '''
        emb_list = []
        for file_name in file_list: 
            audio_file = os.path.join(file_path, file_name)
            if isinstance(audio_file, AudioSegment):
                buffer = BytesIO()
                audio_file.export(buffer, format='wav')
                buffer.seek(0)
                emb = model.extract_embedding(buffer)
            else:
                emb = model.extract_embedding(audio_file)
            emb_list.append(self.prepare_embeddings(emb).squeeze())
        return emb_list

    def get_emb_mean(self, embeddings):
        '''
        embeddings x N  -> np.mean(embeddings)
        화자별 대표 임베딩 구하는데 사용하는 함수 
        '''
        pass 
    

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
                                            
    def get_embedding(self, classifier, file_name):
        if isinstance(file_name, AudioSegment):
            buffer = BytesIO()
            file_name.export(buffer, format="wav")
            buffer.seek(0)    # 반드시 처음으로 포인터 옮기기
            signal, fs = torchaudio.load(buffer)
        else:
            signal, fs = torchaudio.load(file_name)
        embeddings = classifier.encode_batch(signal)
        return embeddings
    
    def get_embeddings(self, classifier, file_path, file_list):
        emb_list = []
        for file_name in file_list: 
            audio_file = os.path.join(file_path, file_name)
            if isinstance(audio_file, AudioSegment):
                buffer = BytesIO()
                audio_file.export(buffer, format='wav')
                buffer.seek(0)
                signal, fs = torchaudio.load(buffer) 
            else:
                signal, fs = torchaudio.load(audio_file)
            emb = classifier.encode_batch(signal)
            emb_list.append(self.prepare_embeddings(emb))
        return emb_list

    def get_emb_mean(self, embeddings):
        '''
        embeddings x N  -> np.mean(embeddings)
        화자별 대표 임베딩 구하는데 사용하는 함수 
        '''
        pass 

    def calc_emb_similarity(self, srmodel, audio_file1, audio_file2):
        score, pred = srmodel.verify_files(audio_file1, audio_file2)
        return score, pred 


class EMBVisualizer(BaseEMB):
    def __init__(self):
        super().__init__()

    def tsne_and_plot(self, embeddings, labels, title):
        """
        embeddings: numpy array [N, D]
        labels: list of str (len=N)
        title: plot title
        """
        tsne = TSNE(n_components=2, metric='cosine', perplexity=5, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8,6))
        unique_labels = list(set(labels))
        colors = plt.colormaps.get_cmap('tab10')
        for idx, label in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, color=colors(idx))

        plt.title(title)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        print(f"Saved plot to {title.replace(' ', '_')}.png")

    def pca_and_plot(self, embeddings, labels, title):
        """
        embeddings: numpy array [N, D]
        labels: list of str (len=N)
        title: plot title
        """
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(8,6))
        unique_labels = list(set(labels))
        colors = plt.colormaps.get_cmap('tab10')
        for idx, label in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, color=colors(idx))

        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        print(f"Saved plot to {title.replace(' ', '_')}.png")