class WeSPEAK:
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

class WeSPEAKEMB(WeSPEAK):
    def __init__(self):
        super().__init__()
    
    def load_model(self, model_dir):
        model = wespeaker.load_model_local(model_dir)
        return model.set_gpu(0)

    def get_embedding(self, model, file_name):
        embedding = model.extract_embedding(file_name)
        return embedding 

    def calc_emb_similarity(self, emb1, emb2):
        from scipy.spatial.distance import cosine
        return 1 - cosine(emb1, emb2)    # cosine()은 distance니까 1 - distance