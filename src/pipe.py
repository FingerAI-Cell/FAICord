from .audio_handler import NoiseHandler, VoiceEnhancer, AudioVisualizer
from .preprocessors import AudioFileProcessor
from .pyannotes import PyannotDIAR, PyannotVAD
from .embeddings import SBEMB, WSEMB, EMBVisualizer
from .clusters import KNNCluster
from intervaltree import Interval, IntervalTree
from scipy.spatial.distance import cosine
from abc import abstractmethod
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import tempfile


class BasePipeline:
    @abstractmethod
    def set_env(self):
        pass


class FrontendPipe(BasePipeline):
    def set_env(self):
        self.noise_handler = NoiseHandler()
        self.voice_enhancer = VoiceEnhancer()
        self.audio_file_processor = AudioFileProcessor()

    def process_audio(self, audio_file, fade_ms=50, chunk_length=300, deverve=False):
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=chunk_length)
        print(f"[DEBUG] Chunk count: {len(chunks)}, chunk_length={chunk_length} sec")
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_io = BytesIO()
            chunk.export(chunk_io, format='wav')
            chunk_io.seek(0)
            denoised = self.noise_handler.denoise_audio(chunk_io)
            print(f"[DEBUG] Denoised chunk duration: {len(chunk) / 1000} sec")
            if deverve == True:             
                clean_chunk = self.noise_handler.deverve_audio(denoised)
                clean_chunk.seek(0)
                seg = self.audio_file_processor.audiofile_to_AudioSeg(clean_chunk)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            else:
                seg = self.audio_file_processor.audiofile_to_AudioSeg(denoised)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            processed_chunks.append(seg)
        clean_audio = self.audio_file_processor.concat_chunk(processed_chunks)
        # normalized_audio = self.voice_enhancer.normalize_audio_lufs(clean_audio)
        return clean_audio

    def save_audio(self, audio_file, file_name=None):
        self.audio_file_processor.save_audio(audio_file, file_name=file_name)


class VADPipe(BasePipeline):
    def set_env(self, vad_config):
        self.vad_model = PyannotVAD()
        self.vad_config = vad_config
        self.audio_visualizer = AudioVisualizer()

    def get_vad_timestamp(self, audio_file):
        vad_pipeline = self.vad_model.load_pipeline_from_pretrained(self.vad_config)
        vad_timestamp = self.vad_model.get_vad_timestamp(vad_pipeline, audio_file)
        return vad_timestamp


class DIARPipe(BasePipeline):
    '''
    resegment: DIAR <-> VAD mapping -> calc Non-Overlapped timeline 
    '''
    def set_env(self, diar_config, chunk_offset=300):
        self.diar_model = PyannotDIAR()
        self.diar_config = diar_config 
        self.audio_visualizer = AudioVisualizer()
        self.audio_file_processor = AudioFileProcessor()
        self.chunk_offset=chunk_offset
        
    def get_diar(self, audio_file, num_speakers=None, return_embeddings=False):
        '''audio_file: AudioSeg'''
        diar_pipe = self.diar_model.load_pipeline_from_pretrained(self.diar_config)
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=self.chunk_offset)
        results = []; emb_results = []
        for idx, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                diar_result, emb = self.diar_model.get_diar_result(diar_pipe, temp_audio.name, num_speakers=num_speakers, return_embeddings=return_embeddings)
            results.append(diar_result)
            emb_results.append(emb)
        return results, emb_results

    def apply_vad(self, vad_result, diar_result):
        '''
        Re-segment diar_result using vad_result
        - diar_result: [( (start_time, end_time), speaker ), ... ]
        - vad_result: [(start_time, end_time), ...]

        1. VAD에 걸친 diar segment만 resegmented_diar에 추가
        2. VAD에 걸리지 않은 diar segment는 non_overlapped_segments에 추가
        '''
        non_overlapped_segments = []
        vad_diar = []
        vad_tree = IntervalTree(Interval(time_s, time_e) for time_s, time_e in vad_result)
        seen_segments = set()    # 중복 방지용
        for (time_s, time_e), speaker in diar_result:
            intersections = vad_tree.overlap(time_s, time_e)           
            segment_key = (round(time_s, 3), round(time_e, 3), speaker)
            if not intersections:
                non_overlapped_segments.append(((time_s, time_e), speaker))
            else:
                if segment_key not in seen_segments:
                    vad_diar.append(((time_s, time_e), speaker))
                    seen_segments.add(segment_key)
        return vad_diar

    def preprocess_result(self, diar_result, vad_result=None, emb_result=None):
        '''
        resegment, speaker_mapping 
        '''
        total_diar = self.diar_model.concat_diar_result(diar_result, chunk_offset=self.chunk_offset)
        if vad_result != None: 
            vad_diar = self.apply_vad(vad_result=vad_result, diar_result=total_diar)
            vad_diar = self.diar_model.split_diar_result(vad_diar, chunk_offset=self.chunk_offset)
            filtered_diar = self.diar_model.filter_filler(vad_diar)
            non_overlapped_diar = [self.diar_model.remove_overlap(diar_result) for diar_result in filtered_diar]
            return filtered_diar, non_overlapped_diar

    def save_files(self, diar_result, file_name, emb_result=None):
        '''
        save diar result, numpy emb as rttm, npy format for each chunk
        '''
        print(f'1: {file_name}')
        save_file_name = file_name.split('/')[-1].split('.')[0]
        print(f'2: {save_file_name}')
        for idx, chunk_diar in enumerate(diar_result):
            if len(diar_result) > 1: 
                save_file_name = f"chunk_{idx}_{file_name.split('/')[-1].split('.')[0]}"
            save_rttm_path = './dataset/rttm/' + save_file_name + '.rttm'
            save_emb_path = './dataset/emb/' + save_file_name + '.npy'
            self.diar_model.save_as_rttm(chunk_diar, output_rttm_path=save_rttm_path, file_name=save_file_name)
            # self.diar_model.save_as_emb(emb_result[idx], output_emb_path=save_emb_path)


class PostProcessPipe(BasePipeline):
    '''
    filler 처리된 diar result와, non-overlapped diar 후처리하는 클래스. 
    1. non-overlapped diar에서 화자별 임베딩 계산 
    2. 청크 내 화자 재레이블링 (KNN)
    3. 청크별 화자별 대표 임베딩 계산 
    4. 청크 병합 (화자 정보 매핑)
    '''
    def set_env(self):
        self.audio_file_processor = AudioFileProcessor()
        self.wsemb = WSEMB()
        self.emb_model = self.wsemb.load_model()
        self.emb_visualizer = EMBVisualizer()
        self.knn_cluster = KNNCluster()

    def prepare_nonoverlapped_labels(self, file_name, diar_result, k=5, chunk_offset=300):
        '''
        input: non overlapped diar result 
        return: relabeled non overlapped diar result 
        '''
        relabeled_diar_result = []
        audio_file_name = file_name.split('/')[-1].split('.')[0]
        for idx, diar in enumerate(diar_result):
            emb_result = self.wsemb.get_embeddings_from_diar(
                self.emb_model, file_name, diar, chunk_offset=idx*chunk_offset
            )
            emb_segment_set = set((start, end) for ((start, end), _, _) in emb_result)
            embeddings = [emb for (_, _, emb) in emb_result]
            original_labels = [speaker for (_, speaker, _) in emb_result]
            emb_array = np.vstack(embeddings)
            
            new_labels = self.knn_cluster.relabel_by_knn(emb_array, original_labels, k=k)
            emb_idx = 0
            relabeled_diar = []
            for segment in diar:
                (start, end), original_label = segment
                if original_label == 'filler':
                    relabeled_diar.append(segment)
                    print(f"[FILLER] ({start:.2f}-{end:.2f}) filler --> 유지")
                elif (start, end) in emb_segment_set:    # 실제 embedding이 있는 경우만 new_label 적용
                    new_label = new_labels[emb_idx]
                    relabeled_diar.append(((start, end), new_label))
                    print(f"({start:.2f}-{end:.2f}) Original: {original_label} --> New: {new_label}")
                    emb_idx += 1
                else:    # 1초 미만으로 embedding이 없었던 경우
                    relabeled_diar.append(segment)
                    print(f"[SHORT] ({start:.2f}-{end:.2f}) {original_label} --> 유지")
            self.emb_visualizer.pca_and_plot(emb_array, labels=original_labels, file_path='./dataset/img/pca',
                                            title=f"Wespeaker Embeddings_{audio_file_name}_{idx} (PCA 2D)")
            self.emb_visualizer.tsne_and_plot(emb_array, labels=original_labels, file_path='./dataset/img/t-sne',
                                            title=f"Wespeaker Embeddings_{audio_file_name}_{idx} (tsne 2D)")
            self.emb_visualizer.pca_and_plot(emb_array, labels=new_labels, file_path='./dataset/img/pca',
                                            title=f"Wespeaker new Embeddings_{audio_file_name}_{idx} (PCA 2D)")
            self.emb_visualizer.tsne_and_plot(emb_array, labels=new_labels, file_path='./dataset/img/t-sne',
                                            title=f"Wespeaker new Embeddings_{audio_file_name}_{idx} (tsne 2D)")
            relabeled_diar_result.append(relabeled_diar)
        return relabeled_diar_result

    def apply_labels_to_full_diar(self, full_diar, relabeled_nonoverlap_diar):
        '''
        full_diar[0]: chunk 0 diar    - [((start, end), speaker), ((start, end), speaker), ... ]  
        full_diar[1]: chunk 1 diar    -                         '' 
        '''
        # Step 1: flatten relabeled_nonoverlap_diar
        # print(full_diar[:2], end='\n\n')
        # print(relabeled_nonoverlap_diar[:3])
        relabeled_full_diar = [] 
        for idx, chunk in enumerate(full_diar):
            chunk_diar = [] 
            for (full_start, full_end), full_label in full_diar[idx]:
                overlap_segments = []
                for (rel_start, rel_end), rel_label in relabeled_nonoverlap_diar[idx]:
                    overlap_start = max(full_start, rel_start)
                    overlap_end = min(full_end, rel_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    if overlap_duration > 0:
                        overlap_segments.append((overlap_duration, rel_label))
                if overlap_segments:
                    best_label = max(overlap_segments, key=lambda x: x[0])[1]
                else:
                    best_label = full_label  # fallback
                chunk_diar.append(((full_start, full_end), best_label))
            relabeled_full_diar.append(chunk_diar)
        return relabeled_full_diar

    def get_speaker_vectoremb(self, diar_result, file_name):
        '''
        조정된 non-overlapped diar result를 입력으로 받아 기존의 diar result 수정 및 최종 화자 구분 결과 획득  
        '''
        pass 

    def calc_emb_similarity(self, emb1, emb2, model_type='sb'):
        if model_type == 'sb':   # [1, 1, 192] 
            emb1 = emb1.view(-1).cpu().numpy()
            emb2 = emb2.view(-1).cpu().numpy()
            similarity = 1 - cosine(emb1, emb2)
            return similarity
        elif model_type == 'wespeaker':
            emb1 = emb1.view(-1).cpu().numpy()
            emb2 = emb2.view(-1).cpu().numpy()            
            return 1 - cosine(emb1, emb2)    # cosine()은 distance니까 1 - distance

     