from intervaltree import Interval, IntervalTree
from pyannote.audio import Inference
from pyannote.audio import Pipeline
from pyannote.audio import Model 
from pyannote.audio import Audio
from pyannote.core import Segment
from pydub import AudioSegment
from pathlib import Path
from io import BytesIO
import numpy as np 
import torchaudio
import random
import torch
import os

class Pyannot:
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
    
    def load_pipeline_from_pretrained(self, path_to_config: str) -> Pipeline:
        '''
        the paths in the config are relative to the current working directory
        so we need to change the working directory to the model path
        and then change it back
        * first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
        '''
        path_to_config = Path(path_to_config)
        print(f"Loading pyannote pipeline from {path_to_config}...")
        cwd = Path.cwd().resolve()    # store current working directory
        cd_to = path_to_config.parent.parent.resolve()
        os.chdir(cd_to)

        pipeline = Pipeline.from_pretrained(path_to_config)
        os.chdir(cwd)
        return pipeline.to(self.device)


class PyannotVAD(Pyannot): 
    ''' voice activity detection  - pytorch.bin 모델 없음 ''' 
    def __init__(self):
        super().__init__()

    def get_vad_timestamp(self, pipeline, audio_file):
        if isinstance(audio_file, AudioSegment):
            buffer = BytesIO()
            audio_file.export(buffer, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)
        elif isinstance(audio_file, (str, bytes, os.PathLike)):
            waveform, sample_rate = torchaudio.load(audio_file)
        else:
            raise TypeError("지원되지 않는 오디오 형식입니다.")
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        vad_result = pipeline(audio_in_memory)
        vad_timestamp = []
        for speech in vad_result.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))
        return vad_timestamp


class PyannotDIAR(Pyannot):
    def __init__(self):
        super().__init__()
    
    def calc_emb_similarity(self, emb1, emb2):
        from scipy.spatial.distance import cosine
        return 1 - cosine(emb1, emb2)    # cosine()은 distance니까 1 - distance

    def get_diar_result(self, pipeline, audio_file, num_speakers=None, return_embeddings=False):
        diarization = pipeline(audio_file, num_speakers=num_speakers, return_embeddings=return_embeddings)
        diar_result = []
        embeddings = None
        seen_segments = set()

        if return_embeddings == False:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                duration = end_time - start_time
                if duration >= 0.3:
                    segment_key = (round(start_time, 3), round(end_time, 3), speaker)
                    if segment_key not in seen_segments:
                        diar_result.append([(start_time, end_time), speaker])
                        seen_segments.add(segment_key)
        else:
            embeddings = diarization[1]
            for segment, _, speaker in diarization[0].itertracks(yield_label=True):
                start_time, end_time = segment.start, segment.end
                duration = end_time - start_time
                if duration >= 0.3:
                    segment_key = (round(start_time, 3), round(end_time, 3), speaker)
                    if segment_key not in seen_segments:
                        diar_result.append([(start_time, end_time), speaker])
                        seen_segments.add(segment_key)
        return diar_result, embeddings

    def concat_diar_result(self, diar_result, chunk_offset=None):
        total_diar_result = []
        for idx, diar in enumerate(diar_result):
            offset_result = [((start_time + chunk_offset * idx, end_time + chunk_offset * idx), speaker) for (start_time, end_time), speaker in diar]
            total_diar_result.extend(offset_result)
        return total_diar_result

    def split_diar_result(self, diar_result, chunk_offset=300):
        """
        diar_result: [( (start, end), speaker ), ... ]
        chunk_offset: 청크 단위 (기본값 300초)
        """
        from collections import defaultdict
        
        splited_diar_result = defaultdict(list)      
        for (start_time, end_time), speaker in diar_result:
            start_chunk_idx = int(start_time // chunk_offset)
            end_chunk_idx = int(end_time // chunk_offset)
            for chunk_idx in range(start_chunk_idx, end_chunk_idx + 1):
                chunk_start = chunk_offset * chunk_idx
                chunk_end = chunk_start + chunk_offset
                
                seg_start = max(start_time, chunk_start)
                seg_end = min(end_time, chunk_end)
                if seg_end - seg_start > 0:
                    splited_diar_result[chunk_idx].append(((seg_start - chunk_start, seg_end - chunk_start), speaker))
        return [splited_diar_result[idx] for idx in sorted(splited_diar_result)]

    def resegment_result(self, vad_result, diar_result):
        '''
        Re-segment diar_result using vad_result
        - diar_result: [( (start_time, end_time), speaker ), ... ]
        - vad_result: [(start_time, end_time), ...]

        목적: 
        1. VAD에 걸친 diar segment만 resegmented_diar에 추가
        2. VAD에 걸리지 않은 diar segment는 non_overlapped_segments에 추가
        3. 추가할 때 중복 방지 (set 사용)
        '''
        non_overlapped_segments = []
        resegmented_diar = []
        vad_tree = IntervalTree(Interval(time_s, time_e) for time_s, time_e in vad_result)
        seen_segments = set()  # 중복 방지용

        for (time_s, time_e), speaker in diar_result:
            intersections = vad_tree.overlap(time_s, time_e)           
            segment_key = (round(time_s, 3), round(time_e, 3), speaker)

            if not intersections:
                non_overlapped_segments.append(((time_s, time_e), speaker))
            else:
                if segment_key not in seen_segments:
                    resegmented_diar.append(((time_s, time_e), speaker))
                    seen_segments.add(segment_key)
        return resegmented_diar

    def map_speaker_info(self, diar_results, embeddings, threshold=0.65):
        speaker_dict = dict() 
        speaker_no = -1 

        for idx, chunk_embedding in enumerate(embeddings): 
            if idx == 0:
                # 첫 청크: 그냥 등록
                for idx2, speaker_emb in enumerate(chunk_embedding):
                    speaker_no += 1
                    speaker_dict[f'speaker_{str(speaker_no).zfill(2)}'] = speaker_emb
            else:
                for idx2, speaker_emb in enumerate(chunk_embedding):
                    best_similarity = -1
                    best_key = None

                    for key, value in speaker_dict.items():
                        emb_similarity = self.calc_emb_similarity(value, speaker_emb)
                        print(f'emb similarity of {key}-{idx2}, chunk {idx}: {emb_similarity}')
                        if emb_similarity > best_similarity:
                            best_similarity = emb_similarity
                            best_key = key

                    if best_similarity >= threshold:
                        print(f'Mapped {idx2} in chunk {idx} to {best_key}')
                    else:
                        speaker_no += 1
                        new_speaker_key = f'speaker_{str(speaker_no).zfill(2)}'
                        speaker_dict[new_speaker_key] = speaker_emb
                        print(f'New speaker {new_speaker_key} registered from {idx2} in chunk {idx}')

    def save_as_rttm(self, diar_result, output_rttm_path=None, file_name=None):
        '''
        rttm: SPEAKER <file-id> <channel> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
        '''
        with open(output_rttm_path, "w") as f:
            for (start_time, end_time), speaker in diar_result:
                duration = end_time - start_time
                rttm_line = f"SPEAKER {file_name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
                f.write(rttm_line)

    def save_as_emb(self, embeddings, output_emb_path=None):
        import numpy as np 
        np.save(output_emb_path, embeddings)

    def load_emb_npy(self, npy_emb_path=None):
        if npy_emb_path is None:
            raise ValueError("npy_emb_path must be specified.")
        embeddings = np.load(npy_emb_path)
        return embeddings


class PyannotOSD(Pyannot):   # Overlap Speech Detection
    def __init__(self):
        super().__init__()

    def get_overlapped_result(self, pipeline, audio_file):
        overlap_result = pipeline(audio_file)
        return overlap_result