from src import NoiseHandler, VoiceEnhancer, AudioFileProcessor, AudioVisualizer
from src import PyannotDIAR, PyannotVAD
from src import SBEMB
from src import WhisperSTT
from abc import abstractmethod
from pydub import AudioSegment
from io import BytesIO
import tempfile
from scipy.spatial.distance import cosine


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
    def set_env(self, diar_config):
        self.diar_model = PyannotDIAR()
        self.diar_config = diar_config 
        self.audio_visualizer = AudioVisualizer()
        self.audio_file_processor = AudioFileProcessor()
    
    def calc_emb_similarity(self, emb1, emb2):
        return 1 - cosine(emb1, emb2)   # cosine()은 distance니까 1 - distance

    def get_diar(self, audio_file, chunk_length=300, num_speakers=None, return_embeddings=False, file_name=None):
        '''audio_file: AudioSeg'''
        diar_pipe = self.diar_model.load_pipeline_from_pretrained(self.diar_config)
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=chunk_length)
        results = []; emb_results = []
        save_file_name = f"{file_name.split('/')[-1].split('.')[0]}"
        for idx, chunk in enumerate(chunks):
            if not chunk_length is None:
                save_file_name = f"chunk_{idx}_{file_name.split('/')[-1].split('.')[0]}"
            save_rttm_path = './dataset/rttm/' + save_file_name + '.rttm'
            save_emb_path = './dataset/emb/' + save_file_name + '.npy'
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                diar_result, emb = self.diar_model.get_diar_result(diar_pipe, temp_audio.name, num_speakers=num_speakers, return_embeddings=return_embeddings)
                self.diar_model.save_as_rttm(diar_result, output_rttm_path=save_rttm_path, file_name=save_file_name)
                self.diar_model.save_as_emb(emb, output_emb_path=save_emb_path)
            results.append(diar_result)
            emb_results.append(emb)
        return results, emb_results

    def map_speaker_info(self, diar_results, embeddings):
        '''
        get diar results of audio file chunks
        diar_result: [(start_time, end_time), speaker info]
        emb: [(speaker 0 emb), (speaker 1 emb), (speaker 2 emb), ...] 
        '''
        embeddings[0][0] 
        '''
        speaker_dict = dict() 
        for idx, embedding in enumerate(embeddings): 
            if idx == 0:
                for idx2, speaker_emb in enumerate(embedding):
                    speaker_dict[f'speaker_{str(idx2).zfill(2)}'] = speaker_emb 
            else: 
                verify_speaker(speaker_dict, embedding) 
        '''

class PostProcessPipe(BasePipeline):
    def set_env(self, emb_config):
        self.audio_file_processor = AudioFileProcessor()
        self.emb_model = SBEMB(emb_config)

    def get_vectoremb(self):
        pass 
    
    def cut_audio(self):
        '''
        cut audio by timestamp
        '''
        pass 

    def map_speaker(self):
        pass 


class STTPipe(BasePipeline):
    def set_env(self, whisper_api, generation_config):
        self.stt_model = WhisperSTT(whisper_api, generation_config)
    