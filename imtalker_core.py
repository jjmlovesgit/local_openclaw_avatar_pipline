import os
import tempfile
import sys
import torch
import time
import numpy as np
import cv2
import subprocess
import face_alignment
import torchvision.transforms as transforms
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor
import math
import io

# Try importing local model definitions
try:
    from generator.FM import FMGenerator
    from renderer.models import IMTRenderer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'generator' and 'renderer' folders are in this directory.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
class AppConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rank = "cuda" if torch.cuda.is_available() else "cpu" 
        
        self.seed = 42
        self.fix_noise_seed = False
        self.crop = True
        
        # Paths
        self.renderer_path = "./checkpoints/renderer.ckpt"
        self.generator_path = "./checkpoints/generator.ckpt"
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"
        
        # Placeholders
        self.ref_path = None
        self.pose_path = None
        self.gaze_path = None
        self.aud_path = None
        self.source_path = None
        self.driving_path = None
        
        # Inference Settings
        self.input_size = 256
        self.fps = 25.0
        self.sampling_rate = 16000
        self.a_cfg_scale = 3.0
        
        # Model Architecture
        self.input_nc = 3
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.nfe = 10
        self.torchdiffeq_ode_method = 'euler'
        self.swin_res_threshold = 128
        self.window_size = 8
        
        # SPEED OPTIMIZATIONS (FFmpeg only, NO FP16 for renderer)
        self.fast_ffmpeg = True  # Use ultrafast preset
        self.video_resolution = (512, 512)  # Keep original resolution


# ==========================================
# 2. DATA PROCESSOR
# ==========================================
class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.sampling_rate = opt.sampling_rate
        
        print("Loading Face Detector...")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)
        
        print("Loading Wav2Vec2...")
        if os.path.exists(opt.wav2vec_model_path):
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        else:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        try:
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
        except:
            bboxes = None
            
        if bboxes is None or len(bboxes) == 0:
            print("Warning: No face detected. Using center crop.")
            short_side = min(h, w)
            cx, cy = w // 2, h // 2
            x1, y1 = cx - short_side // 2, cy - short_side // 2
            x2, y2 = x1 + short_side, y1 + short_side
        else:
            x1, y1, x2, y2, _ = bboxes[0]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            size = int(max(x2-x1, y2-y1) * 0.8)
            x1 = max(0, int(cx - size))
            y1 = max(0, int(cy - size))
            x2 = min(w, int(cx + size))
            y2 = min(h, int(cy + size))
            side = min(x2-x1, y2-y1)
            x2, y2 = x1 + side, y1 + side

        crop_img = img_arr[int(y1):int(y2), int(x1):int(x2)]
        return Image.fromarray(crop_img).resize((self.opt.input_size, self.opt.input_size))

    def process_audio(self, path: str) -> torch.Tensor:
        import librosa
        import numpy as np
        
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        
        speech_array = speech_array - np.mean(speech_array)
        
        limit_threshold = 0.95
        speech_array = np.clip(speech_array, -limit_threshold, limit_threshold)
        
        window_size = 3
        if len(speech_array) > window_size:
            window = np.ones(window_size) / window_size
            speech_array = np.convolve(speech_array, window, mode='same')
        
        fade_duration = 0.05 
        fade_samples = int(fade_duration * sampling_rate)
        
        if len(speech_array) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            speech_array[:fade_samples] *= fade_in
            speech_array[-fade_samples:] *= fade_out
            
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors='pt').input_values[0]


# ==========================================
# 3. INFERENCE AGENT (NO FP16 - KEEPS VIDEO QUALITY)
# ==========================================
class InferenceAgent:            
    def __init__(self, opt=None):
        if opt is None:
            opt = AppConfig()
        
        self.opt = opt
        self.device = opt.device
        self.data_processor = DataProcessor(opt)
        
        print("--- Initializing Inference Agent ---")
        
        self.renderer = IMTRenderer(self.opt).to(self.device)
        self.generator = FMGenerator(self.opt).to(self.device)
        
        print("ℹ️ Loading model weights...")
        self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
        self._load_fm_ckpt(self.generator, self.opt.generator_path)
        
        self.renderer.eval()
        self.generator.eval()
        
        self.driver_cache = {
            'latents': None,
            'video_path': None,
            'num_frames': 0
        }
        
        print("--- Models Loaded Successfully ---")
    
    def _load_ckpt(self, model, path, prefix="gen."):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        model.load_state_dict(clean_state_dict, strict=False)
    
    def _load_fm_ckpt(self, model, path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict: 
            state_dict = state_dict['model']
        prefix = 'model.'
        clean_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(self.device))
    
    def warmup(self):
        print("\n🔥 Starting warmup...")
        start_time = time.time()
        
        dummy_img = torch.randn(1, 3, 256, 256).to(self.device)
        dummy_audio = torch.randn(1, 16000).to(self.device)
        
        try:
            if os.path.exists("avatar.png"):
                real_img = Image.open("avatar.png").convert('RGB')
                real_img = self.data_processor.process_img(real_img)
                dummy_img = self.data_processor.transform(real_img).unsqueeze(0).to(self.device)
        except:
            pass
        
        with torch.no_grad():
            try:
                for i in range(2):
                    print(f"  Warmup iteration {i+1}/2...")
                    
                    f_r, g_r = self.renderer.dense_feature_encoder(dummy_img)
                    t_lat = self.renderer.latent_token_encoder(dummy_img)
                    if isinstance(t_lat, tuple): 
                        t_lat = t_lat[0]
                    
                    dummy_data = {'s': dummy_img, 'a': dummy_audio, 'ref_x': t_lat}
                    sample = self.generator.sample(dummy_data, a_cfg_scale=self.opt.a_cfg_scale, nfe=4, seed=self.opt.seed)
                    
                    ta_r = self.renderer.adapt(t_lat, g_r)
                    m_r = self.renderer.latent_token_decoder(ta_r)
                    
                    for t in range(min(2, sample.shape[1])):
                        ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                        m_c = self.renderer.latent_token_decoder(ta_c)
                        _ = self.renderer.decode(m_c, m_r, f_r)
                
                print(f"  ✓ Warmup successful")
            except Exception as e:
                print(f"  ⚠ Warmup warning: {e}")
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        print(f"✅ Warmup complete in {elapsed:.2f}s\n")
        return True
    
    def try_safe_compile(self):
        print("\n" + "="*50)
        print("🔄 Attempting torch.compile optimization...")
        print("="*50)
        
        try:
            self.generator = torch.compile(self.generator, mode="reduce-overhead")
            print("✅ Generator compiled successfully!")
            self.renderer = torch.compile(self.renderer, mode="reduce-overhead")
            print("✅ Renderer compiled successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Compilation skipped: {e}")
            return False

    @torch.no_grad()
    def generate(self, img_pil, aud_path, output_path, nfe=4, is_first_chunk=False):
        import cv2
        import numpy as np
        
        s_pil = self.data_processor.process_img(img_pil)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        
        data = {'s': s_tensor, 'a': a_tensor, 'ref_x': t_lat}
        torch.manual_seed(self.opt.seed)
        sample = self.generator.sample(data, a_cfg_scale=self.opt.a_cfg_scale, nfe=nfe, seed=self.opt.seed)
        
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        frames_buffer = io.BytesIO()
        num_frames = sample.shape[1]
        
        print(f"Rendering {num_frames} frames to memory buffer...")
        for t in range(num_frames):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_tensor = self.renderer.decode(m_c, m_r, f_r)
            
            frame = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1) * 255
            frame = frame.astype(np.uint8)
            frames_buffer.write(frame.tobytes())
        
        raw_video_data = frames_buffer.getvalue()
        print(f"Video buffer: {len(raw_video_data)} bytes ({num_frames} frames)")
        
        audio_duration = a_tensor.shape[-1] / 16000
        fade_duration = 0.05
        afade_filter = f'afade=t=in:ss=0:d={fade_duration},afade=t=out:st={max(0, audio_duration-fade_duration)}:d={fade_duration}'
        
        w, h = self.opt.video_resolution
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', '25',
            '-i', 'pipe:0',
            '-i', aud_path,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level:v', '3.0',
            '-x264-params', 'keyint=25:min-keyint=1:scenecut=0',
            '-pix_fmt', 'yuv420p',
            '-color_range', 'pc',
            '-vf', 'scale=in_range=full:out_range=full',
            '-preset', 'ultrafast' if self.opt.fast_ffmpeg else 'medium',
            '-crf', '23' if self.opt.fast_ffmpeg else '18',
            '-c:a', 'aac',
            '-b:a', '128k' if self.opt.fast_ffmpeg else '192k',
            '-af', afade_filter,
            '-shortest',
            '-movflags', '+faststart',
            output_path,
            '-loglevel', 'error'
        ]
        
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = process.communicate(input=raw_video_data)
        
        if process.returncode == 0:
            print(f"✅ Generation complete: {output_path}")
            return output_path
        else:
            error_msg = stderr_data.decode()[:500] if stderr_data else "Unknown error"
            print(f"❌ FFmpeg failed: {error_msg}")
            return None

    @torch.no_grad()
    def generate_fast(self, img_pil, aud_path, output_path):
        """Ultra-fast generation for first chunk (NFE=2)"""
        return self.generate(img_pil, aud_path, output_path, nfe=2, is_first_chunk=True)

    @torch.no_grad()
    def generate_with_driver(self, img_pil, aud_path, driver_video_path, output_path, nfe=4, is_first_chunk=False):
        import cv2
        import numpy as np
        from PIL import Image
        import math
        
        s_pil = self.data_processor.process_img(img_pil)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        if (self.driver_cache['video_path'] != driver_video_path or
            self.driver_cache['latents'] is None):
            print(f"🔍 Encoding driver video: {driver_video_path}")
            
            cap = cv2.VideoCapture(driver_video_path)
            if not cap.isOpened():
                return self.generate(img_pil, aud_path, output_path, nfe, is_first_chunk)
            
            driver_frames = []
            max_frames = 900
            
            while len(driver_frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_processed = self.data_processor.process_img(frame_pil)
                frame_tensor = self.data_processor.transform(frame_processed).unsqueeze(0).to(self.device)
                driver_frames.append(frame_tensor)
            
            cap.release()
            
            if not driver_frames:
                return self.generate(img_pil, aud_path, output_path, nfe, is_first_chunk)
            
            print(f"✅ Loaded {len(driver_frames)} driver frames")
            
            batch_size = 16
            video_latents = []
            
            for i in range(0, len(driver_frames), batch_size):
                batch_frames = driver_frames[i:i+batch_size]
                batch_tensor = torch.cat(batch_frames, dim=0)
                t_v = self.renderer.latent_token_encoder(batch_tensor)
                if isinstance(t_v, tuple):
                    t_v = t_v[0]
                video_latents.append(t_v)
            
            video_latents_tensor = torch.cat(video_latents, dim=0)
            
            self.driver_cache['latents'] = video_latents_tensor
            self.driver_cache['video_path'] = driver_video_path
            self.driver_cache['num_frames'] = video_latents_tensor.shape[0]
            
            print(f"✅ Cached {self.driver_cache['num_frames']} driver latents")
        else:
            print(f"♻️ Using cached driver latents ({self.driver_cache['num_frames']} frames)")
        
        video_latents_tensor = self.driver_cache['latents']
        
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        
        data = {'s': s_tensor, 'a': a_tensor, 'ref_x': t_lat}
        torch.manual_seed(self.opt.seed)
        audio_sample = self.generator.sample(data, a_cfg_scale=self.opt.a_cfg_scale, nfe=nfe, seed=self.opt.seed)
        
        audio_frames = audio_sample.shape[1]
        driver_frames = video_latents_tensor.shape[0]
        
        if audio_frames > driver_frames:
            repeat_count = math.ceil(audio_frames / driver_frames)
            repeated_latents = video_latents_tensor.repeat(repeat_count, 1)
            video_latents_trimmed = repeated_latents[:audio_frames]
        else:
            video_latents_trimmed = video_latents_tensor[:audio_frames]
        
        video_latents_expanded = video_latents_trimmed.unsqueeze(0)
        blend_ratio = 0.3
        blended = audio_sample * (1 - blend_ratio) + video_latents_expanded * blend_ratio
        
        frames_buffer = io.BytesIO()
        num_frames = blended.shape[1]
        
        print(f"Rendering {num_frames} blended frames...")
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        for t in range(num_frames):
            frame_latent = blended[:, t, ...]
            ta_c = self.renderer.adapt(frame_latent, g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_tensor = self.renderer.decode(m_c, m_r, f_r)
            
            frame = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1) * 255
            frame = frame.astype(np.uint8)
            frames_buffer.write(frame.tobytes())
        
        raw_video_data = frames_buffer.getvalue()
        
        audio_duration = a_tensor.shape[-1] / 16000
        fade_duration = 0.05
        afade_filter = f'afade=t=in:ss=0:d={fade_duration},afade=t=out:st={max(0, audio_duration-fade_duration)}:d={fade_duration}'
        
        w, h = self.opt.video_resolution
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', '25',
            '-i', 'pipe:0',
            '-i', aud_path,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level:v', '3.0',
            '-x264-params', 'keyint=25:min-keyint=1:scenecut=0',
            '-pix_fmt', 'yuv420p',
            '-color_range', 'pc',
            '-vf', 'scale=in_range=full:out_range=full',
            '-preset', 'ultrafast' if self.opt.fast_ffmpeg else 'medium',
            '-crf', '23' if self.opt.fast_ffmpeg else '18',
            '-c:a', 'aac',
            '-b:a', '128k' if self.opt.fast_ffmpeg else '192k',
            '-af', afade_filter,
            '-shortest',
            '-movflags', '+faststart',
            output_path,
            '-loglevel', 'error'
        ]
        
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = process.communicate(input=raw_video_data)
        
        if process.returncode == 0:
            print(f"✅ Hybrid generation complete: {output_path}")
            return output_path
        else:
            error_msg = stderr_data.decode()[:500] if stderr_data else "Unknown error"
            print(f"❌ Hybrid generation failed: {error_msg}")
            return None


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    opt = AppConfig()
    agent = InferenceAgent(opt)
    
    print("\n=== Running warmup ===")
    agent.warmup()
    
    print("✅ IMTalker ready for inference!")
    