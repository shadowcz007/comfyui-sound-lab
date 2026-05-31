import os,sys,base64
import folder_paths
import numpy as np

import torch,random
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download

import torchaudio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MUSICGEN_ROOT = os.path.join(folder_paths.models_dir, "musicgen")

def get_musicgen_models():

    if not os.path.exists(MUSICGEN_ROOT):
        return ["none"]

    valid = []

    for d in os.listdir(MUSICGEN_ROOT):

        path = os.path.join(MUSICGEN_ROOT, d)

        if not os.path.isdir(path):
            continue

        for root, dirs, files in os.walk(path):

            if (
                "config.json" in files or
                "preprocessor_config.json" in files
            ):
                valid.append(d)
                break

    return sorted(valid) if valid else ["none"]
	
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)

       
from scipy.io import wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .utils import get_new_counter


modelpath=os.path.join(folder_paths.models_dir, "musicgen")
#modelpath = r"D:\comfyuser\models\musicgen\models--facebook--musicgen-small\snapshots\4c8334b02c6ec4e8664a91979669a501ec497792"

## HELPER
def resolve_model_path(model_name):
    base_path = os.path.join(MUSICGEN_ROOT, model_name)

    # Direct model folder
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path

    snapshots_dir = os.path.join(base_path, "snapshots")

    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(
            f"No snapshots folder found: {snapshots_dir}"
        )

    for snapshot_hash in os.listdir(snapshots_dir):

        snapshot_path = os.path.join(
            snapshots_dir,
            snapshot_hash
        )

        if not os.path.isdir(snapshot_path):
            continue

        if os.path.exists(
            os.path.join(snapshot_path, "preprocessor_config.json")
        ):
            return snapshot_path

        if os.path.exists(
            os.path.join(snapshot_path, "config.json")
        ):
            return snapshot_path

    raise FileNotFoundError(
        f"No valid snapshot found for {model_name}"
    )

def init_audio_model(checkpoint, device):

    print("CHECKPOINT =", checkpoint)
    print("FILES =", os.listdir(checkpoint))

    audio_processor = AutoProcessor.from_pretrained(checkpoint)

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    #audio_model = MusicgenForConditionalGeneration.from_pretrained(checkpoint)
    audio_model = MusicgenForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # audio_model.to(device)
    #audio_model = audio_model.to(torch.device('cpu'))
    audio_model = audio_model.to(torch.device(device))

    # increase the guidance scale to 4.0
    audio_model.generation_config.guidance_scale = 4.0

    # set the max new tokens to 256
    # 1500 - 30s
    audio_model.generation_config.max_new_tokens = 1500

    # set the softmax sampling temperature to 1.5
    audio_model.generation_config.temperature = 1.5
   
    return (audio_processor,audio_model)


class MusicNode:
    def __init__(self):
        self.audio_model = None
        self.audio_processor = None
        self.current_model = None
        self.current_device = None
    @classmethod
    def INPUT_TYPES(s):
        SAMPLING_RATES = {
            "16000": "16 kHz (Voice)",
            "22050": "22.05 kHz",
            "32000": "32 kHz (MusicGen default)",
            "44100": "44.1 kHz (CD)",
            "48000": "48 kHz (Pro audio)"
        }		
        return{"required": {
            "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '',
                            "dynamicPrompts": True
                          }),
            
            "seconds":("FLOAT", {
                        "default": 5, 
                        "min": 1, #Minimum value
                        "max": 1000, #Maximum value
                        "step": 0.1, #Slider's step
                        "display": "number" # Cosmetic only: display as "number" or "slider"
                    }),
            "guidance_scale":("FLOAT", {
                        "default": 4.0, 
                        "min": 0, #Minimum value
                        "max": 20, #Maximum value
                    }),

            "seed":  ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}), 

            "device": (["auto","cuda","cpu"],{"default": "auto"}),
            "model": (get_musicgen_models(),),
            "output_mode": (
                ["Original Mono", "Enhanced Stereo"],
                {"default": "Enhanced Stereo"}
            ),            
		}}
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "run"

    CATEGORY = "♾️Sound Lab"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
   
    def unload_model(self):

        if self.audio_model is not None:

            self.audio_model.cpu()

            del self.audio_model
            self.audio_model = None

        if self.audio_processor is not None:

            del self.audio_processor
            self.audio_processor = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def run(self,prompt,seconds,guidance_scale,seed,device,output_mode,
model):
        print("NODE ID:", id(self))
        print("MODEL OBJ:", id(self.audio_model) if self.audio_model else None)
        
        if seed==-1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        # Set the seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model_path = resolve_model_path(model)

        print("MODEL:", model)
        print("RESOLVED MODEL PATH:", model_path)

        # --------------------------------------------------
        # Resolve target device
        # --------------------------------------------------

        requested_device = device

        if requested_device == "auto":

            requested_device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            
        try:

            if (
                self.audio_model is not None
                and getattr(
                    self,
                    "current_device",
                    None
                ) != requested_device
            ):
                print(
                    f"Moving model "
                    f"{getattr(self, 'current_device', 'none')} "
                    f"-> {requested_device}"
                )

                if self.audio_model is None:

                    print(
                        "Model not loaded yet, skipping device move"
                    )

                else:

                    self.audio_model = self.audio_model.to(
                        requested_device
                    )
                    
                if self.audio_model is not None:

                    if requested_device == "cuda":
                        self.audio_model = self.audio_model.half()
                    else:
                        self.audio_model = self.audio_model.float()
                        
                self.current_device = requested_device

        except RuntimeError as e:

            if requested_device == "cuda":

                print(
                    f"CUDA failed: {e}"
                )

                print(
                    "Falling back to CPU"
                )

                if self.audio_model is not None:
                    self.audio_model = self.audio_model.to("cpu")
                    self.audio_model = self.audio_model.float()

                self.current_device = "cpu"

                device = "cpu"

            else:
                raise
        
        # --------------------------------------------------
        # Load model if needed
        # --------------------------------------------------

        if (
            self.audio_model is None
            or self.current_model != model_path
        ):

            # unload previous model first
            self.unload_model()

            self.audio_processor, self.audio_model = (
                init_audio_model(
                    model_path,
                    requested_device
                )
            )

            self.current_model = model_path
            self.current_device = requested_device
    
        elif (
            requested_device == "cuda"
            and not torch.cuda.is_available()
        ):

            print(
                "CUDA requested but unavailable. "
                "Falling back to CPU."
            )

            requested_device = "cpu"

#        if (
#            self.audio_model is None
#            or getattr(self, "current_model", None) != model_path
#        ):

#            self.audio_processor, self.audio_model = init_audio_model(
#                model_path
#        )

        self.current_model = model_path
        
        #self.audio_processor, self.audio_model = init_audio_model(model_path)
        # ONLY load if not already loaded OR model changed
#        if self.audio_model is None or getattr(self, "current_model", None) != model_path:
#            self.audio_processor, self.audio_model = init_audio_model(model_path)
#            self.current_model = model_path

        if self.audio_model ==None:
            
            if os.path.exists(modelpath)==False:
                os.mkdir(modelpath)

            if os.path.exists(modelpath):

                config=os.path.join(modelpath,'config.json')
                if os.path.exists(config)==False:
#                    snapshot_download("facebook/musicgen-small",
#                                                local_dir=modelpath,
#                                                # local_dir_use_symlinks=False,
#                                                # filename="config.json",
#                                                endpoint='https://hf-mirror.com')
                    snapshot_download(
                        "facebook/musicgen-small",
                        local_dir=modelpath
                    )                
                self.audio_processor,self.audio_model=init_audio_model(modelpath)

        inputs = self.audio_processor(
            text=prompt,
            # audio=audio,
            # sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        # --------------------------------------------------
        # Move model only if device changed
        # --------------------------------------------------

        if (
            self.audio_model is not None
            and self.current_device != requested_device
        ):
            print(
                f"Moving model "
                f"{self.current_device} -> "
                f"{requested_device}"
            )

            try:

                self.audio_model = self.audio_model.to(
                    requested_device
                )

                if requested_device == "cuda":

                    self.audio_model = (
                        self.audio_model.half()
                    )

                else:

                    self.audio_model = (
                        self.audio_model.float()
                    )

                self.current_device = requested_device

            except Exception as e:

                print(
                    f"Failed moving to "
                    f"{requested_device}: {e}"
                )

                if requested_device == "cuda":

                    print(
                        "Retrying on CPU"
                    )

                    if self.audio_model is not None:

                        self.audio_model = (
                            self.audio_model.to("cpu")
                        )

                        self.audio_model = (
                            self.audio_model.float()
                        )
        
                    self.current_device = "cpu"
                    requested_device = "cpu"

                else:
                    raise
            
        # max_tokens=256 #default=5, le=30
        # if duration:
        #     max_tokens=int(duration*50)

        # seconds = 10  # 例如，用户输入的秒数
        tokens_per_second = 1500 / 30
        max_tokens = int(tokens_per_second * seconds)


        sampling_rate = self.audio_model.config.audio_encoder.sampling_rate
        #sampling_rate = int(sampling_rate)

        #if "musicgen" in modelpath.lower():
        #    sampling_rate = 32000

        # input_audio
#        audio_values = self.audio_model.generate(**inputs.to(device), 
#                    do_sample=True, 
#                    guidance_scale=guidance_scale, 
#                    max_new_tokens=max_tokens,
#                    )
#        
#        self.audio_model.to(torch.device('cpu'))

        # audio_values = self.audio_model.generate(
            # **inputs.to(device),
            # do_sample=True,
            # guidance_scale=guidance_scale,
            # max_new_tokens=max_tokens,
        # )

        with torch.inference_mode():

            audio_values = self.audio_model.generate(
                #**inputs.to(device),
                **inputs.to(requested_device),
                do_sample=True,
                guidance_scale=guidance_scale,
                max_new_tokens=max_tokens,
            )
            
# ─────────────────────────────────────────────
# 🎛 MINI DAW PROCESSING PIPELINE START
# ─────────────────────────────────────────────

        audio = audio_values[0].detach().cpu()

        # Normalize dimensions
        if audio.ndim == 3:
            audio = audio[0]

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        mode = output_mode.lower()
        # --------------------------------------------------
        # MONO OUTPUT
        # --------------------------------------------------
        if mode.startswith("original") or "mono" in mode:

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            audio_tensor = audio

        # --------------------------------------------------
        # STEREO OUTPUT
        # --------------------------------------------------
        else:

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            left = audio.clone()

            right = torch.roll(
                audio,
                shifts=8,
                dims=-1
            )

            side = (right - left) * 0.4

            left = torch.tanh(
                (left + side) * 1.05
            )

            right = torch.tanh(
                (right - side) * 1.05
            )

            audio_tensor = torch.stack(
                [
                    left.squeeze(),
                    right.squeeze()
                ],
                dim=0
            )
   
        #print("FINAL SHAPE:", audio_tensor.shape)

# ─────────────────────────────────────────────
# 🎛 MINI DAW PROCESSING PIPELINE END
# ─────────────────────────────────────────────

        #self.audio_model.to(torch.device('cpu'))
        
        if requested_device == "cuda":
            torch.cuda.empty_cache()

############

        output_dir = folder_paths.get_output_directory()
    
        audio_file="music_gen"
        counter=get_new_counter(output_dir,audio_file)
        # print('#audio_path',folder_paths, )
        # 添加文件名后缀
        audio_file = f"{audio_file}_{counter:05}.wav"
        
        audio_path=os.path.join(output_dir, audio_file)
 
        # save the best audio sample (index 0) as a .wav file
        #wavfile.write(audio_path, rate=sampling_rate, data=audio)

        # with open(audio_path, "rb") as audio_file:
        #     audio_data = audio_file.read()
        #     audio_base64 = f'data:audio/wav;base64,'+base64.b64encode(audio_data).decode("utf-8")

#        return ({
#                "filename": audio_file,
#                "subfolder": "",
#                "type": "output",
#                "prompt":prompt
#                },)
        # return ({
            # "waveform": audio_values.cpu(),
            # "sample_rate": sampling_rate
        # },)    

# Add batch dimension
        #print("RETURN SHAPE:", audio_tensor.unsqueeze(0).shape)
        
        audio_tensor = audio_tensor.unsqueeze(0)

        #print("RETURN SHAPE:", audio_tensor.shape)

        return ({
            "waveform": audio_tensor,
            "sample_rate": sampling_rate
        },)

class AudioPlayNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""
        self.compress_level = 4
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
  
    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def run(self,audio):

        # 判断是否是 Tensor 类型
        is_tensor = not isinstance(audio, dict)
        print('#判断是否是 Tensor 类型',is_tensor,audio)
        if not is_tensor and 'waveform' in audio and 'sample_rate' in audio:
            # {'waveform': tensor([], size=(1, 1, 0)), 'sample_rate': 44100}
            is_tensor=True

        if is_tensor:
            filename_prefix=""
            # 保存
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            results = list()
            
            filename_with_batch_num = filename.replace("%batch_num%", str(1))
            file = f"{filename_with_batch_num}_{counter:05}_.wav"
            
            #torchaudio.save(os.path.join(full_output_folder, file), audio['waveform'].squeeze(0), audio["sample_rate"])
            #results.append({
            #        "filename": file,
            #        "subfolder": subfolder,
            #        "type": self.type
            #    })
            
        else:
            results=[audio]
                

        # print(audio)
        return {"ui": {"audio":results}}

#todo
# class LoadAudioNode:

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#                     "audio": ("AUDIO",),
#               }, 
#                 }
    
#     RETURN_TYPES = ()
  
#     FUNCTION = "run"

#     CATEGORY = "♾️Sound Lab"

#     INPUT_IS_LIST = False
#     OUTPUT_IS_LIST = ()

#     OUTPUT_NODE = True
  
#     def run(self,audio):
#         # print(audio)
#         return {"ui": {"audio":[audio]}}