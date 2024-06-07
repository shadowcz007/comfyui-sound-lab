import os,sys,base64
import folder_paths
import numpy as np

import torch
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download

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


def init_audio_model(checkpoint):

    audio_processor = AutoProcessor.from_pretrained(checkpoint)

    audio_model = MusicgenForConditionalGeneration.from_pretrained(checkpoint)

    # audio_model.to(device)
    audio_model = audio_model.to(torch.device('cpu'))

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
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '',
                            "dynamicPrompts": True
                          }),
            "guidance_scale":("FLOAT", {
                        "default": 4.0, 
                        "min": 0, #Minimum value
                        "max": 20, #Maximum value
                    }),

            "max_tokens":("INT", {
                        "default": 256, 
                        "min": 0, #Minimum value
                        "max": 2048, #Maximum value
                        "step": 1, #Slider's step
                        "display": "number" # Cosmetic only: display as "number" or "slider"
                    }),
            # "device": (["cpu","cuda"],),
                             },

            
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "run"

    CATEGORY = "♾️Sound Lab"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def run(self,prompt,guidance_scale,max_tokens):
      
        if self.audio_model ==None:
            
            if os.path.exists(modelpath)==False:
                os.mkdir(modelpath)

            if os.path.exists(modelpath):

                config=os.path.join(modelpath,'config.json')
                if os.path.exists(config)==False:
                    snapshot_download("facebook/musicgen-small",
                                                local_dir=modelpath,
                                                # local_dir_use_symlinks=False,
                                                # filename="config.json",
                                                endpoint='https://hf-mirror.com')
                
                self.audio_processor,self.audio_model=init_audio_model(modelpath)
                
          
        inputs = self.audio_processor(
            text=prompt,
            # audio=audio,
            # sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        self.audio_model.to(torch.device('cuda'))

        # max_tokens=256 #default=5, le=30
        # if duration:
        #     max_tokens=int(duration*50)
      
        sampling_rate = self.audio_model.config.audio_encoder.sampling_rate
        # input_audio
        audio_values = self.audio_model.generate(**inputs.to('cuda'), 
                    do_sample=True, 
                    guidance_scale=guidance_scale, 
                    max_new_tokens=max_tokens,
                    )
        
        self.audio_model.to(torch.device('cpu'))

        audio=audio_values[0, 0].cpu().numpy()

        output_dir = folder_paths.get_output_directory()
    
        audio_file="music_gen"
        counter=get_new_counter(output_dir,audio_file)
        # print('#audio_path',folder_paths, )
        # 添加文件名后缀
        audio_file = f"{audio_file}_{counter:05}.wav"
        
        audio_path=os.path.join(output_dir, audio_file)
 
        # save the best audio sample (index 0) as a .wav file
        wavfile.write(audio_path, rate=sampling_rate, data=audio)

        # with open(audio_path, "rb") as audio_file:
        #     audio_data = audio_file.read()
        #     audio_base64 = f'data:audio/wav;base64,'+base64.b64encode(audio_data).decode("utf-8")

        return ({
                "filename": audio_file,
                "subfolder": "",
                "type": "output"
                },)
    


class AudioPlayNode:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
  
    FUNCTION = "run"

    CATEGORY = "♾️Sound Lab"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def run(self,audio):
        print(audio)
        return {"ui": {"audio":[audio]}}