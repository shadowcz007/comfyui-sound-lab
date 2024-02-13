import os,sys,base64
import folder_paths
import numpy as np

import torch
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download

# print('#######s',os.path.join(__file__,'../'))

sys.path.append(os.path.join(__file__,'../../'))
                
from scipy.io import wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


modelpath=os.path.join(folder_paths.models_dir, "musicgen-small")


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
                            "dynamicPrompts": False
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

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio"

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

        if not os.path.exists(folder_paths.get_temp_directory()):
            os.mkdir(folder_paths.get_temp_directory())

        fp=os.path.join(folder_paths.get_temp_directory(),'output_file.wav')
        # save the best audio sample (index 0) as a .wav file
        wavfile.write(fp, rate=sampling_rate, data=audio)

        with open(fp, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = f'data:audio/wav;base64,'+base64.b64encode(audio_data).decode("utf-8")

        return {"ui": {"audio_base64": [audio_base64]}, "result": (audio_base64,)}