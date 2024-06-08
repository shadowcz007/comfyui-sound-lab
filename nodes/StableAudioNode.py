import os,sys
import torch
import torchaudio
from einops import rearrange

import numpy as np

from safetensors.torch import load_file
from .stable_audio_tools.models.factory import create_model_from_config
from .stable_audio_tools.models.utils import load_ckpt_state_dict
from .stable_audio_tools import get_pretrained_model
from .stable_audio_tools.inference.generation import generate_diffusion_cond

import folder_paths

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


from .utils import get_new_counter

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_model_config():
    return {
        "model_type": "diffusion_cond",
        "sample_size": 2097152,
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "iterate_batch": True,
                "config": {
                    "encoder": {
                        "type": "oobleck",
                        "requires_grad": False,
                        "config": {
                            "in_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 128,
                            "use_snake": True
                        }
                    },
                    "decoder": {
                        "type": "oobleck",
                        "config": {
                            "out_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 64,
                            "use_snake": True,
                            "final_tanh": False
                        }
                    },
                    "bottleneck": {
                        "type": "vae"
                    },
                    "latent_dim": 64,
                    "downsampling_ratio": 2048,
                    "io_channels": 2
                }
            },
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "t5",
                        "config": {
                            "t5_model_name": "t5-base",
                            "max_length": 128
                        }
                    },
                    {
                        "id": "seconds_start",
                        "type": "number",
                        "config": {
                            "min_val": 0,
                            "max_val": 512
                        }
                    },
                    {
                        "id": "seconds_total",
                        "type": "number",
                        "config": {
                            "min_val": 0,
                            "max_val": 512
                        }
                    }
                ],
                "cond_dim": 768
            },
            "diffusion": {
                "cross_attention_cond_ids": ["prompt", "seconds_start", "seconds_total"],
                "global_cond_ids": ["seconds_start", "seconds_total"],
                "type": "dit",
                "config": {
                    "io_channels": 64,
                    "embed_dim": 1536,
                    "depth": 24,
                    "num_heads": 24,
                    "cond_token_dim": 768,
                    "global_cond_dim": 1536,
                    "project_cond_tokens": False,
                    "transformer_type": "continuous_transformer"
                }
            },
            "io_channels": 64
        },
        "training": {
            "use_ema": True,
            "log_loss_info": False,
            "optimizer_configs": {
                "diffusion": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": 5e-5,
                            "betas": [0.9, 0.999],
                            "weight_decay": 1e-3
                        }
                    },
                    "scheduler": {
                        "type": "InverseLR",
                        "config": {
                            "inv_gamma": 1000000,
                            "power": 0.5,
                            "warmup": 0.99
                        }
                    }
                }
            },
            "demo": {
                "demo_every": 2000,
                "demo_steps": 250,
                "num_demos": 4,
                "demo_cond": [
                    {"prompt": "Amen break 174 BPM", "seconds_start": 0, "seconds_total": 12},
                    {"prompt": "A beautiful orchestral symphony, classical music", "seconds_start": 0, "seconds_total": 160},
                    {"prompt": "Chill hip-hop beat, chillhop", "seconds_start": 0, "seconds_total": 190},
                    {"prompt": "A pop song about love and loss", "seconds_start": 0, "seconds_total": 180}
                ],
                "demo_cfg_scales": [3, 6, 9]
            }
        }
    }


modelpath=os.path.join(folder_paths.models_dir, "stable_audio")

def get_model_filenames(root):
    model_filenames = []
    for filename in os.listdir(root):
        if filename.endswith(('.safetensors', '.ckpt')):
            model_filenames.append(filename)
    return model_filenames[0]

# 加载模型
def load_model(device):
    model_path=""
    try:
        model_filename=get_model_filenames(modelpath)
        model_path=os.path.join(modelpath,model_filename)
    except:
        print('##will download')

    if model_path.endswith(".safetensors") or model_path.endswith(".ckpt"):
        model_config = get_model_config()
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(model_path))
        
    else:
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    model = model.to(device)
    return model,sample_rate,sample_size


def generate(model,prompt,seed,steps,cfg_scale,sample_size, sigma_min, sigma_max, sampler_type,device):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": 30
    }]
    
    if seed==-1:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return output


class StableAudioSampler:
    def __init__(self):
        self.initialized_model = None
        self.sample_rate=None
        self.sample_size=None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '128 BPM tech house drum loop',
                            "dynamicPrompts": True
                          }),
                
                "steps": ("INT", {"default": 16, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}), 
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                # "sampler_type": ("STRING", {"default": "dpmpp-3m-sde"}),
                "seed":  ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}), 
                "device":(["auto","cpu"],),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "run"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    CATEGORY = "♾️Sound Lab"

    def run(self, prompt,steps, cfg_scale,  sigma_min, sigma_max, seed, device):

        if device=='auto':
            device="cuda" if torch.cuda.is_available() else "cpu"

        if self.initialized_model:
            self.initialized_model=self.initialized_model.to(device)
        else:
            self.initialized_model,self.sample_rate,self.sample_size=load_model(device)

        output=generate(self.initialized_model,prompt,seed,steps,cfg_scale,self.sample_size, sigma_min, sigma_max, "dpmpp-3m-sde",device)

        self.initialized_model.to(torch.device('cpu'))

        output_dir = folder_paths.get_output_directory()
    
        audio_file="stabe_audio"
        counter=get_new_counter(output_dir,audio_file)
        # print('#audio_path',folder_paths, )
        # 添加文件名后缀
        audio_file = f"{audio_file}_{counter:05}.wav"
        
        audio_path=os.path.join(output_dir, audio_file)

        torchaudio.save(audio_path, output, self.sample_rate)
        
        return ({
                "filename": audio_file,
                "subfolder": "",
                "type": "output"
                },)
    