from .nodes.musicNode import MusicNode,AudioPlayNode
from .nodes.StableAudioNode import StableAudioSampler

NODE_CLASS_MAPPINGS = {
    "Musicgen": MusicNode,
    "AudioPlay":AudioPlayNode,
    "StableAudioSampler":StableAudioSampler
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Musicgen": "Music Gen",
    "AudioPlay":"Audio Play",
    "StableAudioSampler":"Stable Audio Sampler"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"