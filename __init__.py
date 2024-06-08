from .nodes.musicNode import MusicNode,AudioPlayNode
from .nodes.StableAudioNode import StableAudioNode

NODE_CLASS_MAPPINGS = {
    "Musicgen_": MusicNode,
    "AudioPlay":AudioPlayNode,
    "StableAudio_":StableAudioNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Musicgen_": "Music Gen",
    "AudioPlay":"Audio Play",
    "StableAudio_":"Stable Audio"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"