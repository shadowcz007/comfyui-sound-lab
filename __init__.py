from .nodes.musicNode import MusicNode,AudioPlayNode

NODE_CLASS_MAPPINGS = {
    "Musicgen": MusicNode,
    "AudioPlay":AudioPlayNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Music Gen": "Musicgen",
    "Audio Play":"AudioPlay"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"