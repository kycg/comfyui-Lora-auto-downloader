
from .nodes.cai_download import Kw_Json_Lora_CivitAIDownloader

NODE_CLASS_MAPPINGS = { 
    "Kw_Json_Lora_CivitAIDownloader":Kw_Json_Lora_CivitAIDownloader,
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "Kw_Json_Lora_CivitAIDownloader" : "Kw_Json_Lora_CivitAIDownloader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']