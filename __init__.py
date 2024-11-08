
from .nodes.cai_download import Kw_Json_Lora_CivitAIDownloader
from .nodes.Json_Lora_loader import Kw_JsonLoraLoader

NODE_CLASS_MAPPINGS = { 
    "Kw_Json_Lora_CivitAIDownloader":Kw_Json_Lora_CivitAIDownloader,
    "Kw_JsonLoraLoader":Kw_JsonLoraLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "Kw_Json_Lora_CivitAIDownloader" : "Kw_Json_Lora_CivitAIDownloader",
    "Kw_JsonLoraLoader" : "Kw_JsonLoraLoader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']