import json
from nodes import LoraLoader
from .utils.power_prompt_utils import get_lora_by_filename, get_checkpoint_by_filename
from .utils.log import log_node_info, log_node_warn
import folder_paths
import comfy.sd
import os

    #A node to load multiple LoRA modules from a JSON configuration.
class Kw_JsonLoraLoader:
    @classmethod
    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_config": ("STRING", {"forceInput": True}),  # User-provided JSON configuration for LoRA modules
            },
            "optional": {
                "Load_Local_Checkpoint": ("BOOLEAN", {"default": True}),
                "Local_model": ("MODEL",),
                "Local_clip": ("CLIP",),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "Positive", "Negative")
    FUNCTION = "load_loras"
    OUTPUT_NODE  = True
    CATEGORY = "loaders"
    
    
    
    def load_loras(self, lora_config, Load_Local_Checkpoint=True, Local_model=None, Local_clip=None, **kwargs):
        """Loads LoRA modules from JSON configuration and applies them."""

        model = Local_model
        clip = Local_clip
        Positive = ""
        Negative = ""

        if lora_config:
            try:
                lora_data = json.loads(lora_config)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for lora_config input")

            Positive = lora_data.get("positive", "")
            Negative = lora_data.get("negative", "")

            if not Load_Local_Checkpoint:
                # Load model and clip from JSON configuration
                checkpoints = lora_data.get("checkpoint", [])
                if checkpoints:
                    checkpoint = checkpoints[0]  # Assuming only one checkpoint is provided
                    ckpt_name = checkpoint.get("name")
                    ckpt_num = checkpoint.get("modelVersionId")
                    if ckpt_name and ckpt_num:
                        # Construct the checkpoint filename
                        ckpt_filename_base = f"{ckpt_name}_{ckpt_num}"
                        # Use the new function to get the checkpoint path
                        ckpt_filename = get_checkpoint_by_filename(ckpt_filename_base, log_node=self.NAME)
                        if ckpt_filename is None:
                            raise FileNotFoundError(f"Checkpoint file '{ckpt_filename_base}' not found.")
                        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_filename)
                        if ckpt_path is None or not os.path.exists(ckpt_path):
                            raise FileNotFoundError(f"Checkpoint file '{ckpt_filename}' does not exist at path '{ckpt_path}'.")
                        out = comfy.sd.load_checkpoint_guess_config(
                            ckpt_path,
                            output_vae=True,
                            output_clip=True,
                            embedding_directory=folder_paths.get_folder_paths("embeddings")
                        )
                        model, clip = out[:2]
                    else:
                        raise ValueError("Checkpoint 'name' or 'modelVersionId' missing in JSON.")
                else:
                    raise ValueError("No checkpoint information found in JSON.")
            else:
                # Use provided model and clip
                if model is None or clip is None:
                    raise ValueError("Load_Local_Checkpoint is True, but 'Local_model' or 'Local_clip' is not provided.")

            # Load and apply LoRA modules
            for lora in lora_data.get("lora", []):
                name = lora.get("name")
                model_version_id = lora.get("modelVersionId")
                strength = float(lora.get("strength", "1"))

                if not name or not model_version_id:
                    continue  # Skip invalid entries

                # Construct the LoRA filename base
                lora_filename_base = f"{name}_{model_version_id}"

                # Use the existing function to get the lora filename
                lora_filename = get_lora_by_filename(lora_filename_base, log_node=self.NAME)
                if lora_filename is None:
                    print(f"LoRA file '{lora_filename_base}' not found.")
                    continue

                # Load the LoRA and apply it
                if strength > 0:
                    loaded_lora = lora_filename  # Since get_lora_by_filename returns the path
                    model, clip = LoraLoader().load_lora(model, clip, loaded_lora, strength, strength)
                    print(f"Successfully loaded LoRA '{lora_filename_base}'")
                else:
                    print(f"Skipping LoRA '{lora_filename_base}' due to zero strength.")
        else:
            raise ValueError("No 'lora_config' provided.")

        return model, clip, Positive, Negative
