import json
from nodes import LoraLoader
from .json_lora_loader.power_prompt_utils import get_lora_by_filename
import folder_paths
import comfy.sd

class Kw_JsonLoraLoader:
    """A node to load multiple LoRA modules from a JSON configuration."""

    NAME = "Kw_JsonLoraLoader"
    CATEGORY = "loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_config": ("STRING",)  # User-provided JSON configuration for LoRA modules
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

    def load_loras(self, lora_config, Load_Local_Checkpoint=True, Local_model=None, Local_clip=None, **kwargs):
        """Loads LoRA modules from JSON configuration and applies them."""
        
        model = Local_model
        clip =Local_clip 
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
                    checkpoint = checkpoints[0]
                    ckpt_name = checkpoint.get("name")
                    ckpt_num = checkpoint.get("modelVersionId")
                    if ckpt_name and ckpt_num:
                        ckpt_filename = f"{ckpt_name}_{ckpt_num}"
                        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_filename)
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
                    raise ValueError("Load_Local_Checkpoint is True, but 'model' or 'clip' is not provided.")

            # Load and apply LoRA modules
            for lora in lora_data.get("lora", []):
                name = lora.get("name")
                model_version_id = lora.get("modelVersionId")
                strength = float(lora.get("strength", "1"))

                if not name or not model_version_id:
                    continue  # Skip invalid entries

                # Construct the LoRA filename
                lora_filename = f"{name}_{model_version_id}"

                # Load the LoRA and apply it
                if strength > 0:
                    loaded_lora = get_lora_by_filename(lora_filename, log_node=self.NAME)
                    if loaded_lora is not None:
                        model, clip = LoraLoader().load_lora(model, clip, loaded_lora, strength, strength)
                    else:
                        print(f"LoRA file '{lora_filename}' not found.")
        else:
            raise ValueError("No 'lora_config' provided.")

        return model, clip, Positive, Negative
