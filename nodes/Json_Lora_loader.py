import json
from nodes import LoraLoader
from .json_lora_loader.power_prompt_utils import get_lora_by_filename
from .json_lora_loader.log import log_node_info, log_node_warn
import folder_paths
import comfy.sd
import os

def get_checkpoint_by_filename(file_path, checkpoint_paths=None, log_node=None):
    """Returns a checkpoint file path by filename, looking for exact paths and then fuzzier matching."""
    checkpoint_paths = checkpoint_paths if checkpoint_paths is not None else folder_paths.get_filename_list('checkpoints')

    if file_path in checkpoint_paths:
        return file_path

    # Remove extensions for comparison
    checkpoint_paths_no_ext = [os.path.splitext(x)[0] for x in checkpoint_paths]

    # Exact match without extension
    if file_path in checkpoint_paths_no_ext:
        found = checkpoint_paths[checkpoint_paths_no_ext.index(file_path)]
        return found

    # Force input without extension and compare
    file_path_no_ext = os.path.splitext(file_path)[0]
    if file_path_no_ext in checkpoint_paths_no_ext:
        found = checkpoint_paths[checkpoint_paths_no_ext.index(file_path_no_ext)]
        return found

    # Compare basenames
    checkpoint_basenames = [os.path.basename(x) for x in checkpoint_paths]
    if file_path in checkpoint_basenames:
        found = checkpoint_paths[checkpoint_basenames.index(file_path)]
        if log_node is not None:
            log_node_info(log_node, f"Matched checkpoint input '{file_path}' to '{found}'.")
        return found

    # Force input to basename and compare
    file_basename = os.path.basename(file_path)
    if file_basename in checkpoint_basenames:
        found = checkpoint_paths[checkpoint_basenames.index(file_basename)]
        if log_node is not None:
            log_node_info(log_node, f"Matched checkpoint input '{file_path}' to '{found}'.")
        return found

    # Compare basenames without extensions
    checkpoint_basenames_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in checkpoint_paths]
    if file_path in checkpoint_basenames_no_ext:
        found = checkpoint_paths[checkpoint_basenames_no_ext.index(file_path)]
        if log_node is not None:
            log_node_info(log_node, f"Matched checkpoint input '{file_path}' to '{found}'.")
        return found

    # Force input to basename without extension and compare
    file_basename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_basename_no_ext in checkpoint_basenames_no_ext:
        found = checkpoint_paths[checkpoint_basenames_no_ext.index(file_basename_no_ext)]
        if log_node is not None:
            log_node_info(log_node, f"Matched checkpoint input '{file_path}' to '{found}'.")
        return found

    # Fuzzy matching
    for index, checkpoint_path in enumerate(checkpoint_paths):
        if file_path in checkpoint_path:
            found = checkpoint_paths[index]
            if log_node is not None:
                log_node_warn(log_node, f"Fuzzy-matched checkpoint input '{file_path}' to '{found}'.")
            return found

    if log_node is not None:
        log_node_warn(log_node, f"Checkpoint '{file_path}' not found, skipping.")

    return None

class Kw_JsonLoraLoader:
    """A node to load multiple LoRA modules from a JSON configuration."""

    NAME = "Kw_JsonLoraLoader"
    CATEGORY = "loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_config": ("STRING",),  # User-provided JSON configuration for LoRA modules
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
