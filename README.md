# Kw_Json_Lora_CivitAIDownloader

The **Kw_Json_Lora_CivitAIDownloader** is a powerful tool that simplifies downloading large numbers of models from CivitAI, especially useful for managing extensive sets of LORA and checkpoint models. This tool uses a JSON configuration to define models for download, enabling you to specify LORA and checkpoint models in bulk and save them to designated directories. With token-based authentication, the downloader ensures secure access and automates the process, making it efficient for users handling large model libraries.

## Features
- **Directory Selection**: Choose directories for saving LORA and checkpoint models.
- **JSON Input for Model Data**: Configure models for download through a JSON structure that defines LORA and checkpoint details.
- **Token Authentication**: Secure downloads with a CivitAI token.

## Requirements
- **Python 3.x**
- **Modules**: `requests`, `os`, `re`, `json`

## How to Use

Set Up Directories: The script will automatically navigate to the models directory, three levels up from the script location, where it will look for directories specified in the configuration.
	Ensure that your LORA and checkpoint directories are subdirectories of the models directory, as the downloader will save files based on this structure.

Prepare JSON Input (Example Below): Configure the models to download by preparing a JSON structure. This JSON includes the lora and checkpoint arrays with name and modelVersionId fields.

Set Token and Model URL/ID: For secure download, use your CivitAI token.

JSON Configuration Example

## Create a JSON input for LORA and checkpoint models like this:

	json

	{
	  "lora": [
		{
		  "name": "Detailed_anime_style_-_SDXL_pony",
		  "modelVersionId": "449290"
		},
		{
		  "name": "Comic_Book_Page_style_XL_F1D",
		  "modelVersionId": "514793"
		}
	  ],
	  "checkpoint": [
		{
		  "name": "Pony_Diffusion_V6_XL",
		  "modelVersionId": "290640"
		}
	  ],
	  "positive": "(((score_9, score_8_up, score_7_up, score_6_up)))",
	  "negative": ""
	}

## Save this JSON data in a file or copy it into the "Json_Lora" input field in the downloader.

## Required and Optional Inputs

save_dir_lora: Directory where LORA files will be saved. This should be a directory inside the models directory.
save_dir_checkpoint: Directory for checkpoint files, also inside models.
Json_Lora: JSON string containing LORA and checkpoint model configurations.
token_id: Your CivitAI token for authenticated access.
