#import install
import requests
from requests.exceptions import HTTPError
import os
import re
import json

def get_base_dir():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate 3 levels up from the current directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    # Append the 'models' directory to the path
    models_dir = os.path.join(base_dir, 'models')
    return models_dir

def get_model_dirs():
    models_dir = get_base_dir()
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    return model_dirs


def download_file_with_token(fname, url, params=None, save_path='.'):
    try:
        # Send a GET request to the URL
        with requests.get(url, params=params, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses
            print(f"Downloading model from {response.url}")

            # Get filename from the content-disposition header if available
            cd = response.headers.get('content-disposition')
            filename = None
            if cd:
                filenames = re.findall('filename="(.+)"', cd)
                if len(filenames) > 0:
                    filename = filenames[0]

            # Default filename if not specified in headers
            if not filename:
                filename = url.split("/")[-1]

            #filename = fname + '_' + filename
            filename = fname
            
            # Prepare the complete file path
            file_path = os.path.join(save_path, filename)

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File already exists: {file_path}. Skipping download.")
                return True
            
            # Prepare to write to a file and track download progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            progress_interval = 5 * 1024 * 1024  # 5 MB
            next_progress = progress_interval

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    downloaded_size += len(chunk)

                    # Print progress every 5 MB
                    if downloaded_size >= next_progress:
                        print(f"Downloaded: {downloaded_size / total_size:.2%}")
                        next_progress += progress_interval
            
            print(f"File downloaded successfully: {file_path}")
            return True

    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'An error occurred: {err}')

            
def download_cai(NAME, MODEL_ID, TOKEN, LOCAL_PATH, FULL_URL):
    # Directory path where the file will be saved
    directory_path = os.path.join(get_base_dir(), LOCAL_PATH)

    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    # URL and parameters for the request
    if not FULL_URL and not MODEL_ID:
        print("Should have at least full_url or model_id for model download.")
    
    if FULL_URL:
        url = f'{FULL_URL}'
    else:
        url = f'https://civitai.com/api/download/models/{MODEL_ID}'
    params = {'token': TOKEN } if TOKEN else {}
    
    fname = NAME

    # Call the download function without checking for file existence
    download_success = download_file_with_token(fname,url, params, directory_path)
    if download_success:
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")
        

class Kw_Json_Lora_CivitAIDownloader:     
    @classmethod
  
    def INPUT_TYPES(cls):
        return {
            "required": {       
                "save_dir_lora": (get_model_dirs(),),
                "save_dir_checkpoint": (get_model_dirs(),),
                "save_dir_embedding": (get_model_dirs(),),
            },
            "optional" : {
                "Json_Lora":("STRING",{"multiline": True}),
                "ignore": ("BOOLEAN", { "default": False}),
                "ignore_down_checkpoint": ("BOOLEAN", { "default": False}),
                "model_id":  ("STRING", {"multiline": False, "default": ""}),
                "token_id": ("STRING", {"multiline": False, "default": ""}),
                "full_url": ("STRING", {"multiline": False, "default": ""})
            }
        }
        
    RETURN_TYPES = ("STRING","STRING","STRING","STRING")
    RETURN_NAMES = ('Json Lora','checkpoint','Positive','Nagative')
    FUNCTION     = "download"
    OUTPUT_NODE  = True
    CATEGORY     = "loaders"

    def download(self, model_id, token_id, save_dir_lora, ignore,ignore_down_checkpoint, full_url, Json_Lora,save_dir_checkpoint,save_dir_embedding):  
        print("Downloading")
        #print(f"\tToken: {token_id}")
        #print(f"\tFull URL: {full_url}")
        #print(f"\tSaving to: {save_dir_lora}")
        #print(f"\tSaving to: {save_dir_checkpoint}")
 
        Json_Lora_string = Json_Lora
        # Parse Json_Lora if it's a string
        try:
            Json_Lora = json.loads(Json_Lora)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Json_Lora: {e}")
            return  # Exit if JSON parsing fails 
            
        # Download LORA files

        if "lora" in Json_Lora:
            for lora_entry in Json_Lora["lora"]:
                lora_name = lora_entry["name"]
                lora_model_id = lora_entry["modelVersionId"]
                lora_filename = lora_name + '_' + lora_model_id + ".safetensors"
                print(f"\tDownloading LORA: {lora_name} (Model ID: {lora_model_id})")
                
                if not ignore:
                    download_cai(lora_filename,lora_model_id, token_id, save_dir_lora, full_url)

        if "embedding" in Json_Lora:
            for embedding_entry in Json_Lora["embedding"]:
                embedding_name = embedding_entry["name"]
                embedding_model_id = embedding_entry["modelVersionId"]
                embedding_filename = embedding_name + '_' + embedding_model_id + ".safetensors"
                print(f"\tDownloading LORA: {embedding_name} (Model ID: {embedding_model_id})")
                
                if not ignore:
                    download_cai(embedding_filename,embedding_model_id, token_id, save_dir_embedding, full_url)
                    
        # Download checkpoint files
        if "checkpoint" in Json_Lora:
            for checkpoint_entry in Json_Lora["checkpoint"]:
                checkpoint_name = checkpoint_entry["name"]
                checkpoint_model_id = checkpoint_entry["modelVersionId"]
                checkpoint_filename = checkpoint_name + '_' + checkpoint_model_id + ".safetensors"
                print(f"\tDownloading Checkpoint: {checkpoint_name} (Model ID: {checkpoint_model_id})")
                
                if not ignore_down_checkpoint:
                    download_cai(checkpoint_filename,checkpoint_model_id, token_id, save_dir_checkpoint, full_url)

        checkpoint_name_return = Json_Lora["checkpoint"][0]["name"] if Json_Lora["checkpoint"] else None
        positive = Json_Lora['positive']
        negative= Json_Lora['negative']
        return Json_Lora_string,checkpoint_name_return,positive,negative
