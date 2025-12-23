import os, sys, time, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download   #Easy way to download hugging face models 
import shutil, librosa

# CONFIG PATHS
REPO_ID = "leduckhai/MultiMed-ST"
SUBFOLDER = "asr/whisper-small-vietnamese"
LOCAL_DIR = "./whisper_model_local"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = snapshot_download(repo_id = REPO_ID,
                            allow_patterns = f"{SUBFOLDER}/*",
                            ignore_patterns = ["*runs/*", "*.tfevents*"],  #github has some log files we dont need
                            local_dir = LOCAL_DIR)
                            # local_dir_use_symlinks = False)

    config_path = os.path.join(root, SUBFOLDER)
    weight_path = os.path.join(config_path, "checkpoint-5000")

    #Download the config and weight files
    if os.path.exists(config_path) and os.path.exists(weight_path):
        for filename in os.listdir(config_path):
            if filename.endswith(".json") or filename.endswith(".txt"):
                src = os.path.join(config_path, filename)
                dst = os.path.join(config_path, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    processor = WhisperProcessor.from_pretrained(config_path)
    model = WhisperForConditionalGeneration.from_pretrained(weight_path, 
                                                                use_safetensors = True).to(device)
    model.eval()


    return model, processor, device


def transcribe(audio_path, model, processor, device):
    #Model is trained on 16000khz audio so the audio must match it
    speech, _ = librosa.load(audio_path, sr = 16000)
    
    tensors = processor(speech, sampling_rate = 16000, return_tensors = "pt") #  <--- Converts audio to tensors
    features = tensors.input_features.to(device)

    with torch.no_grad():
        ids = model.generate(features, language = "vi", task = "transcribe") # Makes tokens ig for decoding later

    text = processor.batch_decode(ids, skip_special_tokens=True)[0] #special token is html tags kinda -> trash token dont need it
    return text

def main():
    if len(sys.argv) < 2:
        print("You must provide the path to an audio file.")
        print("USAGE: python main.py sample.wav")
        sys.exit(1)

        
    file_name = sys.argv[1] # Use the dragged file

    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found")
        return

    model, processor, device = load_model() 
    result = transcribe(file_name, model, processor, device)

    print('-' * 40)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()