"""Upload CT2 model to HuggingFace Hub."""
from dotenv import load_dotenv
import os
from huggingface_hub import create_repo, upload_folder

load_dotenv()

token = os.getenv("HF_TOKEN")
repo_id = "lducc/MediVoice-ct2"

print(f"Creating repo: {repo_id}")
create_repo(repo_id, repo_type="model", exist_ok=True, token=token)

print(f"Uploading model-ct2/ to {repo_id}...")
upload_folder(folder_path="./model-ct2", repo_id=repo_id, repo_type="model", token=token)

print("Done!")
