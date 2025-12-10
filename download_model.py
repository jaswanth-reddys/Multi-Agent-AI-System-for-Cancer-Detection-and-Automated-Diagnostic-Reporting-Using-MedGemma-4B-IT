import os
from huggingface_hub import snapshot_download

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

snapshot_download(
    repo_id='google/medgemma-4b-it',
    local_dir=r'c:\NEWPROJECT\medgemma-4b-it',
    local_dir_use_symlinks=False,
    token='YOUR_HUGGINGFACE_HUB_TOKEN',
    ignore_patterns=['*.h5']
)

print("Model downloaded successfully!")
    