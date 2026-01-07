from huggingface_hub import snapshot_download

print("Downloading Qwen2.5-0.5B locally...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B",
    local_dir="./Qwen2.5-0.5B-Local",
    local_dir_use_symlinks=False,  # Important: Actual files, not links
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional cleanup
)
print("Download complete. Model is in ./Qwen2.5-0.5B-Local")