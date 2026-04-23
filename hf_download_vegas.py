import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

log = open("/media/skr/storage/autoresearch/CarPlanner_Implementation/hf_download.log", "w", buffering=1)

def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    log.write(line + "\n")
    print(line, flush=True)

log_msg("Starting download: sangramrout/CarPlanner / nuplan-v1.1_train_vegas_5.zip")
t0 = time.time()

path = hf_hub_download(
    repo_id="sangramrout/CarPlanner",
    filename="nuplan-v1.1_train_vegas_5.zip",
    repo_type="dataset",
    local_dir="/media/skr/storage/autoresearch/CarPlanner_Implementation/downloads",
)

elapsed = time.time() - t0
size_gb = os.path.getsize(path) / 1e9
log_msg(f"Done: {path}")
log_msg(f"Size: {size_gb:.2f} GB in {elapsed:.0f}s ({size_gb*1024/elapsed:.1f} MB/s)")
log.close()
