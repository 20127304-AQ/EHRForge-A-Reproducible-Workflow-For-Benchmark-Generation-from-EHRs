"""Read-only: sha256 + size of the frozen inputs stored on the Step-3 volume (no copies made)."""
import os  # release adaptation
import modal
app = modal.App("ehrforge-step3-verify")
vol = modal.Volume.from_name(os.environ.get("EHRFORGE_STEP3_VOLUME", "ehrforge-step3-v13"))

@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/vol": vol}, cpu=2, timeout=900)
def verify():
    import hashlib, os
    out = {}
    for p in ["data/corpus.json", "data/dataset.csv", "shared/canonical/protocol_v1.2.yaml", "shared/canonical/protocol_v1.3.yaml"]:
        h = hashlib.sha256()
        with open("/vol/" + p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        out[p] = (h.hexdigest(), os.path.getsize("/vol/" + p))
    return out

@app.local_entrypoint()
def main():
    import json
    r = verify.remote()
    print(json.dumps(r, indent=1))
    json.dump(r, open("results/volume_inputs_verify.json", "w"), indent=1)
