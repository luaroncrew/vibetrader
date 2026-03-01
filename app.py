"""Flask backend for VibeTrader interactive prediction UI."""

import base64
import io
import json
import random
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from inference.predict import load_pipeline, predict
from inference.extract_signal import extract_signal
from inference.extract_signal_mistral import extract_signal_mistral

app = Flask(__name__)

METADATA_PATH = Path("data/rendered/metadata.json")
INPUT_DIR = Path("data/rendered/input")
TARGET_DIR = Path("data/rendered/target")
CHECKPOINT_DIR = "checkpoints"

# Global pipeline — loaded once at startup
pipe = None
# 5 random IDs from top 50 — selected at startup
showcase_ids = []


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend-v3", "index.html")


@app.route("/api/showcase")
def showcase():
    """Return metadata for the 5 showcase charts (id only, no spoilers)."""
    with open(METADATA_PATH) as f:
        all_data = json.load(f)

    lookup = {d["id"]: d for d in all_data}
    items = []
    for sid in showcase_ids:
        meta = lookup.get(sid)
        if meta:
            items.append({"id": meta["id"]})

    return jsonify({"items": items})


@app.route("/api/images/input/<image_id>.png")
def serve_input_image(image_id):
    return send_from_directory(INPUT_DIR, f"{image_id}.png")


@app.route("/api/images/target/<image_id>.png")
def serve_target_image(image_id):
    return send_from_directory(TARGET_DIR, f"{image_id}.png")


@app.route("/api/predict", methods=["POST"])
def run_prediction():
    data = request.get_json()
    if not data or "image_id" not in data:
        return jsonify({"error": "image_id is required"}), 400

    image_id = data["image_id"]
    use_mistral = data.get("use_mistral", False)

    image_path = INPUT_DIR / f"{image_id}.png"
    if not image_path.exists():
        return jsonify({"error": f"Image {image_id} not found"}), 404

    target_path = TARGET_DIR / f"{image_id}.png"

    # Look up prompt from metadata
    with open(METADATA_PATH) as f:
        all_meta = json.load(f)
    meta = next((m for m in all_meta if m["id"] == image_id), None)
    prompt = meta["prompt"] if meta else "Predict next 4 candles."

    input_image = Image.open(image_path).convert("RGB")

    # Run diffusion inference
    generated_image = predict(pipe, input_image, prompt)

    # Extract signal
    if use_mistral:
        signal = extract_signal_mistral(input_image, generated_image)
        signal_data = {
            "action": signal.action,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
        }
    else:
        signal = extract_signal(generated_image)
        signal_data = {
            "action": signal.action,
            "confidence": signal.confidence,
            "green_pct": signal.green_pct,
            "red_pct": signal.red_pct,
        }

    response = {
        "image_id": image_id,
        "prompt": prompt,
        "input_image": image_to_base64(input_image.resize((256, 256))),
        "generated_image": image_to_base64(generated_image),
        "signal": signal_data,
        "use_mistral": use_mistral,
        "ground_truth": meta["signal"] if meta else None,
        "pct_change": meta["pct_change"] if meta else None,
    }

    # Include target image if it exists
    if target_path.exists():
        target_image = Image.open(target_path).convert("RGB")
        response["target_image"] = image_to_base64(target_image.resize((256, 256)))

    return jsonify(response)


def pick_showcase_ids():
    """Pick 5 random IDs from the top 50 trades by strategy return."""
    with open(METADATA_PATH) as f:
        all_data = json.load(f)

    # Compute strategy return for each entry
    scored = []
    for d in all_data:
        pc = d["pct_change"]
        sig = d["signal"]
        if sig == "BUY":
            ret = pc
        elif sig == "SELL":
            ret = -pc
        else:
            ret = 0
        scored.append((ret, d["id"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_50 = [sid for _, sid in scored[:50]]
    return random.sample(top_50, 5)


if __name__ == "__main__":
    showcase_ids = pick_showcase_ids()
    print(f"Showcase IDs: {showcase_ids}")

    print("Loading diffusion pipeline...")
    pipe = load_pipeline(CHECKPOINT_DIR)

    # Warm up with a dummy prediction
    print("Warming up pipeline...")
    dummy = Image.new("RGB", (256, 256), (0, 0, 0))
    predict(pipe, dummy, "warmup")
    print("Pipeline ready.")

    app.run(host="0.0.0.0", port=8923, debug=False)
