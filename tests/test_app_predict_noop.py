import importlib
import json
import sys
import types

from PIL import Image


def _install_stub_modules():
    predict_module = types.ModuleType("inference.predict")

    def load_pipeline(checkpoint_path, device="auto"):
        return {"checkpoint": checkpoint_path, "device": device}

    def predict(pipe, image, prompt, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.0):
        return image.convert("RGB").resize((256, 256))

    predict_module.load_pipeline = load_pipeline
    predict_module.predict = predict
    sys.modules["inference.predict"] = predict_module

    signal_module = types.ModuleType("inference.extract_signal")

    class Signal:
        def __init__(self, action, confidence, green_pct, red_pct):
            self.action = action
            self.confidence = confidence
            self.green_pct = green_pct
            self.red_pct = red_pct

    def extract_signal(image):
        return Signal("HOLD", 0.99, 0.5, 0.5)

    signal_module.Signal = Signal
    signal_module.extract_signal = extract_signal
    sys.modules["inference.extract_signal"] = signal_module

    mistral_module = types.ModuleType("inference.extract_signal_mistral")

    class MistralSignal:
        def __init__(self, action, confidence, reasoning):
            self.action = action
            self.confidence = confidence
            self.reasoning = reasoning

    def extract_signal_mistral(input_image, generated_image):
        return MistralSignal("HOLD", 0.75, "stubbed")

    mistral_module.Signal = MistralSignal
    mistral_module.extract_signal_mistral = extract_signal_mistral
    sys.modules["inference.extract_signal_mistral"] = mistral_module


def test_predict_route_noop_flow(tmp_path, monkeypatch):
    _install_stub_modules()

    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    input_dir.mkdir()
    target_dir.mkdir()

    image_id = "case-001"
    metadata = [
        {
            "id": image_id,
            "prompt": "Predict next 4 candles.",
            "signal": "HOLD",
            "pct_change": 0.0,
        }
    ]
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    Image.new("RGB", (32, 32), (12, 34, 56)).save(input_dir / f"{image_id}.png")
    Image.new("RGB", (32, 32), (65, 43, 21)).save(target_dir / f"{image_id}.png")

    app_module = importlib.import_module("app")

    monkeypatch.setattr(app_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(app_module, "INPUT_DIR", input_dir)
    monkeypatch.setattr(app_module, "TARGET_DIR", target_dir)
    monkeypatch.setattr(app_module, "pipe", {"stub": True})

    client = app_module.app.test_client()
    response = client.post("/api/predict", json={"image_id": image_id})

    assert response.status_code == 200

    payload = response.get_json()
    assert payload["image_id"] == image_id
    assert payload["prompt"] == "Predict next 4 candles."
    assert payload["signal"]["action"] == "HOLD"
    assert payload["signal"]["confidence"] == 0.99
    assert payload["ground_truth"] == "HOLD"
    assert payload["pct_change"] == 0.0
    assert payload["use_mistral"] is False
    assert payload["input_image"]
    assert payload["generated_image"]
    assert payload["target_image"]
