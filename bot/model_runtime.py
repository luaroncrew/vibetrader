"""Model runtime that converts chart generation into a machine-safe signal contract."""

from __future__ import annotations

from bot.contracts import SignalContract, new_id, utc_now_iso
from data.render_charts import render_candlestick
from inference.extract_signal import extract_signal
from inference.predict import load_pipeline, predict


class ModelRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
    ):
        self.pipe = load_pipeline(checkpoint_path, device=device)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale

    def infer_signal(
        self,
        symbol: str,
        timeframe: str,
        candles,
        future_candles: int,
        rsi: float,
        macd: float,
    ) -> SignalContract:
        total_slots = len(candles) + future_candles
        price_low = float(candles["low"].min())
        price_high = float(candles["high"].max())
        prompt = (
            f"Predict next {future_candles} candles. "
            f"RSI={round(float(rsi), 1)}, MACD={round(float(macd), 2)}"
        )
        chart = render_candlestick(
            candles,
            draw_marker=False,
            total_slots=total_slots,
            price_low=price_low,
            price_high=price_high,
        )
        generated = predict(
            self.pipe,
            chart,
            prompt,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
        )
        extracted = extract_signal(
            generated,
            window_size=len(candles),
            future_candles=future_candles,
        )
        last_price = float(candles.iloc[-1]["close"])
        reasons = [
            f"pixel_green_pct={extracted.green_pct:.3f}",
            f"pixel_red_pct={extracted.red_pct:.3f}",
            f"confidence={extracted.confidence:.3f}",
        ]
        diagnostics = {
            "green_pct": extracted.green_pct,
            "red_pct": extracted.red_pct,
            "window_size": len(candles),
            "future_candles": future_candles,
        }
        return SignalContract(
            signal_id=new_id("sig"),
            timestamp=utc_now_iso(),
            symbol=symbol,
            timeframe=timeframe,
            action=extracted.action,
            confidence=extracted.confidence,
            price=last_price,
            prompt=prompt,
            source="diffusion_pixel_contract_v1",
            reasons=reasons,
            diagnostics=diagnostics,
        )
