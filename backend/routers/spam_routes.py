from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from backend.schemas.input_schema import SpamInput
from backend.schemas.output_schema import SpamPredictionResponse, ErrorResponse
from backend.schemas.history_schema import HistoryItem, HistoryUpdate
from backend.utils.history_store import HISTORY_STORE
from backend.utils.preprocessing import preprocess_text
from backend.utils.postprocessing import build_response
from backend.utils.error_handler import ModelNotLoadedError
from backend.utils.model_service import MODEL_SERVICE

logger = logging.getLogger("spam_app")

router = APIRouter(tags=["spam"])


@router.post(
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"/predict_spam",
    response_model=SpamPredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_spam(payload: SpamInput) -> Dict[str, Any]:
    if not MODEL_SERVICE.loaded():
        raise ModelNotLoadedError()

    text = preprocess_text(payload.text)

    try:
        if MODEL_SERVICE.vectorizer is not None and MODEL_SERVICE.model is not None:
            X = MODEL_SERVICE.vectorizer.transform([text])
            preds = MODEL_SERVICE.model.predict(X)
            probs = MODEL_SERVICE.model.predict_proba(X)[0]
            classes = getattr(MODEL_SERVICE.model, "classes_", None)
        elif MODEL_SERVICE.pipeline is not None:
            preds = MODEL_SERVICE.pipeline.predict([text])
            probs = MODEL_SERVICE.pipeline.predict_proba([text])[0]
            classes = getattr(MODEL_SERVICE.pipeline, "classes_", None)
        else:
            raise ModelNotLoadedError()

        pred_raw = preds[0]

        def to_label(val: Any) -> str:
            if isinstance(val, (int, float)):
                return "spam" if int(val) == 1 else "ham"
            s = str(val).lower()
            if s in {"1", "spam", "true", "yes"}:
                return "spam"
            return "ham"

        pred_label = to_label(pred_raw)

        class_prob_map: Dict[str, float] = {}
        if classes is not None:
            for cls, p in zip(list(classes), list(probs)):
                key = to_label(cls)
                class_prob_map[key] = float(p)
        else:
            if len(probs) == 2:
                class_prob_map = {"ham": float(probs[0]), "spam": float(probs[1])}
            else:
                class_prob_map = {pred_label: float(probs.max() if hasattr(probs, "max") else probs)}

        result = build_response(pred_label, class_prob_map)

        # Persist to history and include id in response
        try:
            item_id = HISTORY_STORE.add(
                text=payload.text,
                prediction=pred_label,
                confidence=float(result.get("confidence", 0.0)),
                timestamp=str(result.get("timestamp")),
            )
            result["id"] = item_id
        except Exception:
            # If persistence fails, continue to return prediction
            pass

        try:
            logger.info("predict_spam input='%s' result=%s", payload.text.replace("\n", " ")[:500], result)
        except Exception:
            pass

        return result
    except ModelNotLoadedError:
        raise
    except Exception as exc:
        logging.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.get("/history", response_model=list[HistoryItem])
async def list_history() -> list[dict[str, Any]]:
    rows = HISTORY_STORE.list()
    # Convert timestamp strings to ISO strings; Pydantic will parse
    return rows


@router.put("/history/{item_id}", response_model=HistoryItem)
async def update_history(item_id: str, payload: HistoryUpdate) -> dict[str, Any]:
    updated = HISTORY_STORE.update(
        item_id,
        prediction=payload.prediction,
        confidence=payload.confidence,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="History item not found")
    return updated  # type: ignore[return-value]


@router.delete("/history/{item_id}", status_code=204)
async def delete_history(item_id: str) -> None:
    ok = HISTORY_STORE.delete(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="History item not found")
