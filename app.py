import os
import re
from io import BytesIO
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class GeminiDetection(BaseModel):
    food_name: str = Field(description="Detected food item name")
    confidence: float = Field(description="Confidence as a percentage from 0 to 100")
    box_2d: List[float] = Field(
        description="Bounding box in [ymin, xmin, ymax, xmax], normalized to 0..1000"
    )


class GeminiDetectionResponse(BaseModel):
    detections: List[GeminiDetection] = Field(default_factory=list)


class RecognizeFoodResponse(BaseModel):
    detected_items: List[str]
    confidence_scores: List[float]
    bounding_boxes: List[List[int]]
    query_keys: List[str]


class GenerateMealPlanRequest(BaseModel):
    age: int = Field(ge=1, le=120)
    gender: str = Field(min_length=1, max_length=32)
    height_cm: float = Field(gt=50, lt=300)
    weight_kg: float = Field(gt=20, lt=500)
    activity_level: str = Field(min_length=1, max_length=64)
    allergies: List[str] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)


class MacroSplit(BaseModel):
    carbs: float = Field(ge=0, le=100)
    proteins: float = Field(ge=0, le=100)
    fats: float = Field(ge=0, le=100)


class MealItem(BaseModel):
    food: str = Field(min_length=1, max_length=120)
    portion: str = Field(min_length=1, max_length=120)
    calories: int = Field(ge=0, le=3000)


class MealPlan(BaseModel):
    breakfast: List[MealItem] = Field(default_factory=list)
    lunch: List[MealItem] = Field(default_factory=list)
    dinner: List[MealItem] = Field(default_factory=list)
    snacks: List[MealItem] = Field(default_factory=list)


class GeminiMealPlanResponse(BaseModel):
    target_daily_calories: int = Field(ge=800, le=7000)
    macronutrient_split: MacroSplit
    recommended_plan: MealPlan


class GenerateMealPlanResponse(BaseModel):
    target_daily_calories: int
    macronutrient_split: Dict[str, float]
    recommended_plan: Dict[str, List[Dict[str, object]]]


class ChatRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=4000)
    user_context: Dict[str, Any] = Field(default_factory=dict)


class GeminiChatResponse(BaseModel):
    llm_response: str = Field(min_length=1, max_length=6000)


class ChatResponse(BaseModel):
    llm_response: str


app = FastAPI(
    title="MyAiPlate ML Microservice",
    version="1.0.0",
    description="Gemini-powered food recognition API",
)


def _to_query_key(food_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", food_name.lower()).strip("_")
    return normalized


def _norm_to_abs_box(box_2d: List[float], width: int, height: int) -> List[int]:
    if len(box_2d) != 4:
        return [0, 0, 0, 0]
    ymin, xmin, ymax, xmax = box_2d
    abs_x1 = int((max(0.0, min(1000.0, xmin)) / 1000.0) * width)
    abs_y1 = int((max(0.0, min(1000.0, ymin)) / 1000.0) * height)
    abs_x2 = int((max(0.0, min(1000.0, xmax)) / 1000.0) * width)
    abs_y2 = int((max(0.0, min(1000.0, ymax)) / 1000.0) * height)
    return [abs_x1, abs_y1, abs_x2, abs_y2]


def _build_prompt() -> str:
    return (
        "Detect all visible food items in the image. "
        "Return one detection per item. "
        "For each item, provide: "
        "food_name, confidence (0..100), "
        "and box_2d=[ymin, xmin, ymax, xmax] normalized to 0..1000. "
        "Do not include non-food objects."
    )


def _activity_factor(activity_level: str) -> float:
    key = activity_level.strip().lower()
    mapping = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9,
    }
    return mapping.get(key, 1.375)


def _estimate_calories(
    age: int, gender: str, height_cm: float, weight_kg: float, activity_level: str
) -> int:
    # Mifflin-St Jeor baseline with conservative adjustments.
    g = gender.strip().lower()
    sex_offset = 5 if g in {"male", "man", "m"} else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + sex_offset
    tdee = bmr * _activity_factor(activity_level)
    return int(round(max(1000, min(5000, tdee))))


def _build_meal_plan_prompt(
    req: GenerateMealPlanRequest, target_calories: int, macro_hint: MacroSplit
) -> str:
    allergies = ", ".join(req.allergies) if req.allergies else "none"
    prefs = ", ".join(req.dietary_preferences) if req.dietary_preferences else "none"
    return (
        "Create a safe personalized 1-day meal plan.\n"
        "Requirements:\n"
        f"- Age: {req.age}\n"
        f"- Gender: {req.gender}\n"
        f"- Height (cm): {req.height_cm}\n"
        f"- Weight (kg): {req.weight_kg}\n"
        f"- Activity level: {req.activity_level}\n"
        f"- Allergies: {allergies}\n"
        f"- Dietary preferences: {prefs}\n"
        f"- Target calories: {target_calories}\n"
        f"- Macro split target (%): carbs {macro_hint.carbs}, proteins {macro_hint.proteins}, fats {macro_hint.fats}\n"
        "Output only valid JSON matching schema. Avoid allergy ingredients completely. "
        "Provide practical foods with clear portions and calories."
    )


def _default_macro_split(activity_level: str) -> MacroSplit:
    key = activity_level.strip().lower()
    if key in {"very active", "extra active"}:
        return MacroSplit(carbs=50, proteins=25, fats=25)
    if key == "sedentary":
        return MacroSplit(carbs=40, proteins=30, fats=30)
    return MacroSplit(carbs=45, proteins=30, fats=25)


def _build_chat_prompt(payload: ChatRequest) -> str:
    return (
        "You are a nutrition assistant. "
        "Answer the user's question using the provided context. "
        "Respect strict health constraints and allergies from context. "
        "Do not suggest foods that violate constraints. "
        "If context is insufficient, state assumptions briefly.\n\n"
        f"User question:\n{payload.user_prompt}\n\n"
        f"User context object:\n{payload.user_context}"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ml/recognize_food", response_model=RecognizeFoodResponse)
async def recognize_food(image_file: UploadFile = File(...)) -> RecognizeFoodResponse:
    if image_file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                f"Use one of: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
            ),
        )

    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY environment variable.",
        )

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=image_file.content_type,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, _build_prompt()],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": GeminiDetectionResponse.model_json_schema(),
            },
        )
        parsed = GeminiDetectionResponse.model_validate_json(response.text or "{}")
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Gemini inference failed: {str(exc)}"
        ) from exc

    detected_items: List[str] = []
    confidence_scores: List[float] = []
    bounding_boxes: List[List[int]] = []
    query_keys: List[str] = []

    for det in parsed.detections:
        item = det.food_name.strip()
        if not item:
            continue
        detected_items.append(item)
        confidence_scores.append(round(max(0.0, min(100.0, det.confidence)), 2))
        bounding_boxes.append(_norm_to_abs_box(det.box_2d, width=width, height=height))
        query_keys.append(_to_query_key(item))

    return RecognizeFoodResponse(
        detected_items=detected_items,
        confidence_scores=confidence_scores,
        bounding_boxes=bounding_boxes,
        query_keys=query_keys,
    )


@app.post("/api/ml/generate_meal_plan", response_model=GenerateMealPlanResponse)
async def generate_meal_plan(
    payload: GenerateMealPlanRequest,
) -> GenerateMealPlanResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY environment variable.",
        )

    target_calories = _estimate_calories(
        age=payload.age,
        gender=payload.gender,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        activity_level=payload.activity_level,
    )
    macro_hint = _default_macro_split(payload.activity_level)

    client = genai.Client(api_key=api_key)
    prompt = _build_meal_plan_prompt(payload, target_calories, macro_hint)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": GeminiMealPlanResponse.model_json_schema(),
            },
        )
        parsed = GeminiMealPlanResponse.model_validate_json(response.text or "{}")
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Gemini inference failed: {str(exc)}"
        ) from exc

    total_macro = (
        parsed.macronutrient_split.carbs
        + parsed.macronutrient_split.proteins
        + parsed.macronutrient_split.fats
    )
    if total_macro <= 0:
        macros = {"carbs": 45.0, "proteins": 30.0, "fats": 25.0}
    else:
        macros = {
            "carbs": round((parsed.macronutrient_split.carbs / total_macro) * 100, 2),
            "proteins": round(
                (parsed.macronutrient_split.proteins / total_macro) * 100, 2
            ),
            "fats": round((parsed.macronutrient_split.fats / total_macro) * 100, 2),
        }

    plan = parsed.recommended_plan.model_dump()
    categorized_plan = {
        "Breakfast": plan.get("breakfast", []),
        "Lunch": plan.get("lunch", []),
        "Dinner": plan.get("dinner", []),
        "Snacks": plan.get("snacks", []),
    }

    return GenerateMealPlanResponse(
        target_daily_calories=parsed.target_daily_calories,
        macronutrient_split=macros,
        recommended_plan=categorized_plan,
    )


@app.post("/api/ml/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY environment variable.",
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_chat_prompt(payload)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": GeminiChatResponse.model_json_schema(),
            },
        )
        parsed = GeminiChatResponse.model_validate_json(response.text or "{}")
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Gemini inference failed: {str(exc)}"
        ) from exc

    return ChatResponse(llm_response=parsed.llm_response.strip())
