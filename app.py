import asyncio
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from fastapi import FastAPI, File, HTTPException, UploadFile
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
    calories: int = Field(ge=0, le=3000, description="Estimated calories for this food item")


class GeminiDetectionResponse(BaseModel):
    detections: List[GeminiDetection] = Field(default_factory=list)


class RecognizeFoodResponse(BaseModel):
    detected_items: List[str]
    confidence_scores: List[float]
    calories: Dict[str, int]
    total_calories: int
    query_keys: List[str]


class ConsumedMeal(BaseModel):
    mealName: str = Field(min_length=1, max_length=64)
    caloriesConsumed: int = Field(ge=0)


class AdjustMealPlanRequest(BaseModel):
    dailyCalorieGoal: int = Field(gt=0)
    totalMealsPlanned: int = Field(gt=0)
    consumedMeals: List[ConsumedMeal] = Field(default_factory=list)


class AdaptedMeal(BaseModel):
    mealName: str
    suggestedFoods: List[str]
    calorieTarget: int


class AdjustMealPlanResponse(BaseModel):
    remaining_calorie_allowance: int
    remaining_meals: int
    adapted_calorie_targets: List[AdaptedMeal]
    warning: Optional[str] = None
    message: Optional[str] = None


class GeminiMealSuggestion(BaseModel):
    mealName: str
    suggestedFoods: List[str]
    calorieTarget: int


class GeminiMealSuggestionsResponse(BaseModel):
    suggestions: List[GeminiMealSuggestion] = Field(default_factory=list)


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


T = TypeVar("T", bound=BaseModel)


app = FastAPI(
    title="MyAiPlate ML Microservice",
    version="1.0.0",
    description="OpenRouter-powered food recognition API",
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
        "food_name, confidence (0..100), calories. "
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


def _get_meal_names(total_meals: int) -> List[str]:
    if total_meals == 3:
        return ["Breakfast", "Lunch", "Dinner"]
    elif total_meals == 4:
        return ["Breakfast", "Lunch", "Dinner", "Snack"]
    elif total_meals == 5:
        return ["Breakfast", "Lunch", "Dinner", "Snack", "Dessert"]
    else:
        return [f"Meal {i+1}" for i in range(1, total_meals + 1)]


def _build_chat_prompt(payload: ChatRequest) -> str:
    return (
        "You are a nutrition assistant. "
        "Answer the user's question using the provided context. "
        "Respect strict health constraints and allergies from context. "
        "Do not suggest foods that violate constraints. "
        "If context is insufficient, state assumptions briefly. "
        "Output only valid JSON with a single field named llm_response.\n\n"
        f"User question:\n{payload.user_prompt}\n\n"
        f"User context object:\n{payload.user_context}"
    )


def _openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")


def _openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def _get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server is missing OPENROUTER_API_KEY environment variable.",
        )
    return OpenAI(api_key=api_key, base_url=_openrouter_base_url())


def _extract_json_from_text(text: str) -> Any:
    if not text or not isinstance(text, str):
        raise ValueError(f"Expected non-empty string, got: {repr(text)}")
    
    text = text.strip()
    if not text:
        raise ValueError("Response text is empty")
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")
        raise ValueError(f"No JSON object found in response: {text[:200]}")


async def _generate_structured_json(
    contents: str,
    schema_model: Type[T],
) -> T:
    client = _get_openrouter_client()
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON generator. Output ONLY valid JSON, nothing else. No markdown, no explanations.",
        },
        {"role": "user", "content": contents},
    ]

    def _call_openrouter() -> str:
        try:
            response = client.chat.completions.create(
                model=_openrouter_model(),
                messages=messages,
                temperature=0.0,
                max_tokens=1600,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("OpenRouter returned empty response")
            return content
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenRouter API error: {str(e)}")

    raw_text = await asyncio.to_thread(_call_openrouter)
    try:
        parsed_json = _extract_json_from_text(raw_text)
        return schema_model.model_validate(parsed_json)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse response: {str(e)}")


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

    # Convert image to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_media_type = image_file.content_type or "image/jpeg"

    client = _get_openrouter_client()
    
    # --- FIXED MESSAGES PAYLOAD START ---
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON generator. Output ONLY valid JSON, nothing else. "
                "Your output must be a JSON object with a single key 'detections' containing a list of objects. "
                "Each object must have 'food_name' (string), 'confidence' (float 0-100), and 'calories' (integer)."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{base64_image}"
                    },
                },
                {"type": "text", "text": _build_prompt()},
            ],
        }
    ]
    # --- FIXED MESSAGES PAYLOAD END ---

    def _call_openrouter_vision() -> str:
        try:
            response = client.chat.completions.create(
                model=_openrouter_model(),
                messages=messages,
                temperature=0.0,
                max_tokens=1600,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("OpenRouter returned empty response for image")
            return content
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenRouter image API error: {str(e)}")

    raw_text = await asyncio.to_thread(_call_openrouter_vision)
    try:
        parsed_json = _extract_json_from_text(raw_text)
        parsed = GeminiDetectionResponse.model_validate(parsed_json)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse image response: {str(e)}")

    detected_items: List[str] = []
    confidence_scores: List[float] = []
    calories: Dict[str, int] = {}
    query_keys: List[str] = []

    for det in parsed.detections:
        item = det.food_name.strip()
        if not item:
            continue
        detected_items.append(item)
        confidence_scores.append(round(max(0.0, min(100.0, det.confidence)), 2))
        calories[item] = int(round(max(0, det.calories)))
        query_keys.append(_to_query_key(item))

    total_calories = sum(calories.values())

    return RecognizeFoodResponse(
        detected_items=detected_items,
        confidence_scores=confidence_scores,
        calories=calories,
        total_calories=total_calories,
        query_keys=query_keys,
    )


@app.post("/api/ml/generate_meal_plan", response_model=GenerateMealPlanResponse)
async def generate_meal_plan(
    payload: GenerateMealPlanRequest,
) -> GenerateMealPlanResponse:
    target_calories = _estimate_calories(
        age=payload.age,
        gender=payload.gender,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        activity_level=payload.activity_level,
    )
    macro_hint = _default_macro_split(payload.activity_level)

    prompt = _build_meal_plan_prompt(payload, target_calories, macro_hint)

    parsed = await _generate_structured_json(
        contents=prompt,
        schema_model=GeminiMealPlanResponse,
    )

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
    prompt = _build_chat_prompt(payload)

    parsed = await _generate_structured_json(
        contents=prompt,
        schema_model=GeminiChatResponse,
    )

    return ChatResponse(llm_response=parsed.llm_response.strip())


@app.post("/api/ml/adjust_meal_plan", response_model=AdjustMealPlanResponse)
async def adjust_meal_plan(payload: AdjustMealPlanRequest) -> AdjustMealPlanResponse:
    # Input validation
    if payload.dailyCalorieGoal <= 0 or payload.totalMealsPlanned <= 0:
        raise HTTPException(status_code=400, detail="dailyCalorieGoal and totalMealsPlanned must be positive numbers.")
    for meal in payload.consumedMeals:
        if meal.caloriesConsumed < 0:
            raise HTTPException(status_code=400, detail="caloriesConsumed cannot be negative.")

    total_consumed = sum(meal.caloriesConsumed for meal in payload.consumedMeals)
    remaining_calories = max(0, payload.dailyCalorieGoal - total_consumed)
    consumed_count = len(payload.consumedMeals)
    remaining_meals_count = payload.totalMealsPlanned - consumed_count

    if remaining_meals_count <= 0:
        return AdjustMealPlanResponse(
            remaining_calorie_allowance=remaining_calories,
            remaining_meals=0,
            adapted_calorie_targets=[],
            message="All planned meals are completed."
        )

    all_meal_names = _get_meal_names(payload.totalMealsPlanned)
    consumed_names = {meal.mealName for meal in payload.consumedMeals}
    remaining_names = [name for name in all_meal_names if name not in consumed_names][:remaining_meals_count]

    if remaining_calories == 0 and total_consumed > payload.dailyCalorieGoal:
        adapted_targets = [
            AdaptedMeal(mealName=name, suggestedFoods=[], calorieTarget=0)
            for name in remaining_names
        ]
        warning = "Calorie goal exceeded."
        return AdjustMealPlanResponse(
            remaining_calorie_allowance=remaining_calories,
            remaining_meals=remaining_meals_count,
            adapted_calorie_targets=adapted_targets,
            warning=warning
        )

    per_meal_calories = remaining_calories // remaining_meals_count

    meals_data = [{"mealName": name, "calorieTarget": per_meal_calories} for name in remaining_names]
    prompt = (
        "Suggest balanced, healthy foods for the following meals, each with the specified calorie target. "
        "Consider meal times and nutritional balance. "
        "Output a JSON object with 'suggestions' as a list of objects, each with mealName, suggestedFoods (list of food items), calorieTarget. "
        f"Meals: {meals_data}"
    )

    parsed = await _generate_structured_json(
        contents=prompt,
        schema_model=GeminiMealSuggestionsResponse,
    )

    adapted_targets = [
        AdaptedMeal(**suggestion.model_dump()) for suggestion in parsed.suggestions
    ]

    return AdjustMealPlanResponse(
        remaining_calorie_allowance=remaining_calories,
        remaining_meals=remaining_meals_count,
        adapted_calorie_targets=adapted_targets
    )