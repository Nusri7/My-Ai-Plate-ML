---
title: MyAiPlate ML API
emoji: 🍽️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
---

# MyAiPlate ML Microservice

This Space exposes:

- `POST /api/ml/recognize_food`
- `POST /api/ml/generate_meal_plan`
- `POST /api/ml/chat`

## Environment Secret

Set this secret in Hugging Face Space Settings:

- `GEMINI_API_KEY` = your Google Gemini API key

Optional reliability variables:

- `GEMINI_MODELS` (default: `gemini-2.5-flash,gemini-2.0-flash`)
- `GEMINI_MAX_RETRIES` (default: `3`)
- `GEMINI_RETRY_BASE_DELAY_SEC` (default: `1.5`)

## Request

`multipart/form-data`

- `image_file` (file): meal image

## Response

```json
{
  "detected_items": ["grilled chicken", "rice"],
  "confidence_scores": [95.4, 91.2],
  "bounding_boxes": [[32, 114, 420, 620], [430, 100, 760, 700]],
  "query_keys": ["grilled_chicken", "rice"]
}
```

## Meal Plan API

`POST /api/ml/generate_meal_plan`

Request body:

```json
{
  "age": 26,
  "gender": "male",
  "height_cm": 175,
  "weight_kg": 72,
  "activity_level": "moderately active",
  "allergies": ["peanut"],
  "dietary_preferences": ["high protein"]
}
```

## Chat API

`POST /api/ml/chat`

Request body:

```json
{
  "user_prompt": "I skipped lunch. What can I eat for dinner within my plan?",
  "user_context": {
    "diet_plan": {"target_calories": 2200},
    "logged_meals_today": [{"meal": "breakfast", "calories": 450}],
    "strict_health_constraints": ["no shellfish", "low sodium"]
  }
}
```

Response body:

```json
{
  "llm_response": "You can have grilled chicken with quinoa and steamed vegetables, keep sodium low by avoiding packaged sauces, and stay within your remaining calorie budget."
}
```

Response body:

```json
{
  "target_daily_calories": 2430,
  "macronutrient_split": {
    "carbs": 45.0,
    "proteins": 30.0,
    "fats": 25.0
  },
  "recommended_plan": {
    "Breakfast": [{"food": "Greek yogurt", "portion": "200g", "calories": 180}],
    "Lunch": [{"food": "Grilled chicken bowl", "portion": "1 bowl", "calories": 650}],
    "Dinner": [{"food": "Salmon with vegetables", "portion": "1 plate", "calories": 700}],
    "Snacks": [{"food": "Apple", "portion": "1 medium", "calories": 95}]
  }
}
```

## Health Check

- `GET /health`
