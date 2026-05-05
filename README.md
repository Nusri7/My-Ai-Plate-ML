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
- `POST /api/ml/adjust_meal_plan`

## Environment Secret

Set this secret in Hugging Face Space Settings:

- `OPENROUTER_API_KEY` = your OpenRouter API key

Optional settings:

- `OPENROUTER_MODEL` (default: `google/gemini-3-flash-preview`)
- `OPENROUTER_API_BASE` (default: `https://openrouter.ai/api/v1`)

## Request

`multipart/form-data`

- `image_file` (file): meal image

## Response

```json
{
  "detected_items": ["grilled chicken", "rice"],
  "confidence_scores": [95.4, 91.2],
  "calories": {
    "grilled chicken": 320,
    "rice": 210
  },
  "total_calories": 530,
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

## Adjust Meal Plan API

`POST /api/ml/adjust_meal_plan`

Request body:

```json
{
  "dailyCalorieGoal": 2000,
  "totalMealsPlanned": 4,
  "consumedMeals": [
    {"mealName": "Breakfast", "caloriesConsumed": 400},
    {"mealName": "Lunch", "caloriesConsumed": 600}
  ]
}
```

Response body:

```json
{
  "remaining_calorie_allowance": 1000,
  "remaining_meals": 2,
  "adapted_calorie_targets": [
    {
      "mealName": "Dinner",
      "suggestedFoods": ["grilled salmon", "quinoa", "broccoli"],
      "calorieTarget": 500
    },
    {
      "mealName": "Snack",
      "suggestedFoods": ["apple", "handful of almonds"],
      "calorieTarget": 500
    }
  ]
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
