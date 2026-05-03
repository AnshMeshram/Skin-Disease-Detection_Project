# SkinAI — Frontend

## File Structure

```
frontend/
├── index.html              ← Main single-page app
├── css/
│   └── styles.css          ← All styles (CSS Variables, layout, components)
├── js/
│   └── app.js              ← All logic (API calls, animations, state)
├── assets/
│   └── hero-bg.jpg         ← (ADD THIS) Medical flatlay background image
└── FIGMA_PROMPT.md         ← Full Figma design specification
```

## Setup

### 1. Add the hero background image
Place any medical background image at `assets/hero-bg.jpg`.

Then in `css/styles.css`, uncomment the hero-bg lines:
```css
/* Find .hero-bg-gradient and change to: */
background-image: url('../assets/hero-bg.jpg');
background-size: cover;
background-position: center;
filter: blur(3px) brightness(0.25);
transform: scale(1.05);
```

### 2. Set your API base URL
In `js/app.js`, update line 17:
```js
const API_BASE = 'http://localhost:8000';   // ← your FastAPI server
```

### 3. Backend API endpoints required

| Method | Endpoint      | Request                    | Response |
|--------|--------------|----------------------------|----------|
| GET    | /health      | —                          | `{ "status": "ok" }` |
| POST   | /predict     | `multipart/form-data: file` | `{ disease, confidence, model, ensemble_vote }` |
| POST   | /preprocess  | `multipart/form-data: file` | `{ steps: [{ name, image_b64 }] }` |

### 4. Run
Open `index.html` directly in a browser, or serve with:
```bash
python -m http.server 3000
# → open http://localhost:3000
```

Or use Live Server in VS Code.

## Features
- Upload or drag-and-drop dermoscopy image
- Animated 6-step preprocessing pipeline visualization
- Disease prediction with confidence score bar
- Ensemble model agreement display
- Model architecture tabs (EfficientNet-B3 / InceptionV3 / ConvNeXt Tiny)
- 9-class disease grid
- Demo mode if backend is offline
- Fully responsive (mobile + desktop)

## API Response Format

### `/predict` response
```json
{
  "disease": "Melanoma",
  "confidence": 90.46,
  "model": "EfficientNet-B3 (Ensemble)",
  "ensemble_vote": "3/3 models agree",
  "class_probabilities": {
    "Melanoma": 0.9046,
    "Melanocytic nevi": 0.0412,
    ...
  }
}
```

### `/preprocess` response
```json
{
  "steps": [
    { "name": "White Balance", "image_b64": "<base64 jpeg>" },
    { "name": "Original",      "image_b64": "<base64 jpeg>" },
    { "name": "Hair Removed",  "image_b64": "<base64 jpeg>" },
    { "name": "ITA CLAHE",     "image_b64": "<base64 jpeg>" },
    { "name": "Skin Mask",     "image_b64": "<base64 jpeg>" },
    { "name": "Lesion Region", "image_b64": "<base64 jpeg>" }
  ]
}
```
