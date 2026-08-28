# VisionArc Architecture

VisionArc is organized as a React frontend and Flask API backend with a dedicated machine learning layer for preprocessing, inference, and interpretability.

## Backend

- `app.py` creates the Flask application and registers API routes.
- `routes/` contains upload, prediction, and result endpoints.
- `ml_model/` contains model loading, image preprocessing, and Grad-CAM utilities.

## Frontend

- `src/App.jsx` is the application shell.
- `src/components/` contains user-facing UI components.
- `src/components/3D/` is reserved for Three.js scenes and visual systems.
- `src/animations/` is reserved for motion presets and animation helpers.

## Data Flow

1. A fundus image is uploaded through the frontend.
2. The backend stores or validates the image.
3. The prediction route preprocesses the image and runs model inference.
4. Grad-CAM can generate interpretability overlays.
5. Results are returned to the frontend for display.
