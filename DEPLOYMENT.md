# VisionArc Deployment

## Backend

The backend is a Flask service. Configure environment values using `.env` and install dependencies from `backend/requirements.txt`.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Frontend

The frontend is a React application.

```bash
cd frontend
npm install
npm start
```

## Production Notes

- Store trained model weights outside source control.
- Set strict CORS origins for deployed environments.
- Use managed object storage for uploaded images.
- Run the Flask app behind a production WSGI server.
