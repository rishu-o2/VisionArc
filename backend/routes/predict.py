from flask import Blueprint, jsonify


predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict():
    return jsonify({"message": "Prediction endpoint ready"})
