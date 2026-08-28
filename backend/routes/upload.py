from flask import Blueprint, jsonify, request


upload_bp = Blueprint("upload", __name__)


@upload_bp.post("/upload")
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    return jsonify({"message": "Upload endpoint ready"})
