from flask import Blueprint, jsonify


results_bp = Blueprint("results", __name__)


@results_bp.get("/results/<result_id>")
def get_result(result_id):
    return jsonify({"result_id": result_id, "message": "Results endpoint ready"})
