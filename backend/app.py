from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from routes.predict import predict_bp
from routes.results import results_bp
from routes.upload import upload_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, origins=Config.CORS_ORIGINS)

    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(results_bp, url_prefix="/api")

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok", "service": "visionarc"})

    return app


if __name__ == "__main__":
    create_app().run(
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        debug=Config.FLASK_DEBUG,
    )
