from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from .extensions import db
from .localization.engine import LocalizationEngine
from .routes.api import api
from .services.copilot import CopilotService
from .services.scenes import SceneManager


def create_app(test_config=None):
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    app = Flask(__name__)
    from .config import Config

    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)
    scene_manager = SceneManager(app.config)
    localization_engine = LocalizationEngine(app.config)
    app.extensions["scene_manager"] = scene_manager
    app.extensions["localization_engine"] = localization_engine
    app.extensions["copilot_service"] = CopilotService(
        app.config, scene_manager, localization_engine
    )
    app.register_blueprint(api)

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Upload exceeds the configured size limit."}), 413

    with app.app_context():
        db.create_all()

    return app
