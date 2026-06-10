"""
Flask Web Application for Number Generator.
Provides REST API and web dashboard.
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import os


def create_app(config: dict = None) -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # Configuration
    app.config.update({
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
        "SQLALCHEMY_DATABASE_URI": os.environ.get("DATABASE_URL", "sqlite:///numgen.db"),
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "JSON_SORT_KEYS": False,
    })
    if config:
        app.config.update(config)

    # Enable CORS
    CORS(app)

    # Register blueprints
    from src.web.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Web routes
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "code": 404}), 404

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "code": 400}), 400

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error", "code": 500}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
