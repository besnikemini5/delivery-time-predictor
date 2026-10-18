"""Delivery Time Predictor: a Flask app serving a scikit-learn model."""
from urllib.parse import urlparse

from flask import Flask, abort, g, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException

from .data import CATEGORIES, MAX_DISTANCE_KM, MIN_DISTANCE_KM
from .i18n import DEFAULT_LANGUAGE, LANGUAGES, translator
from .model import DeliveryModel
from .validation import parse_inputs

LANGUAGE_COOKIE = "lang"


def create_app(model=None):
    app = Flask(__name__)
    app.extensions["delivery_model"] = model or DeliveryModel.load_or_train()

    def get_model():
        return app.extensions["delivery_model"]

    @app.before_request
    def pick_language():
        lang = request.cookies.get(LANGUAGE_COOKIE)
        g.lang = lang if lang in LANGUAGES else DEFAULT_LANGUAGE
        g.t = translator(g.lang)

    @app.context_processor
    def template_helpers():
        if "t" not in g:  # error raised before the language was picked
            g.lang, g.t = DEFAULT_LANGUAGE, translator(DEFAULT_LANGUAGE)
        return {
            "t": g.t,
            "lang": g.lang,
            "languages": LANGUAGES,
            "categories": CATEGORIES,
            "min_km": MIN_DISTANCE_KM,
            "max_km": MAX_DISTANCE_KM,
            "duration": format_duration,
        }

    def format_duration(minutes):
        total = round(minutes)
        hours, mins = divmod(total, 60)
        return g.t("hours_minutes", h=hours, m=mins) if hours else g.t("minutes", m=mins)

    # ---------- pages ----------

    @app.get("/")
    def index():
        return render_template("index.html", values={}, errors={})

    @app.post("/predict")
    def predict_form():
        features, errors = parse_inputs(request.form)
        if errors:
            return render_template("index.html", values=request.form, errors=errors), 400
        prediction = get_model().predict(features)
        return render_template("result.html", features=features, prediction=prediction)

    @app.get("/model")
    def model_info():
        return render_template("model.html", metrics=get_model().metrics)

    @app.get("/language/<code>")
    def set_language(code):
        if code not in LANGUAGES:
            abort(404)
        response = redirect(_safe_back_url())
        response.set_cookie(LANGUAGE_COOKIE, code, max_age=60 * 60 * 24 * 365, samesite="Lax")
        return response

    def _safe_back_url():
        # Only redirect back to a page on this site, never to another domain
        referrer = request.referrer or ""
        parsed = urlparse(referrer)
        if parsed.netloc == request.host and parsed.path and not parsed.path.startswith("/language/"):
            return parsed.path
        return url_for("index")

    # ---------- JSON API ----------

    @app.post("/api/predict")
    def predict_api():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Request body must be a JSON object"), 400
        features, errors = parse_inputs(data)
        if errors:
            english = translator("en")
            messages = {field: english(key, min=MIN_DISTANCE_KM, max=MAX_DISTANCE_KM) for field, key in errors.items()}
            return jsonify(error="Invalid input", fields=messages), 400
        prediction = get_model().predict(features)
        metrics = get_model().metrics
        return jsonify(
            predicted_minutes=prediction["minutes"],
            likely_range={"low": prediction["low"], "high": prediction["high"]},
            inputs=features,
            model={"name": metrics["model"], "test_mae_minutes": metrics["test_metrics"]["mae"]},
        )

    @app.get("/api/model")
    def model_api():
        return jsonify(get_model().metrics)

    # ---------- errors ----------

    @app.errorhandler(HTTPException)
    def http_error(err):
        if request.path.startswith("/api/"):
            return jsonify(error=err.description), err.code
        message = translator(g.get("lang", DEFAULT_LANGUAGE))("not_found") if err.code == 404 else err.description
        return render_template("error.html", code=err.code, message=message), err.code

    @app.errorhandler(Exception)
    def unexpected_error(err):
        app.logger.exception("Unexpected error")
        if request.path.startswith("/api/"):
            return jsonify(error="Something went wrong on our side"), 500
        message = translator(g.get("lang", DEFAULT_LANGUAGE))("server_error")
        return render_template("error.html", code=500, message=message), 500

    return app
