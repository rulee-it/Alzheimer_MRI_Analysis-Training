from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

    project_root = Path(__file__).resolve().parent
    static_dir = project_root / "static"
    upload_dir = static_dir / "uploads"
    charts_dir = static_dir / "predictions"
    upload_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def model_ready() -> bool:
        return (project_root / "models" / "alzheimer_model.h5").exists()

    @app.route("/", methods=["GET", "POST"]) 
    def index():
        if request.method == "POST":
            if "image" not in request.files:
                flash("No file part in the request")
                return redirect(request.url)
            file = request.files["image"]
            if file.filename == "":
                flash("No file selected")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                # Save with a timestamped name
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                ext = file.filename.rsplit(".", 1)[1].lower()
                filename = f"upload_{ts}.{ext}"
                saved_path = upload_dir / filename
                file.save(str(saved_path))

                # Run prediction, handle missing model gracefully
                try:
                    # Lazy import to avoid heavy TensorFlow import at app startup
                    from Alzheimer_MRI_Analysis.predict_mri import predict_mri
                    predicted_class, probs = predict_mri(str(saved_path))
                except FileNotFoundError as e:
                    flash(str(e))
                    return redirect(url_for("index"))

                # Create a probability chart and save it
                import matplotlib.pyplot as plt

                classes = list(probs.keys())
                values = [probs[k] for k in classes]
                plt.figure(figsize=(8, 5))
                bars = plt.bar(classes, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
                plt.ylim(0, 1)
                plt.ylabel("Probability")
                plt.title("Prediction Probabilities")
                for rect, p in zip(bars, values):
                    plt.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01, f"{p:.2f}", ha="center", va="bottom")
                plt.tight_layout()
                chart_name = f"probs_{ts}.png"
                chart_path = charts_dir / chart_name
                plt.savefig(chart_path)
                plt.close()

                return render_template(
                    "result.html",
                    image_url=url_for("static", filename=f"uploads/{filename}"),
                    predicted_class=predicted_class,
                    chart_url=url_for("static", filename=f"predictions/{chart_name}"),
                    probs=probs,
                )
            else:
                flash("Unsupported file type. Please upload an image (png, jpg, jpeg, bmp, gif).")
                return redirect(request.url)

        # GET request: render upload page
        return render_template("index.html", model_ready=model_ready())

    return app


if __name__ == "__main__":
    app = create_app()
    # Allow overriding via environment variables for easier local/dev control
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError:
        port = 5000
    debug_env = os.environ.get("FLASK_DEBUG", os.environ.get("DEBUG", "1"))
    debug = str(debug_env).lower() in {"1", "true", "yes", "on"}

    app.run(host=host, port=port, debug=debug)
