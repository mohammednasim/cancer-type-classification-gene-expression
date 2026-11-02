
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from flask import Flask, render_template, request

from data_utils import BASE_DIR
from predict import load_new_samples


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None
    model_name = None
    input_filename = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a CSV file."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are supported."
        elif not MODEL_PATH.exists():
            error = "Trained model not found. Run train.py first."
        else:
            try:
                artifact = joblib.load(MODEL_PATH)
                model = artifact["pipeline"]
                model_name = artifact["model_name"]
                input_filename = uploaded_file.filename

                # Use a temporary file so the original upload is not retained.
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".csv", delete=False
                ) as temp:
                    uploaded_file.save(temp)
                    temp_path = Path(temp.name)

                try:
                    X_new = load_new_samples(
                        temp_path,
                        artifact["feature_columns"],
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

                predicted_classes = model.predict(X_new)

                results = pd.DataFrame({
                    "Sample": range(1, len(X_new) + 1),
                    "Predicted Cancer Type": predicted_classes,
                })

                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_new)

                    for i, class_name in enumerate(model.classes_):
                        results[f"Score: {class_name}"] = scores[:, i]

                elif hasattr(model, "decision_function"):
                    import numpy as np

                    scores = np.asarray(model.decision_function(X_new))

                    if scores.ndim == 2:
                        for i, class_name in enumerate(model.classes_):
                            results[f"Decision: {class_name}"] = scores[:, i]

                predictions = results.to_dict(orient="records")

            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        model_name=model_name,
        input_filename=input_filename,
    )


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        error="File is too large. The maximum upload size is 25 MB.",
        predictions=None,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)

