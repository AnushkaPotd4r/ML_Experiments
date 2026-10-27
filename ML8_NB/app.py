from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained Naive Bayes model
model = joblib.load(
    os.path.join(BASE_DIR, "naive_bayes_model.pkl")
)

# Load text vectorizer
vectorizer = joblib.load(
    os.path.join(BASE_DIR, "vectorizer.pkl")
)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""
    error = None

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if not message:
            error = "Please enter a message."

        else:
            # Convert message into numerical features
            message_vector = vectorizer.transform([message])

            # Predict category
            prediction = model.predict(message_vector)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)