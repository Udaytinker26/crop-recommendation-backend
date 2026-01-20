from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "🌱 Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"])
    ]]

    probabilities = model.predict_proba(features)[0]
    crops = model.classes_

    results = []

    for crop, prob in zip(crops, probabilities):
        confidence = round(prob * 100, 2)
        if confidence >= 10:
            results.append({
                "crop": crop,
                "confidence": confidence,
                "level": "Possible Choice"
            })

    if not results:
        top_index = probabilities.argmax()
        results.append({
            "crop": crops[top_index],
            "confidence": round(probabilities[top_index] * 100, 2),
            "level": "Best Choice"
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return jsonify({"recommendations": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
