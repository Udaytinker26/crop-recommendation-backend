from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)   # ✅ VERY IMPORTANT

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

def get_reason(data, crop):
    reasons = []

    temp = float(data["temperature"])
    humidity = float(data["humidity"])
    rainfall = float(data["rainfall"])
    ph = float(data["ph"])

    if crop == "rice":
        if rainfall > 150:
            reasons.append("High rainfall is ideal for rice.")
        if humidity > 70:
            reasons.append("High humidity supports rice growth.")
        if 20 <= temp <= 30:
            reasons.append("Temperature is suitable for rice.")
        if 5.5 <= ph <= 6.5:
            reasons.append("Soil pH is suitable for rice.")

    elif crop == "wheat":
        if rainfall < 100:
            reasons.append("Low rainfall is suitable for wheat.")
        if 15 <= temp <= 25:
            reasons.append("Cool temperature favors wheat.")
        if 6.0 <= ph <= 7.5:
            reasons.append("Soil pH is good for wheat.")

    elif crop == "maize":
        if 20 <= temp <= 30:
            reasons.append("Warm temperature is ideal for maize.")
        if 50 <= humidity <= 80:
            reasons.append("Moderate humidity supports maize.")
        if rainfall > 80:
            reasons.append("Sufficient rainfall helps maize growth.")

    if not reasons:
        reasons.append("Environmental conditions match the crop requirements.")

    return " ".join(reasons)


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

        # 🔥 LOWER THRESHOLD
        if confidence >= 10:
            if confidence >= 60:
                level = "Best Choice"
            elif confidence >= 40:
                level = "Good Choice"
            else:
                level = "Possible Choice"

            results.append({
                "crop": crop,
                "confidence": confidence,
                "level": level
            })

    # 🔥 SAFETY FALLBACK (VERY IMPORTANT)
    if not results:
        top_index = probabilities.argmax()
        results.append({
            "crop": crops[top_index],
            "confidence": round(probabilities[top_index] * 100, 2),
            "level": "Best Choice"
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "recommendations": results
    })

if __name__ == "__main__":
    # ✅ IMPORTANT: host=0.0.0.0
    app.run(host="0.0.0.0", port=5000)
