from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load new dataset
data = pd.read_csv("real_crop_recommendation_dataset.csv")

# features and label
X = data.drop("crop", axis=1)
y = data["crop"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# improved model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# train
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        values = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(values)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    from sklearn.metrics import accuracy_score

# after training
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round(accuracy * 100, 2)
