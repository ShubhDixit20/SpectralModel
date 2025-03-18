from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained PLS model
pls_model = joblib.load("pls_regressor.pkl")

# The 18 wavelengths used as input
specific_wavelengths = [410, 450, 470, 490, 510, 530, 550, 570, 590, 610, 
                        630, 650, 670, 690, 710, 730, 860, 940]

def extract_advanced_features(df, wavelengths):
    df.columns = pd.to_numeric(df.columns, errors='ignore')

    df['mean_reflectance'] = df[wavelengths].mean(axis=1)
    df['std_reflectance'] = df[wavelengths].std(axis=1)
    df['max_reflectance'] = df[wavelengths].max(axis=1)
    df['min_reflectance'] = df[wavelengths].min(axis=1)
    df['range_reflectance'] = df['max_reflectance'] - df['min_reflectance']
    df['median_reflectance'] = df[wavelengths].median(axis=1)

    df['slope'] = df[wavelengths].diff(axis=1).mean(axis=1)
    df['first_derivative_mean'] = df[wavelengths].diff(axis=1).mean(axis=1)
    df['second_derivative_mean'] = df[wavelengths].diff(axis=1).diff(axis=1).mean(axis=1)

    if 860 in wavelengths and 670 in wavelengths:
        df['NDVI'] = (df[860] - df[670]) / (df[860] + df[670])
        df['SR'] = df[860] / df[670]

    if 740 in wavelengths and 705 in wavelengths:
        df['RedEdge'] = (df[740] - df[705]) / (df[740] + df[705])

    visible = [w for w in wavelengths if 410 <= w <= 700]
    nir = [w for w in wavelengths if 700 < w <= 940]
    df['visible_mean'] = df[visible].mean(axis=1)
    df['nir_mean'] = df[nir].mean(axis=1)
    df['visible_nir_ratio'] = df['visible_mean'] / df['nir_mean']

    df['skewness'] = df[wavelengths].skew(axis=1)
    df['kurtosis'] = df[wavelengths].kurt(axis=1)

    for i, w1 in enumerate(wavelengths):
        for w2 in wavelengths[i+1:]:
            df[f'poly_{w1}_{w2}'] = df[w1] * df[w2]

    return df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.json
        X_new = np.array([data["wavelengths"]])  # (1, 18)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_new, columns=specific_wavelengths)

        # Extract features to match model input size (187 features)
        X_transformed = extract_advanced_features(X_df, specific_wavelengths)

        # Ensure feature count is correct before prediction
        if X_transformed.shape[1] != 187:
            return jsonify({"error": f"Feature extraction failed. Expected 187 features, got {X_transformed.shape[1]}."})

        # Make prediction
        y_pred = pls_model.predict(X_transformed)

        return jsonify({"prediction": y_pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return jsonify({"message": "PLS Regression API is running!"})

if __name__ == "__main__":
    app.run(debug=True)
