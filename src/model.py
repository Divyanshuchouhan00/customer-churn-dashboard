from sklearn.linear_model import LogisticRegression

def train_model(df):
    # Drop missing values to avoid misalignment
    df = df.dropna()

    # Ensure 'Churn' column exists
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in DataFrame.")

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Debug prints to confirm alignment
    print("Training model...")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in sample size: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model

def predict_churn(model, input_df, threshold=0.5):
    # Predict probability of churn
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)
    return prediction, prob