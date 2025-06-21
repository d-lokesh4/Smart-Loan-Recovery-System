from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare model
df = pd.read_csv("loan-recovery.csv")

features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
            'Num_Missed_Payments', 'Days_Past_Due']

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Borrower_Segment'] = kmeans.fit_predict(df_scaled)

df['Segment_Name'] = df['Borrower_Segment'].map({
    0: 'Moderate Income, High Loan Burden',
    1: 'High Income, Low Default Risk',
    2: 'Moderate Income, Medium Risk',
    3: 'High Loan, Higher Default Risk'
})

df['High_Risk_Flag'] = df['Segment_Name'].apply(
    lambda x: 1 if x in ['High Loan, Higher Default Risk', 'Moderate Income, High Loan Burden'] else 0
)

X = df[features]
y = df['High_Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    age = float(request.form['Age'])
    income = float(request.form['Monthly_Income'])
    loan_amount = float(request.form['Loan_Amount'])
    loan_tenure = float(request.form['Loan_Tenure'])
    interest_rate = float(request.form['Interest_Rate'])
    collateral_value = float(request.form['Collateral_Value'])
    outstanding_amount = float(request.form['Outstanding_Loan_Amount'])
    monthly_emi = float(request.form['Monthly_EMI'])
    missed_payments = float(request.form['Num_Missed_Payments'])
    days_past_due = float(request.form['Days_Past_Due'])

    # Create input DataFrame
    input_df = pd.DataFrame([[age, income, loan_amount, loan_tenure, interest_rate,
                              collateral_value, outstanding_amount, monthly_emi,
                              missed_payments, days_past_due]],
                            columns=features)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    risk_score = rf_model.predict_proba(input_df)[0][1]
    predicted_class = rf_model.predict(input_df)[0]

    if risk_score > 0.75:
        strategy = "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        strategy = "Settlement offers & repayment plans"
    else:
        strategy = "Automated reminders & monitoring"

    result_text = "High Risk" if predicted_class == 1 else "Low Risk"

    return render_template('result.html', result=result_text,
                           risk_score=round(risk_score, 2),
                           strategy=strategy)


if __name__ == '__main__':
    app.run(debug=True)
