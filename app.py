import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

app = dash.Dash(__name__)

# Load data and model
df = pd.read_csv('data/processed/processed_data.csv')
df_raw = pd.read_csv('data/raw/customer_data.csv')

with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Apply preprocessing to data
def preprocess_data(df):
    for column, encoder in label_encoders.items():
        df[column] = encoder.transform(df[column])
    return df

def create_confusion_matrix():
    df_processed = preprocess_data(df.copy())
    X = df_processed.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df_processed['Churn']
    y_pred = model.predict(X)
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    if isinstance(y_pred[0], str):
        le = LabelEncoder()
        y_pred = le.fit_transform(y_pred)

    cm = confusion_matrix(y, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn'],
        colorscale='Blues',
        colorbar=dict(title='Count'),
        zmin=0,
        zmax=cm.max()
    ))
    fig.update_layout(title='Confusion Matrix')
    return fig

def create_roc_curve():
    df_processed = preprocess_data(df.copy())
    X = df_processed.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df_processed['Churn']
    
    # Convert y to numeric
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    y_prob = model.predict_proba(X)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob, pos_label=1)  # Ensure pos_label is set
    roc_auc = auc(fpr, tpr)
    fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}, title=f'ROC Curve (AUC = {roc_auc:.2f})')
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="Red", width=2, dash="dash"))
    return fig


def create_feature_importance_plot():
    df_processed = preprocess_data(df.copy())
    importances = model.feature_importances_
    features = df_processed.drop(['Churn', 'customerID'], axis=1, errors='ignore').columns
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    fig = px.bar(feature_importances, x='Importance', y='Feature', title='Feature Importances')
    return fig

app.layout = html.Div([
    html.H1('Churn Prediction Dashboard'),
    
    html.Div([
        html.H3('Model Performance'),
        dcc.Graph(
            id='confusion-matrix',
            figure=create_confusion_matrix()
        ),
        dcc.Graph(
            id='roc-curve',
            figure=create_roc_curve()
        )
    ]),

    html.Div([
        html.H3('Feature Importances'),
        dcc.Graph(
            id='feature-importances',
            figure=create_feature_importance_plot()
        )
    ]),

    html.Div([
        html.H3('Churn Distribution'),
        dcc.Graph(
            id='churn-distribution',
            figure=px.histogram(df_raw, x='MonthlyCharges', color='Churn', title='Monthly Charges Distribution by Churn')
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
