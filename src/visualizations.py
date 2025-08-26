import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import confusion_matrix

# ---------------------- Monthly Charges by Churn ----------------------
def plot_monthly_charges(df):
    fig = px.box(
        df, 
        x='Churn', 
        y='MonthlyCharges', 
        color='Churn',
        title='Monthly Charges by Churn',
        points="all"  # Shows individual points
    )
    return fig

# ---------------------- Tenure Group vs Churn ----------------------
def plot_tenure_churn(df):
    df = df.copy()
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins=[0,12,24,48,60,72], 
        labels=["0-12","13-24","25-48","49-60","61+"]
    )
    fig = px.histogram(
        df, 
        x='tenure_group', 
        color='Churn', 
        barmode='group',
        title='Tenure Group vs Churn'
    )
    return fig

# ---------------------- Feature Correlation Heatmap ----------------------
def plot_correlation_heatmap(df):
    fig = px.imshow(
        df.corr(), 
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Feature Correlation Heatmap'
    )
    return fig

# ---------------------- Confusion Matrix Plot ----------------------
def plot_confusion_matrix(y_true, y_pred):
    # Fill NaN and ensure integers
    y_true = pd.Series(y_true).fillna(0).astype(int)
    y_pred = pd.Series(y_pred).fillna(0).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    x = ["Predicted No", "Predicted Yes"]
    y = ["Actual No", "Actual Yes"]
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=x,
        y=y,
        colorscale="Viridis",
        showscale=True
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

