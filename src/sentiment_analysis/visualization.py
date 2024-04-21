import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sentiment_analysis.assets import sentiment_analysis, sentiment_summary, preprocessed_comments 
from dagster import build_op_context
from src.sentiment_analysis.resources import text_preprocessor_resource

raw_comments_df = pd.read_csv("data/mental_health.csv") 

resources = {
    'text_preprocessor': text_preprocessor_resource
}

context = build_op_context(resources=resources)

df_preprocessed = preprocessed_comments(context=context, raw_comments=raw_comments_df)
df_sentiment_analysis = sentiment_analysis(df_preprocessed)
summary = sentiment_summary(df_sentiment_analysis)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Sentiment Analysis Dashboard'),
    html.Div(children='''Sentiment distribution of mental health comments.'''),
    dcc.Graph(
        id='sentiment-bar-chart',
    ),
    html.Div(id='selected-data')
])

@app.callback(
    Output('sentiment-bar-chart', 'figure'),
    [Input('sentiment-bar-chart', 'clickData')])
def update_graph(clickData):
    
    fig = go.Figure(data=[
        go.Bar(
            x=summary['sentiment'],
            y=summary['count'],
            marker_color=['green', 'blue', 'red']
        )
    ])
    fig.update_layout(
        title='Count of Comments by Sentiment',
        xaxis_title='Sentiment',
        yaxis_title='Count',
        bargap=0.2,
    )
    return fig

@app.callback(
    Output('selected-data', 'children'),
    [Input('sentiment-bar-chart', 'clickData')])
def display_click_data(clickData):
    if clickData is not None:
        sentiment = clickData['points'][0]['x']
        count = summary[summary['sentiment'] == sentiment]['count'].values[0]
        return f'You clicked on {sentiment} which has {count} comments.'
    return 'Click on a bar to see details.'

if __name__ == '__main__':
    app.run_server(debug=True)