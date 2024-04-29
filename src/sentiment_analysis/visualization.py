import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sentiment_analysis.assets import sentiment_analysis, sentiment_summary, preprocessed_comments, emotion_analysis
from dagster import build_op_context
from src.sentiment_analysis.resources import text_preprocessor_resource

raw_comments_df = pd.read_csv("data/mental_health.csv") 

resources = {
    'text_preprocessor': text_preprocessor_resource
}

context = build_op_context(resources=resources)

df_preprocessed = preprocessed_comments(context=context, raw_comments=raw_comments_df)
df_sentiment_analysis = sentiment_analysis(df_preprocessed)
df_emotion_analysis = emotion_analysis(df_preprocessed)
summary = sentiment_summary(df_sentiment_analysis)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Mental Health Analysis Dashboard'),
    html.Div(children='''Visualizing Sentiment and Emotion Analysis of Mental Health Comments.'''),
    dcc.Graph(id='sentiment-bar-chart'),
    dcc.Graph(id='sentiment-pie-chart'),
    dcc.Graph(id='emotion-bar-chart'),
    dcc.Graph(id='emotion-heatmap'),
    html.Div(id='selected-data')
])

@app.callback(
    Output('sentiment-bar-chart', 'figure'),
    [Input('sentiment-bar-chart', 'clickData')])
def update_sentiment_graph(clickData):
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
    Output('sentiment-pie-chart', 'figure'),
    [Input('sentiment-bar-chart', 'clickData')])
def update_sentiment_pie_chart(clickData):
    fig = go.Figure(data=[
        go.Pie(
            labels=summary['sentiment'],
            values=summary['count'],
            hole=.3
        )
    ])
    fig.update_layout(
        title='Proportion of Comments by Sentiment'
    )
    return fig

@app.callback(
    Output('emotion-bar-chart', 'figure'),
    [Input('emotion-bar-chart', 'clickData')])
def update_emotion_graph(clickData):
    emotions = df_emotion_analysis.columns[1:] 
    emotion_counts = df_emotion_analysis[emotions].sum()
    fig = go.Figure(data=[
        go.Bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            marker_color='purple'
        )
    ])
    fig.update_layout(
        title='Count of Emotions in Comments',
        xaxis_title='Emotion',
        yaxis_title='Count',
        bargap=0.2,
    )
    return fig

@app.callback(
    Output('emotion-heatmap', 'figure'),
    [Input('emotion-bar-chart', 'clickData')])
def update_emotion_heatmap(clickData):
    emotions = df_emotion_analysis.columns[1:]
    emotion_data = df_emotion_analysis[emotions]
    fig = go.Figure(data=[
        go.Heatmap(
            z=emotion_data.T,
            x=emotion_data.index,
            y=emotions,
            colorscale='Viridis'
        )
    ])
    fig.update_layout(
        title='Heatmap of Emotion Intensities',
        xaxis_title='Comment Index',
        yaxis_title='Emotions'
    )
    return fig

@app.callback(
    Output('selected-data', 'children'),
    [Input('sentiment-bar-chart', 'clickData'), Input('emotion-bar-chart', 'clickData')])
def display_click_data(sentiment_click_data, emotion_click_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'Click on a bar to see details.'
    click_data = ctx.triggered[0]['value']
    if click_data is not None:
        point = click_data['points'][0]
        return f'You clicked on {point["x"]} which has {point["y"]} comments.'
    return 'Click on a bar to see details.'

if __name__ == '__main__':
    app.run_server(debug=True)