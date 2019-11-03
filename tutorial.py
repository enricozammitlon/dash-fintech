import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import quandl
import pandas as pd
import plotly.graph_objs as go
from findatapy.market import Market, MarketDataRequest, MarketDataGenerator

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='my-id', value='',debounce=True, type='text'),
    dcc.Input(id='my-id2', value='',debounce=True, type='text'),
    dcc.Input(id='my-id3', value='',debounce=True, type='text'),
   #html.Button('Submit',id='submit-api-key'),
    html.Div(id='my-div'),
    dcc.Graph(
        id='graph'
    )
])

a=""

@app.callback(
    Output(component_id='graph', component_property='figure'),
   # [State('submit-api-key','')],
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    market = Market(market_data_generator=MarketDataGenerator())
    # download equities data from Yahoo
    md_request = MarketDataRequest(
        start_date="decade",            # start date
        data_source='yahoo',            # use Bloomberg as data source
        tickers=['Apple', 'Citigroup'], # ticker (findatapy)
        fields=['close'],               # which fields to download
        vendor_tickers=['aapl', 'c'],   # ticker (Yahoo)
        vendor_fields=['Close'])        # which Bloomberg fields to download)

    data = market.fetch_market(md_request)

    
    if input_value is None:
        # PreventUpdate prevents ALL outputs updating
        raise dash.exceptions.PreventUpdate

    quandl.ApiConfig.api_key = input_value

    #data = quandl.get("EIA/PET_RWTC_D",rows=5)
    #print(data.head())
    print(data.info())
    figure={
            'data':[go.Scatter(
                x=data.index,
                y=data['Apple.close']
            )],
            'layout':{
                'title':'Test Graph'
            }
        }
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
