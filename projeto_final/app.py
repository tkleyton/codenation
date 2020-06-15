import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


n_of_portfolios = 3
features = [
    'sg_uf',
    'setor',
    'idade_emp_cat',
    'de_saude_tributaria',
    'de_faixa_faturamento_estimado',
    'natureza_juridica_macro',
    'nm_divisao',
    'nm_segmento',
    'de_ramo',
]

external_url = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(external_stylesheets=[external_url])
server = app.server
app.title = 'Leads recommender dashboard'

# ---------------------------------------------------------------
# Preparing data for barplots

ports_dfs = [pd.read_csv(f'data/ports_dfs{i+1}.csv') for i in range(n_of_portfolios)]
leads_dfs = [pd.read_csv(f'data/leads_dfs{i+1}.csv') for i in range(n_of_portfolios)]

# ---------------------------------------------------------------
# Preparing data for scatterplot
reduced_df = pd.read_csv('data/reduced_df.csv')

ports_dfs_clean = [pd.read_csv(f'data/ports_dfs_clean{i+1}.csv') for i in range(n_of_portfolios)]
leads_dfs_clean = [pd.read_csv(f'data/leads_dfs_clean{i+1}.csv') for i in range(n_of_portfolios)]

# ---------------------------------------------------------------

@app.callback(
    Output(component_id='bar_fig', component_property='figure'),
    [Input('port_num', 'value')]
)
def make_features_plot(i):
    global features
    global ports_dfs
    global leads_dfs
    port_df = ports_dfs[i]
    leads_df = leads_dfs[i]
    title = f'Portfólio {i+1} vs Leads'

    cols = 3
    rows = -(len(features) // -cols) # Hacky ceil division
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features,
                        vertical_spacing=0.2)

    for i in range(rows):
        for j in range(cols):
            ix = i*cols + j
            
            # Since there's no option to share legends between subplots,
            # we're applying a workaround by grouping the legends and
            # showing only the legends of the first subplot.
            showlegend = False
            if not ix:
                showlegend = True
                
            port_plot = port_df.groupby(features[ix]).agg({features[ix]: 'count'})
            leads_plot = leads_df.groupby(features[ix]).agg({features[ix]: 'count'})

            fig.add_trace(
                go.Bar(name='Portfólio', legendgroup='a', showlegend=showlegend,
                       x=port_plot.index.map(lambda x: x[:20]),  # limit char length
                       y=port_plot[features[ix]].values,
                       marker_color='steelblue'),
                row=i+1, col=j+1
            )
            fig.add_trace(
                go.Bar(name='Leads', legendgroup='b', showlegend=showlegend,
                       x=leads_plot.index.map(lambda x: x[:20]),
                       y=leads_plot[features[ix]].values,
                       marker_color='indianred'),
                row=i+1, col=j+1
            )
    fig.update_layout(title=title)
    fig.update_xaxes(tickangle=40, tickfont={'size': 8})
    return fig


def make_scatterplot(reduced_df, ports_dfs_clean, leads_dfs_clean):
    fig = go.Figure()
    
    # For large data volumes, use Scattergl
    fig.add_trace(go.Scattergl(
        x=reduced_df['x'],
        y=reduced_df['y'],
        mode='markers',
        name='Base',
        marker={
            'color': 'rgba(150, 150, 150, 0.5)',
            'size': 16,
        },
    ))

    for i in range(n_of_portfolios):
        fig.add_trace(go.Scattergl(
            x=ports_dfs_clean[i]['x'],
            y=ports_dfs_clean[i]['y'],
            mode='markers',
            name=f'Portfólio {i+1}',
            opacity=0.7,
            marker={
            'size': 16,
            'line': {'width': 1,
                     'color': 'DarkSlateGrey'},
            },
        ))
        fig.add_trace(go.Scattergl(
            x=leads_dfs_clean[i]['x'],
            y=leads_dfs_clean[i]['y'],
            mode='markers',
            name=f'Leads {i+1}',
            opacity=0.5,
            marker={
            'size': 16,
            'line': {'width': 1,
                     'color': 'DarkSlateGrey'},
            },
        ))
        
    return fig

# ---------------------------------------------------------------

app.layout = html.Div([
    html.Div([
        html.H6('Comparações por features', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='port_num',
            options=[{'label': f'Portfólio {i+1}', 'value': i} 
                    for i in range(n_of_portfolios)],
            value=0,
            style={'width': '30vw'},
            clearable=False,
        ),
        dcc.Graph(id='bar_fig', style={'height': '85vh'}),
    ], className='eight columns'),
   html.Div([
       html.H6('Gráfico de dispersão', style={'textAlign': 'center'}),
       dcc.Graph(
           figure=make_scatterplot(reduced_df, ports_dfs_clean, leads_dfs_clean),
           style={'height': '80vh'}
       ),
       html.P('Clique nos itens nas legendas para mostrar/esconder.', style={'textAlign': 'center'})
   ], className='four columns')
], style={'width': '98vw'})

# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server()