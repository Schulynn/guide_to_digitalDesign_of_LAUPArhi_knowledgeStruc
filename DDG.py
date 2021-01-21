import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import plotly.express as px
import networkx as nx
import sys,os

# sys.path.append(os.getcwd())
# print(os.getcwd())
G=nx.read_gpickle(r'C:\Users\richi\omen-richiebao\omen_github\guide_to_digitalDesign_of_LAUPArhi_knowledgeStruc\model\G_1.pkl')
edges=list(G.edges())
nodes=G.nodes(data=True)
layers=[(v,data["layer"]) for v, data in G.nodes(data=True)]

elements_nodes=[{'data':{'id':i[0],'label':i[0]}} for i in layers]
elements_edges=[{'data':{'source':i[0],'target':i[1]}} for i in edges]
elements_nodes_edges=elements_nodes+elements_edges

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.title = "Digital design Knowledge points Structure"

app.layout = html.Div([
    html.P("Digital design Knowledge points Structure:"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=elements_nodes_edges,
        layout={'name': 'breadthfirst'},
        style={'width': '2000px', 'height': '5000px'}
    )
])


if __name__ == '__main__':
    app.run_server(debug=True)

