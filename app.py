import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, html
import dash_bootstrap_components as dbc
import svgelements
import numpy as np
from tensorflow import keras
from preprocessing import center_by_mass
from dash_bootstrap_templates import load_figure_template

#predictive model
MODEL = keras.models.load_model("my_first.model")

#global styling
TEMPLATE = 'darkly'
load_figure_template([TEMPLATE])
GRAPH_MARGINS = dict(l=30,r=30,t=30,b=30)
IMAGE_SHAPE = (28,28)

#These functions return plotly figures to be displayed
def canvas_fig(display_image):
    '''
    Returns a plotly figure that displays the display_image. 
    
    This figure is intended to be drawn on and will pass
    the svg paths of shapes drawn on it to the on_draw callback.

    Args:
        display_image: A 2d-array of binary pixel values. Should be a numpy array.

    Returns:
        a Plotly figure displaying display_image
    '''
    fig = px.imshow(display_image,binary_string=True,range_color=(0,1),title="Draw a digit",template=TEMPLATE)
    fig.update_layout(
        newshape=dict(line=dict(color='white')),
        dragmode="drawopenpath",
        margin = GRAPH_MARGINS,
    )
    return fig

def preprocess_fig(display_image):
    '''Returns a plotly figure that displays the display_image. 
    
    This figure is intended to have the preprocessed digit from the
    on_draw callback passed to it.

    Args:
        @display_image A 2d-array of binary pixel values. Should be a numpy array.

    Returns:
        a Plotly figure displaying display_image
    '''
    fig = px.imshow(display_image,binary_string=True,title="Preprocessed Digit",template=TEMPLATE)
    fig.update_layout(
        dragmode = False,
        hovermode = False,
        clickmode="none",
        margin = GRAPH_MARGINS
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def prediction_fig(prediction,color):
    '''Returns a plotly figure that displays the display_image. 
    
    This figure is intended to have the model prediction from the
    on_draw callback passed to it.

    Args:
        display_image: A 2d-array of binary pixel values. Should be a numpy array.

    Returns:
        a Plotly figure displaying display_image.
    '''
    fig = px.imshow([[prediction]],text_auto=True,title="Model Prediction",color_continuous_scale=[color,color],template=TEMPLATE)
    fig.update_traces(
        textfont_size=100,
        showscale=False
    )
    fig.update_layout(
        dragmode=False,
        hovermode = False,
        clickmode="none",
        coloraxis_showscale=False,
        margin=GRAPH_MARGINS
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

#other helper functions
def get_pixels_from_svg_path(shapes):
    '''Converts svg paths to pixel arrays
    
    The svg paths are low resolution so we interpolate the lines between each point.

    Args:
        shapes: list of svg paths
    
    Returns:
        a binary matrix whose values correspond to pixels.
    '''
    pixels = np.zeros(IMAGE_SHAPE)
    for shape in shapes:
        svg = shape["path"]
        points =  list(svgelements.Path(svg).as_points())
        for i in range(len(points) - 1):
            p1,p2 = (points[i].x,points[i].y),(points[i+1].x,points[i+1].y)
            interpolation = np.linspace(p1,p2,10)
            for x,y in interpolation.astype(np.intc):
                pixels[max(0,y-1):min(y+2,IMAGE_SHAPE[1]),max(0,x-1):min(x+2,IMAGE_SHAPE[0])] = 1
    return pixels

#app declaration and layout
app = DashProxy(transforms=[MultiplexerTransform()],external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.layout = dbc.Container(
    [
        html.H1("MNIST Digit Classifier",className="mt-3"),
        html.Hr(),
        html.P("This model classifies handwritten digits. It was trained on a modified version of the MNIST database."),
        dbc.Row(
            [
                dbc.Col([
                    canvas := dcc.Graph(figure=canvas_fig(np.zeros(IMAGE_SHAPE)),style={'width': '70vh', 'height': '70vh'}),
                    reset_button := dbc.Button('Reset',n_clicks=0,className="d-grid gap-2 col-6 mx-auto")
                ],width=5),

                dbc.Col(html.Div([
                    preprocess := dcc.Graph(figure=preprocess_fig(np.zeros(IMAGE_SHAPE)),style={'width': '35vh', 'height': '35vh'}),
                    prediction := dcc.Graph(figure=prediction_fig(0,'black'),style={'width': '35vh', 'height': '35vh'})
                ]),width=3),

                # dbc.Col(html.Div([
                #     html.P("""The images in the training set are 28x28. The digits are fit to a 20x20 box and centered by mass.
                #      A threshold function was applied to the original MNIST dataset to obtain black and white images. Before your handwritten digit is fed
                #      to the classifier, it is preprocessed in the same way as shown on the left.
                #     """),
                #     html.P("The model achieves high accuracy both on the training set and new digits drawn by visitors.")
                # ]),width=3),
            ])
    ],
    fluid=True,
)

#app callbacks
@app.callback(
    Output(canvas,"figure"),
    Output(preprocess,"figure"),
    Output(prediction,"figure"),
    Input(reset_button, "n_clicks"),
    prevent_initial_call=True,
)
def on_reset(n_clicks):
    '''Triggered on reset_button click. 
    
    Resets all figures on the page to their original state

    Args:
        n_clicks: the n_clicks attribute of the reset_button. When the button is clicked this attribute is incremented triggering the callback
    
    Returns:
        tuple containing canvas, preprocess and prediction figures
    '''
    return canvas_fig(np.zeros(IMAGE_SHAPE)),preprocess_fig(np.zeros(IMAGE_SHAPE)),prediction_fig(0,"black")

@app.callback(
    Output(prediction,"figure"),
    Output(preprocess, "figure"),
    Input(canvas, "relayoutData"),
    prevent_initial_call=True,
)
def on_draw(relayout_data):
    '''Triggered when a shape is drawn on the canvas.
    
    Converts the svg paths of the shapes drawn on the campus to pixel matrices and passes them to the model for prediction. The preprocessed 
    pixel matrix and an image of the predicted digit are passed to the preprocesss and prediction figures respectively.

    Args:
        relayout_data: the relayout_data attribute of the canvas figure. Its shapes attribute contains the svg paths of 
        the objects drawn on the canvas

    Returns:
        tuple with first element updated prediction figure and second element updated preprocess figure
    '''
    if "shapes" not in relayout_data:
        return dash.no_update
    shapes = relayout_data["shapes"]
    pixels = get_pixels_from_svg_path(shapes)
    pixels = center_by_mass(pixels)
    prediction = np.argmax(MODEL.predict(np.array([pixels]))[0])
    return prediction_fig(prediction,"green"),preprocess_fig(pixels)

if __name__ == "__main__":
    app.run_server(debug=True)