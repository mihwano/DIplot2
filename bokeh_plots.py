from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, PanTool, WheelZoomTool,\
                         BoxZoomTool, ResetTool, ResizeTool, OpenURL
import pandas as pd
from bokeh.sampledata import us_states
from collections import OrderedDict
from collections import Counter
from bokeh.charts import Bar
from bokeh.palettes import YlOrRd4
import pdb
import io
import requests


STATE_DICT = {'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AS': 'American Samoa','AZ': 'Arizona','CA': 'California',
               'CO': 'Colorado','CT': 'Connecticut','DC': 'District of Columbia','DE': 'Delaware','FL': 'Florida','GA': 'Georgia','GU': 'Guam',
               'HI': 'Hawaii','IA': 'Iowa','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana',
               'MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan','MN': 'Minnesota','MO': 'Missouri','MP': 'Northern Mariana Islands',
               'MS': 'Mississippi','MT': 'Montana','NA': 'National','NC': 'North Carolina','ND': 'North Dakota','NE': 'Nebraska','NH': 'New Hampshire',
               'NJ': 'New Jersey','NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio','OK': 'Oklahoma','OR': 'Oregon','PA': 'Pennsylvania',
               'PR': 'Puerto Rico','RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
               'UT': 'Utah','VA': 'Virginia','VI': 'Virgin Islands','VT': 'Vermont','WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia','WY': 'Wyoming'}


def display_map(df, states_coords):
    #remove puerto rico, hawai and alaska
    df = df[~df['location'].isin(['Hawaii', 'Puerto Rico', 'Alaska'])]

    # initialize map
    us_states = states_coords.data.copy()
    ordered_states = {}
    for state in us_states:
        if STATE_DICT[state] in pd.unique(df['location']):
            ordered_states[STATE_DICT[state]] = us_states[state]
        else:
            continue
    state_xs = [ordered_states[code]["lons"] for code in df['location']]
    state_ys = [ordered_states[code]["lats"] for code in df['location']]
    title = df.groupby('1rst tgrm', as_index=False).count().sort_values('location', ascending=False)[:2]
    mapfig = figure(tools='pan, hover, box_zoom, reset, wheel_zoom', width=1100, height=700,
                    title='main trigrams: %s    and    %s!' %(' '.join(title['1rst tgrm'].iloc[0]).upper(), 
                                                              ' '.join(title['1rst tgrm'].iloc[1]).upper()))
    mapfig.axis.visible = False

    # create columndatasource
    df['colors'] = ['blue' if df['main_party'].iloc[i]=='Democratic' else 'red' for i in range(len(df))]
    df['alphas'] = [1.0 * df['partisan_split'].iloc[i] / df['partisan_split'].max() for i in range(len(df))]
    
    #source = ColumnDataSource(data=dict(state=[], term1=[], term2=[], term3=[],
    #                                    trigram=[], color=[], alphas=[]))
    # source.data = source.from_df(df[['location', '1rst term', '2nd term', '3rd term',
    #                                  '1rst tgrm', 'colors', 'alphas']])

    df.rename(columns={'1rst term': 'term1', '2nd term': 'term2', '3rd term': 'term3', '1rst tgrm': 'trigram'}, inplace=True)
    source = ColumnDataSource(ColumnDataSource.from_df(df[['location', 'term1', 'term2', 'term3',
                                     'trigram', 'colors', 'alphas']]))

    # define map
    hover = mapfig.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('State', '@location'), ('1rst term', '@term1'),
                                  ('2nd term', '@term2'), ('3rd term', '@term3'), ('1rst trigram', '@trigram')])
    output_file("mapplot.html")
    mapfig.patches(state_xs, state_ys, fill_color="colors", fill_alpha="alphas",
                   line_color="white", line_width=0.5, source=source)
    return mapfig


def display_graph(df):
    df['tweet_ratio'] = 1.0 * df['count'] / df['population']
    df['partisan_split'] = [df['partisan_split'].iloc[i] if df['main_party'].iloc[i] == 'Democratic'\
                            else -df['partisan_split'].iloc[i] for i in range(len(df))]
    df = df[['tweet_ratio', 'partisan_split', 'sentiment', 'location']]
    colors = []
    for i in range(len(df)):
        if df['sentiment'].iloc[i] < df['sentiment'].describe()['25%']:
            colors.append('quartile_1')
        elif (df['sentiment'].iloc[i] < df['sentiment'].describe()['50%']) & (df['sentiment'].iloc[i] > df['sentiment'].describe()['25%']):
            colors.append('quartile_2')
        elif (df['sentiment'].iloc[i] < df['sentiment'].describe()['75%']) & (df['sentiment'].iloc[i] > df['sentiment'].describe()['50%']):
            colors.append('quartile_3')
        else:
            colors.append('quartile_4')
    df['colors'] = colors
    source = ColumnDataSource(df)

    #p = figure(tools='pan, hover, box_zoom, reset, wheel_zoom')
    p = Bar(values='tweet_ratio', label='partisan_split', color='colors', palette=YlOrRd4[::-1], data=df,
            legend=False, plot_width=1100, plot_height=700, stack='location',
            title='Tweets per Capita vs Partisan Split and average tweet sentiment (redder = higher sentiment)', tools='hover')
    p.yaxis.axis_label = 'Tweet per Person'
    p.xaxis.axis_label = 'Partisan Split (positive number = more self identified democrats)'

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('State', '@location'), ('Tweet per Capita', '@height'),
                                  ('Partisanship', '@partisan_split'), ('sentiment', '@colors')])
 #   output_file("bar.html")
    return p


def first_plot():
    url = 'https://raw.githubusercontent.com/mihwano/DIplot1/master/final2.csv'
    df = pd.read_csv(url, encoding='utf8')
    df = df.groupby('location', as_index=False)[['1rst term', '2nd term', '3rd term', '1rst tgrm',
                                                 'main_party', 'partisan_split']].first()
    mapfig = display_map(df, us_states)
    # show(mapfig)
    return p


def second_plot():
    url = 'https://raw.githubusercontent.com/mihwano/DIplot1/master/final2.csv'
    df = pd.read_csv(url, encoding='utf8')
    # df = pd.read_csv('final.csv', encoding='latin')

    df = df.groupby('location', as_index=False)[['population', 'count', 'sentiment', 'main_party','partisan_split']].first()
    p = display_graph(df)
#    show(p)
    return p


p = first_plot()
#p = second_plot()

l = layout([[p]], sizing_mode='fixed')

curdoc().add_root(l)