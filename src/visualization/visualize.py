# Import standard libraries
import sys
sys.path.append('C:/Users/rohan/Documents/Projects/Food_Demand_Forecasting_Challenge/Food_Demand_Forecasting_Challenge')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
import warnings
warnings.filterwarnings("ignore")


def make_jitter_plot(data,x,y):
    """
        Make a scatter plot between two variables data[x] and data[y]
    """
    (p9.ggplot(data=data,
               mapping=p9.aes(x=x, y=y))
    + p9.geom_jitter(alpha=0.2)
    + p9.scales.scale_color_cmap(name='viridis'));

def make_bar_plot(data,x,y):
    """
        Make a bar plot between two variables data[x] and data[y]
    """
    (p9.ggplot(data=data,
           mapping=p9.aes(x=x, y=y))
    + p9.geom_bar(stat='identity'))
    + p9.theme(axis_text_x=p9.element_text(angle=90))
    + p9.labs(title='{} By {}'.format(x,y)));

def make_violin_plot(data,x,y,color):
    """
        Make a violin plot between data[x], data[y] and data[color]
    """
    (p9.ggplot(data=train_data,
               mapping=p9.aes(x='factor(checkout_price)', y='num_orders', color='emailer_for_promotion'))
    + p9.geom_violin()
    + p9.theme(axis_text_x=p9.element_text(angle=90))
    + p9.labs(title='{} Vs {} Vs {}'.format(x,y,color)));