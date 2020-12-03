import os
import sys
import pickle
import numpy as np
import plotly.offline as po
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from datetime import datetime
from configurations.global_config import GlobalConfig


class CombinedVisualizer:

    def __init__(self, stock_ticker):
        self.ticker = stock_ticker
        with open('../feature_dictionaries/'+stock_ticker+'_stock_feature_dict.pkl', 'rb') as file:
            self.stock_feature_dict = pickle.load(file)

        with open('../feature_dictionaries/'+stock_ticker+'_twitter_feature_dict.pkl', 'rb') as file:
            self.twitter_feature_dict = pickle.load(file)

        with open('../feature_dictionaries/'+stock_ticker+'_reddit_feature_dict.pkl', 'rb') as file:
            self.reddit_feature_dict = pickle.load(file)



    def rescale_array(self, x, lower_bound, upper_bound):

        x_np = np.asarray(x)
        x_max = np.amax(x)
        x_min = np.amin(x)
        x_rescaled = []
        for i in x_np:
            i_rescaled = (((upper_bound-lower_bound) * (i - x_min)) / (x_max - x_min)) + lower_bound
            x_rescaled.append(i_rescaled)

        return np.asarray(x_rescaled)




    def extract_sentiment_and_stock_features(self):

        # stock feature lists
        over_night_diff_list = []
        max_margin_list = []
        abs_diff_list = []
        per_change_list = []
        abs_vol_list = []
        vol_fluc_list = []
        price_fluc_list = []

        # social media lists
        twitter_sentiment_list = []
        reddit_sentiment_list = []

        # date list
        date_list = []

        for date_key, feature_dict in self.stock_feature_dict.items():

            # exclude weekends
            if (date_key.weekday() == 5 or date_key.weekday() == 6):
                continue

            # only trading days with reddit and reddit posts are of interest
            if (date_key not in self.twitter_feature_dict.keys()) or (date_key not in self.reddit_feature_dict.keys()):
                continue

            # extract stock and social media features
            else:

                # stock features
                for feature_key, feature_value in feature_dict.items():
                    if feature_key == GlobalConfig.OVER_NIGHT_DIFF:
                        over_night_diff_list.append(feature_value)
                    elif feature_key == GlobalConfig.MAX_MARGIN:
                        max_margin_list.append(feature_value)
                    elif feature_key == GlobalConfig.ABS_DIFFERENCE:
                        abs_diff_list.append(feature_value)
                    elif feature_key == GlobalConfig.PER_CHANGE:
                        per_change_list.append(feature_value)
                    elif feature_key == GlobalConfig.ABS_VOL:
                        abs_vol_list.append(feature_value)
                    elif feature_key == GlobalConfig.VOL_FLUC:
                        vol_fluc_list.append(feature_value)
                    elif feature_key == GlobalConfig.PRICE_FLUC:
                        price_fluc_list.append(feature_value)
                    else:
                        continue

                # social media features
                # =============ADAPT TO FINAL TWITTER AND REDDIT DICT ================

                twitter_sentiment_list.append(self.twitter_feature_dict.get(date_key))
                reddit_sentiment_list.append(self.reddit_feature_dict.get(date_key))

                # =============ADAPT TO FINAL TWITTER AND REDDIT DICT ================

                # date list
                date_list.append(date_key)

        return np.asarray(over_night_diff_list), np.asarray(max_margin_list), np.asarray(abs_diff_list), \
               np.asarray(per_change_list), np.asarray(abs_vol_list), np.asarray(vol_fluc_list), \
               np.asarray(price_fluc_list), np.asarray(twitter_sentiment_list), \
               np.asarray(reddit_sentiment_list), np.asarray(date_list)




    def plot_sentiment_stock_feature_grafics(self, stock_feature_min=-1, stock_feature_max=1):

        # extract sentiment and stock features
        over_night_diff_list, max_margin_list, abs_diff_list, \
        per_change_list, abs_vol_list, vol_fluc_list, price_fluc_list, \
        twitter_sentiment_list, reddit_sentiment_list, date_list = \
            self.extract_sentiment_and_stock_features()

        # rescale stock features to range values between -1 and 1
        over_night_diff_list_rescaled = self.rescale_array(over_night_diff_list,
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        max_margin_list_rescaled = self.rescale_array(max_margin_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_diff_list_rescaled = self.rescale_array(abs_diff_list,
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        per_change_list_rescaled = self.rescale_array(per_change_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_vol_list_rescaled = self.rescale_array(abs_vol_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        vol_fluc_list_rescaled = self.rescale_array(vol_fluc_list,
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        price_fluc_list_rescaled = self.rescale_array(price_fluc_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)

        # create traces and layout for to plot figure
        twitter_sentiment_trace = go.Scattergl(x=date_list, y=twitter_sentiment_list, mode='lines+markers',
                                             line=dict(color='blue', width=0.5),
                                             marker=dict(color='deepskyblue', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='blue', width=2)),
                                             name='twitter_sentiment_scores',
                                             text='Sentiment scores of tweets calculated through finbert model [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='blue'))
        reddit_sentiment_trace = go.Scattergl(x=date_list, y=reddit_sentiment_list, mode='lines+markers',
                                             line=dict(color='orange', width=0.5),
                                             marker=dict(color='darkorange', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='orange', width=2)),
                                             name='reddit_sentiment_scores',
                                             text='Sentiment scores of reddit posts calculated through finbert model [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='orange'))
        over_night_diff_trace = go.Scattergl(x=date_list, y=over_night_diff_list_rescaled, mode='lines+markers',
                                             line=dict(color='grey', width=0.5),
                                             marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='grey', width=2)),
                                             name=GlobalConfig.OVER_NIGHT_DIFF,
                                             text='Price difference from close (yesterday) to open (today) - rescaled to [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='grey'))
        max_margin_trace = go.Scattergl(x=date_list, y=max_margin_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='square',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.MAX_MARGIN,
                                        text='Maximum price difference - rescaled to [-1; 1]',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        abs_diff_trace = go.Scattergl(x=date_list, y=abs_diff_list_rescaled, mode='lines+markers',
                                      line=dict(color='grey', width=0.5),
                                      marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='diamond',
                                                  line=dict(color='grey', width=2)),
                                      name=GlobalConfig.ABS_DIFFERENCE,
                                      text='Price difference between open and close today - rescaled to [-1; 1',
                                      showlegend=True,
                                      hoverinfo='text',
                                      opacity=0.7,
                                      hoverlabel=dict(bgcolor='grey'))
        per_change_trace = go.Scattergl(x=date_list, y=per_change_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='hexagram',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.PER_CHANGE,
                                        text='Percentage price change between open and close today - rescaled to [-1; 1] ',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        abs_vol_trace = go.Scattergl(x=date_list, y=abs_vol_list_rescaled, mode='lines+markers',
                                     line=dict(color='grey', width=0.5),
                                     marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='star',
                                                 line=dict(color='grey', width=2)),
                                     name=GlobalConfig.ABS_VOL,
                                     text='Absolute trading volume between open and close today - rescaled to [-1; 1] ',
                                     showlegend=True,
                                     hoverinfo='text',
                                     opacity=0.7,
                                     hoverlabel=dict(bgcolor='grey'))
        vol_fluc_trace = go.Scattergl(x=date_list, y=vol_fluc_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='bowtie',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.VOL_FLUC,
                                        text='Volume standard deviation between open and close today - rescaled to [-1; 1] ',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        price_fluc_trace = go.Scattergl(x=date_list, y=price_fluc_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='pentagon',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.PRICE_FLUC,
                                        text='Price standard deviation during the day - rescaled to [-1; 1]',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        layout_plot = dict(title=self.ticker + '<br>Sentiment Scores and Rescaled Stock Price Features',
                           xaxis=dict(title='Date',
                                      titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                           yaxis=dict(title='Features',
                                      titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                           hovermode='closest')
        figure_plot_1 = dict(data=[twitter_sentiment_trace, reddit_sentiment_trace,
                                   over_night_diff_trace, max_margin_trace, abs_diff_trace,
                                   per_change_trace, abs_vol_trace, vol_fluc_trace, price_fluc_trace],
                             layout=layout_plot)
        po.plot(figure_plot_1, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                                     self.ticker, "sentiment_scores_and_price_features.html"), auto_open=False)
        print(datetime.now(), ':', self.ticker, 'sentiment_scores_and_price_feature_plot created.')




    def plot_correlation_bar_chart(self, stock_feature_min=-1, stock_feature_max=1):

        # extract sentiment and stock features
        over_night_diff_list, max_margin_list, abs_diff_list, \
        per_change_list, abs_vol_list, vol_fluc_list, price_fluc_list, \
        twitter_sentiment_list, reddit_sentiment_list, date_list = \
            self.extract_sentiment_and_stock_features()

        # rescale stock features to range values between -1 and 1
        over_night_diff_list_rescaled = self.rescale_array(over_night_diff_list,
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        max_margin_list_rescaled = self.rescale_array(max_margin_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_diff_list_rescaled = self.rescale_array(abs_diff_list,
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        per_change_list_rescaled = self.rescale_array(per_change_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_vol_list_rescaled = self.rescale_array(abs_vol_list,
                                                   lower_bound=stock_feature_min,
                                                   upper_bound=stock_feature_max)
        vol_fluc_list_rescaled = self.rescale_array(vol_fluc_list,
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        price_fluc_list_rescaled = self.rescale_array(price_fluc_list,
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)

        # calculate stock feature correlations to reddit sentiment
        reddit_corrcoef_list = []
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, over_night_diff_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, max_margin_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, abs_diff_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, per_change_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, abs_vol_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, vol_fluc_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(reddit_sentiment_list, price_fluc_list_rescaled)[0, 1])

        # calculate stock feature correlations to twitter sentiment
        twitter_corrcoef_list = []
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, over_night_diff_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, max_margin_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, abs_diff_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, per_change_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, abs_vol_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, vol_fluc_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(twitter_sentiment_list, price_fluc_list_rescaled)[0, 1])

        stock_feature_list = [GlobalConfig.OVER_NIGHT_DIFF, GlobalConfig.MAX_MARGIN,
                              GlobalConfig.ABS_DIFFERENCE, GlobalConfig.PER_CHANGE,
                              GlobalConfig.ABS_VOL, GlobalConfig.VOL_FLUC, GlobalConfig.PRICE_FLUC]

        # create figure and layout
        fig = go.Figure(data=[
            go.Bar(name='Correlation with Twitter Sentiment', x=stock_feature_list, y=twitter_corrcoef_list),
            go.Bar(name='Correlation with Reddit Sentiment', x=stock_feature_list, y=reddit_corrcoef_list)
        ])
        fig.update_layout(title=self.ticker+'<br>Correlation of Stock Features with Twitter and Reddit Sentiment',
                          xaxis=dict(title='Stock Features', titlefont_size=16, tickfont_size=14),
                          yaxis=dict(title='Correlation', titlefont_size=16, tickfont_size=14),
                          barmode='group', bargap=0.15, bargroupgap=0.05)
        po.plot(fig, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                           self.ticker, "correlation_bar_chart.html"), auto_open=False)
        print(datetime.now(), ':', self.ticker, 'correlation_bar_chart_plot created.')

