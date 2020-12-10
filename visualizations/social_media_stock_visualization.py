import os
import sys
import json
import numpy as np
import seaborn as sns
import plotly.offline as po
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter
from analysis.stat_tests import KolmogorovSmirnov
from configurations.global_config import GlobalConfig


class CombinedVisualizer:

    def __init__(self, stock_ticker):
        self.ticker = stock_ticker
        with open('../feature_dictionaries/'+stock_ticker+'_stock_feature_dict.json', 'r') as file:
            stock_feature_dict_json = json.load(file)
        self.stock_feature_dict = prepare_stock_dict_from_json(dict_item=stock_feature_dict_json)

        with open('../feature_dictionaries/'+stock_ticker+'_twitter_feature_dict.json', 'r') as file:
            twitter_feature_dict_json = json.load(file)
        self.twitter_feature_dict = prepare_social_media_dict_from_json(dict_item=twitter_feature_dict_json)

        with open('../feature_dictionaries/'+stock_ticker+'_reddit_feature_dict.json', 'r') as file:
            reddit_feature_dict = json.load(file)
        self.reddit_feature_dict = prepare_social_media_dict_from_json(dict_item=reddit_feature_dict)


    def rescale_array(self, x, lower_bound, upper_bound):

        x_np = np.asarray(x)
        x_max = np.amax(x)
        x_min = np.amin(x)
        x_rescaled = []
        for i in x_np:
            i_rescaled = (((upper_bound-lower_bound) * (i - x_min)) / (x_max - x_min)) + lower_bound
            x_rescaled.append(i_rescaled)

        return np.asarray(x_rescaled)




    def extract_sentiment_and_stock_features(self, return_as_dict=False):

        # stock feature lists
        over_night_diff_list = []
        max_margin_list = []
        abs_diff_list = []
        per_change_list = []
        abs_vol_list = []
        vol_fluc_list = []
        price_fluc_list = []

        # social media lists
        twitter_avg_sentiment_list = []
        twitter_pos_per_list = []
        twitter_neu_per_list = []
        twitter_neg_per_list = []
        twitter_num_posts_list = []
        twitter_max_sentiment_margin_list = []
        reddit_avg_sentiment_list = []
        reddit_pos_per_list = []
        reddit_neu_per_list = []
        reddit_neg_per_list = []
        reddit_num_posts_list = []
        reddit_max_sentiment_margin_list = []

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

                # social media features - twitter
                twitter_predictions_list_help = self.twitter_feature_dict.get(date_key).get(GlobalConfig.PREDICTION_LIST)
                counter = Counter(twitter_predictions_list_help)
                num_tweets = len(twitter_predictions_list_help)
                twitter_sentiments_list_help = self.twitter_feature_dict.get(date_key).get(GlobalConfig.SENTIMENT_LIST)
                max_sentiment = np.amax(twitter_sentiments_list_help)
                min_sentiment = np.amin(twitter_sentiments_list_help)

                twitter_avg_sentiment_list.append(np.mean(twitter_sentiments_list_help))
                twitter_pos_per_list.append(counter[GlobalConfig.POSITIVE_PREDICTION_STR] / num_tweets)
                twitter_neu_per_list.append(counter[GlobalConfig.NEUTRAL_PREDICTION_STR] / num_tweets)
                twitter_neg_per_list.append(counter[GlobalConfig.NEGATIVE_PREDICTION_STR] / num_tweets)
                twitter_num_posts_list.append(num_tweets)
                twitter_max_sentiment_margin_list.append(max_sentiment - min_sentiment)


                # social media features - reddit
                reddit_predictions_list_help = self.reddit_feature_dict.get(date_key).get(GlobalConfig.PREDICTION_LIST)
                counter = Counter(reddit_predictions_list_help)
                num_tweets = len(reddit_predictions_list_help)
                reddit_sentiments_list_help = self.reddit_feature_dict.get(date_key).get(GlobalConfig.SENTIMENT_LIST)
                max_sentiment = np.amax(reddit_sentiments_list_help)
                min_sentiment = np.amin(reddit_sentiments_list_help)

                reddit_avg_sentiment_list.append(np.mean(reddit_sentiments_list_help))
                reddit_pos_per_list.append(counter[GlobalConfig.POSITIVE_PREDICTION_STR] / num_tweets)
                reddit_neu_per_list.append(counter[GlobalConfig.NEUTRAL_PREDICTION_STR] / num_tweets)
                reddit_neg_per_list.append(counter[GlobalConfig.NEGATIVE_PREDICTION_STR] / num_tweets)
                reddit_num_posts_list.append(num_tweets)
                reddit_max_sentiment_margin_list.append(max_sentiment - min_sentiment)

                # date list
                date_list.append(date_key)

        if return_as_dict:
            stock_social_media_dict=dict()
            stock_social_media_dict.update({GlobalConfig.DATE_LIST: np.asarray(date_list)})
            stock_social_media_dict.update({GlobalConfig.OVER_NIGHT_DIFF: np.asarray(over_night_diff_list)})
            stock_social_media_dict.update({GlobalConfig.MAX_MARGIN: np.asarray(max_margin_list)})
            stock_social_media_dict.update({GlobalConfig.ABS_DIFFERENCE: np.asarray(abs_diff_list)})
            stock_social_media_dict.update({GlobalConfig.PER_CHANGE: np.asarray(per_change_list)})
            stock_social_media_dict.update({GlobalConfig.ABS_VOL: np.asarray(abs_vol_list)})
            stock_social_media_dict.update({GlobalConfig.VOL_FLUC: np.asarray(vol_fluc_list)})
            stock_social_media_dict.update({GlobalConfig.PRICE_FLUC: np.asarray(price_fluc_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_AVERAGE_SENTIMENT: np.asarray(twitter_avg_sentiment_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_POS_PER_SENTIMENT: np.asarray(twitter_pos_per_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_NEU_PER_SENTIMENT: np.asarray(twitter_neu_per_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_NEG_PER_SENTIMENT: np.asarray(twitter_neg_per_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_NUM_POS: np.asarray(twitter_num_posts_list)})
            stock_social_media_dict.update({GlobalConfig.TWITTER_MAX_SENTIMENT_MARGIN: np.asarray(twitter_max_sentiment_margin_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_AVERAGE_SENTIMENT: np.asarray(reddit_avg_sentiment_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_POS_PER_SENTIMENT: np.asarray(reddit_pos_per_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_NEU_PER_SENTIMENT: np.asarray(reddit_neu_per_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_NEG_PER_SENTIMENT: np.asarray(reddit_neg_per_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_NUM_POS: np.asarray(reddit_num_posts_list)})
            stock_social_media_dict.update({GlobalConfig.REDDIT_MAX_SENTIMENT_MARGIN: np.asarray(reddit_max_sentiment_margin_list)})

            return stock_social_media_dict

        else:
            return np.asarray(over_night_diff_list), np.asarray(max_margin_list), \
                   np.asarray(abs_diff_list), np.asarray(per_change_list), \
                   np.asarray(abs_vol_list), np.asarray(vol_fluc_list), np.asarray(price_fluc_list), \
                   np.asarray(twitter_avg_sentiment_list), np.asarray(twitter_pos_per_list), \
                   np.asarray(twitter_neu_per_list), np.asarray(twitter_neg_per_list), \
                   np.asarray(twitter_num_posts_list), np.asarray(twitter_max_sentiment_margin_list), \
                   np.asarray(reddit_avg_sentiment_list), np.asarray(reddit_pos_per_list), \
                   np.asarray(reddit_neu_per_list), np.asarray(reddit_neg_per_list), \
                   np.asarray(reddit_num_posts_list), np.asarray(reddit_max_sentiment_margin_list), \
                   np.asarray(date_list)




    def plot_sentiment_stock_feature_grafics(self, twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                             reddit_feature=GlobalConfig.REDDIT_NUM_POS,
                                             stock_feature_min=-1, stock_feature_max=1):

        # get all features
        sentiment_stock_feature_dict = self.extract_sentiment_and_stock_features(return_as_dict=True)

        # rescale stock features to range values between -1 and 1
        over_night_diff_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.OVER_NIGHT_DIFF],
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        max_margin_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.MAX_MARGIN],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_diff_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.ABS_DIFFERENCE],
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        per_change_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.PER_CHANGE],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_vol_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.ABS_VOL],
                                                   lower_bound=stock_feature_min,
                                                   upper_bound=stock_feature_max)
        vol_fluc_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.VOL_FLUC],
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        price_fluc_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.PRICE_FLUC],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)

        # rescale social media feature lists
        twitter_feature_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[twitter_feature],
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        reddit_feature_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[reddit_feature],
                                                          lower_bound=stock_feature_min,
                                                          upper_bound=stock_feature_max)

        # create traces and layout for to plot figure
        twitter_sentiment_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                               y=twitter_feature_list_rescaled, mode='lines+markers',
                                             line=dict(color='blue', width=0.5),
                                             marker=dict(color='deepskyblue', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='blue', width=2)),
                                             name='twitter_sentiment_scores',
                                             text='Sentiment scores of tweets calculated through finbert model [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='blue'))
        reddit_sentiment_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                              y=reddit_feature_list_rescaled, mode='lines+markers',
                                             line=dict(color='orange', width=0.5),
                                             marker=dict(color='darkorange', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='orange', width=2)),
                                             name='reddit_sentiment_scores',
                                             text='Sentiment scores of reddit posts calculated through finbert model [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='orange'))
        over_night_diff_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                             y=over_night_diff_list_rescaled, mode='lines+markers',
                                             line=dict(color='grey', width=0.5),
                                             marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='circle',
                                                         line=dict(color='grey', width=2)),
                                             name=GlobalConfig.OVER_NIGHT_DIFF,
                                             text='Price difference from close (yesterday) to open (today) - rescaled to [-1; 1]',
                                             showlegend=True,
                                             hoverinfo='text',
                                             opacity=0.7,
                                             hoverlabel=dict(bgcolor='grey'))
        max_margin_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                        y=max_margin_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='square',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.MAX_MARGIN,
                                        text='Maximum price difference - rescaled to [-1; 1]',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        abs_diff_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                      y=abs_diff_list_rescaled, mode='lines+markers',
                                      line=dict(color='grey', width=0.5),
                                      marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='diamond',
                                                  line=dict(color='grey', width=2)),
                                      name=GlobalConfig.ABS_DIFFERENCE,
                                      text='Price difference between open and close today - rescaled to [-1; 1',
                                      showlegend=True,
                                      hoverinfo='text',
                                      opacity=0.7,
                                      hoverlabel=dict(bgcolor='grey'))
        per_change_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                        y=per_change_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='hexagram',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.PER_CHANGE,
                                        text='Percentage price change between open and close today - rescaled to [-1; 1] ',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        abs_vol_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                     y=abs_vol_list_rescaled, mode='lines+markers',
                                     line=dict(color='grey', width=0.5),
                                     marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='star',
                                                 line=dict(color='grey', width=2)),
                                     name=GlobalConfig.ABS_VOL,
                                     text='Absolute trading volume between open and close today - rescaled to [-1; 1] ',
                                     showlegend=True,
                                     hoverinfo='text',
                                     opacity=0.7,
                                     hoverlabel=dict(bgcolor='grey'))
        vol_fluc_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                      y=vol_fluc_list_rescaled, mode='lines+markers',
                                        line=dict(color='grey', width=0.5),
                                        marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='bowtie',
                                                    line=dict(color='grey', width=2)),
                                        name=GlobalConfig.VOL_FLUC,
                                        text='Volume standard deviation between open and close today - rescaled to [-1; 1] ',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='grey'))
        price_fluc_trace = go.Scattergl(x=sentiment_stock_feature_dict[GlobalConfig.DATE_LIST],
                                        y=price_fluc_list_rescaled, mode='lines+markers',
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
                                                     self.ticker, str(twitter_feature)+'_'+str(reddit_feature)+\
                                                     '_sentiment_scores_and_price_features.html'), auto_open=False)
        print(datetime.now(), ':', self.ticker, 'sentiment_scores_and_price_feature_plot created.')




    def plot_correlation_bar_chart(self, twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                   reddit_feature=GlobalConfig.REDDIT_NUM_POS,
                                   stock_feature_min=-1, stock_feature_max=1):

        # get all features
        sentiment_stock_feature_dict = self.extract_sentiment_and_stock_features(return_as_dict=True)

        # rescale stock features to range values between -1 and 1
        over_night_diff_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.OVER_NIGHT_DIFF],
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        max_margin_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.MAX_MARGIN],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_diff_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.ABS_DIFFERENCE],
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        per_change_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.PER_CHANGE],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)
        abs_vol_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.ABS_VOL],
                                                   lower_bound=stock_feature_min,
                                                   upper_bound=stock_feature_max)
        vol_fluc_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.VOL_FLUC],
                                                    lower_bound=stock_feature_min,
                                                    upper_bound=stock_feature_max)
        price_fluc_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[GlobalConfig.PRICE_FLUC],
                                                      lower_bound=stock_feature_min,
                                                      upper_bound=stock_feature_max)

        # # calculate stock feature correlations to reddit sentiment
        reddit_corrcoef_list = []
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], over_night_diff_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], max_margin_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], abs_diff_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], per_change_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], abs_vol_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], vol_fluc_list_rescaled)[0, 1])
        reddit_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[reddit_feature], price_fluc_list_rescaled)[0, 1])
        #
        # # calculate stock feature correlations to twitter sentiment
        twitter_corrcoef_list = []
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], over_night_diff_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], max_margin_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], abs_diff_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], per_change_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], abs_vol_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], vol_fluc_list_rescaled)[0, 1])
        twitter_corrcoef_list.append(np.corrcoef(sentiment_stock_feature_dict[twitter_feature], price_fluc_list_rescaled)[0, 1])

        stock_feature_list = [GlobalConfig.OVER_NIGHT_DIFF, GlobalConfig.MAX_MARGIN,
                              GlobalConfig.ABS_DIFFERENCE, GlobalConfig.PER_CHANGE,
                              GlobalConfig.ABS_VOL, GlobalConfig.VOL_FLUC, GlobalConfig.PRICE_FLUC]

        # create figure and layout
        fig = go.Figure(data=[
            go.Bar(name='Correlation with Twitter Sentiment',
                   x=stock_feature_list, y=twitter_corrcoef_list,
                   text=[str(round(corrcoef, 3)) for corrcoef in twitter_corrcoef_list], textposition="auto"),
            go.Bar(name='Correlation with Reddit Sentiment',
                   x=stock_feature_list, y=reddit_corrcoef_list,
                   text=[str(round(corrcoef, 3)) for corrcoef in reddit_corrcoef_list], textposition="auto")
        ])
        fig.update_layout(title=self.ticker+'<br>Correlation of Stock Features with Twitter and Reddit Sentiment',
                          xaxis=dict(title='Stock Features', titlefont_size=16, tickfont_size=14),
                          yaxis=dict(title='Correlation', titlefont_size=16, tickfont_size=14),
                          barmode='group', bargap=0.15, bargroupgap=0.05)
        po.plot(fig, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                           self.ticker, str(twitter_feature)+'_'+str(reddit_feature)+\
                                           '_stock_features_correlation_bar_chart.html'), auto_open=False)
        print(datetime.now(), ':', self.ticker, 'correlation_bar_chart_plot created.')



    def plot_correlation_heatmap(self, use_time_lag=False, time_lag_direction=GlobalConfig.TIME_LAG_RIGHT, num_days=1,
                                 rescale=False, rescale_lower_boud=-1, rescale_upper_bound=1):

        # extract sentiment and stock features
        over_night_diff_list, max_margin_list, abs_diff_list, \
        per_change_list, abs_vol_list, vol_fluc_list, price_fluc_list, \
        twitter_avg_sentiment_list, twitter_pos_per_list, twitter_neu_per_list, \
        twitter_neg_per_list, twitter_num_posts_list, twitter_max_sentiment_margin_list, \
        reddit_avg_sentiment_list, reddit_pos_per_list, reddit_neu_per_list, \
        reddit_neg_per_list, reddit_num_posts_list, reddit_max_sentiment_margin_list, \
        _ = self.extract_sentiment_and_stock_features()

        # lists of feature lists and feature names
        stock_feature_lists = [over_night_diff_list, max_margin_list, abs_diff_list,
                               per_change_list, abs_vol_list, vol_fluc_list, price_fluc_list]
        stock_feature_names_list = [GlobalConfig.OVER_NIGHT_DIFF, GlobalConfig.MAX_MARGIN,
                                    GlobalConfig.ABS_DIFFERENCE, GlobalConfig.PER_CHANGE,
                                    GlobalConfig.ABS_VOL, GlobalConfig.VOL_FLUC, GlobalConfig.PRICE_FLUC]
        twitter_feature_lists = [twitter_avg_sentiment_list, twitter_pos_per_list, twitter_neu_per_list,
                                 twitter_neg_per_list, twitter_num_posts_list, twitter_max_sentiment_margin_list]
        reddit_feature_lists = [reddit_avg_sentiment_list, reddit_pos_per_list, reddit_neu_per_list,
                                reddit_neg_per_list, reddit_num_posts_list, reddit_max_sentiment_margin_list]
        social_media_feature_names_list = [GlobalConfig.AVERAGE_SENTIMENT, GlobalConfig.POS_PER_SENTIMENT,
                                           GlobalConfig.NEU_PER_SENTIMENT, GlobalConfig.NEG_PER_SENTIMENT,
                                           GlobalConfig.NUM_POS, GlobalConfig.MAX_SENTIMENT_MARGIN]

        # twitter and reddit correlation matrix
        stock_twitter_corr_matrix = []
        stock_reddit_corr_matrix = []
        for stock_feature_list in stock_feature_lists:
            twitter_row = []
            reddit_row = []
            for twitter_feature_list, reddit_feature_list in zip(twitter_feature_lists, reddit_feature_lists):

                # initialize correlation coefficients
                corr_coef_twitter = None
                corr_coef_reddit = None

                if rescale:
                    stock_feature_list_rescaled = self.rescale_array(stock_feature_list,
                                                                     lower_bound=rescale_lower_boud,
                                                                     upper_bound=rescale_upper_bound)
                    twitter_feature_list_rescale = self.rescale_array(twitter_feature_list,
                                                                      lower_bound=rescale_lower_boud,
                                                                      upper_bound=rescale_upper_bound)
                    reddit_feature_list_rescale = self.rescale_array(reddit_feature_list,
                                                                     lower_bound=rescale_lower_boud,
                                                                     upper_bound=rescale_upper_bound)
                    if use_time_lag:
                        if time_lag_direction == GlobalConfig.TIME_LAG_RIGHT:
                            pass
                        elif time_lag_direction == GlobalConfig.TIME_LAG_LEFT:
                            pass
                        else:
                            print('Pass in valid time_lag_direction argument!')
                            sys.exit(0)
                    else:
                        corr_coef_twitter = np.corrcoef(stock_feature_list_rescaled, twitter_feature_list_rescale)[0, 1]
                        corr_coef_reddit = np.corrcoef(stock_feature_list_rescaled, reddit_feature_list_rescale)[0, 1]

                else:
                    if use_time_lag:
                        # time lag direction is right -> stock after social media
                        if time_lag_direction == GlobalConfig.TIME_LAG_RIGHT:
                            corr_coef_twitter = np.corrcoef(stock_feature_list[num_days:],
                                                            twitter_feature_list[:-num_days])[0, 1]
                            corr_coef_reddit = np.corrcoef(stock_feature_list[num_days:],
                                                           reddit_feature_list[:-num_days])[0, 1]
                        # time lag direction is left -> social media after stock
                        elif time_lag_direction == GlobalConfig.TIME_LAG_LEFT:
                            corr_coef_twitter = np.corrcoef(stock_feature_list[:-num_days],
                                                            twitter_feature_list[num_days:])[0, 1]
                            corr_coef_reddit = np.corrcoef(stock_feature_list[:-num_days],
                                                           reddit_feature_list[num_days:])[0, 1]
                        else:
                            print('Pass in valid time_lag_direction argument!')
                            sys.exit(0)
                    else:
                        corr_coef_twitter = np.corrcoef(stock_feature_list, twitter_feature_list)[0, 1]
                        corr_coef_reddit = np.corrcoef(stock_feature_list, reddit_feature_list)[0, 1]

                twitter_row.append(corr_coef_twitter)
                reddit_row.append(corr_coef_reddit)
            stock_twitter_corr_matrix.append(twitter_row)
            stock_reddit_corr_matrix.append(reddit_row)

        # create heatmaps
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        stock_twitter_hm_ax = sns.heatmap(data=stock_twitter_corr_matrix, cbar=True, cmap="YlGnBu",
                                          annot=True, annot_kws=dict(fontsize=7), linewidths=0.05, square=True,
                                          xticklabels=social_media_feature_names_list,
                                          yticklabels=stock_feature_names_list, ax=ax1)
        stock_twitter_hm_ax.set_title('Stock & Twitter Features \n Correlation Heatmap for '+self.ticker, fontsize=10)
        stock_twitter_hm_ax.set_xticklabels(stock_twitter_hm_ax.get_xticklabels(), rotation=15, fontsize=5)
        stock_twitter_hm_ax.set_yticklabels(stock_twitter_hm_ax.get_yticklabels(), rotation=0, fontsize=5)
        stock_twitter_hm_ax.set_xlabel('Twitter Features', fontsize=8)
        stock_twitter_hm_ax.set_ylabel('Stock Features', fontsize=8)

        stock_reddit_hm_ax = sns.heatmap(data=stock_reddit_corr_matrix, cbar=True, cmap="YlGnBu",
                                          annot=True, annot_kws=dict(fontsize=7), linewidths=0.05, square=True,
                                          xticklabels=social_media_feature_names_list,
                                          yticklabels=stock_feature_names_list, ax=ax2)
        stock_reddit_hm_ax.set_title('Stock & Reddit Features \n Correlation Heatmap for '+self.ticker, fontsize=10)
        stock_reddit_hm_ax.set_xticklabels(stock_reddit_hm_ax.get_xticklabels(), rotation=15, fontsize=5)
        stock_reddit_hm_ax.set_yticklabels(stock_reddit_hm_ax.get_yticklabels(), rotation=0, fontsize=5)
        stock_reddit_hm_ax.set_xlabel('Reddit Features', fontsize=8)
        stock_reddit_hm_ax.set_ylabel('Stock Features', fontsize=8)

        # save figure
        if use_time_lag:
            filename = 'Stock_Social_Media_Correlation_Heatmap_time_lag_'+ \
                       str(time_lag_direction)+'_'+str(num_days)+'.png'
        else:
            filename = 'Stock_Social_Media_Correlation_Heatmap_no_time_lag.png'

        plt.savefig(os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                 self.ticker, filename),
                    dpi=400)

        print(datetime.now(), ':', self.ticker, 'correlation_heatmap created.')




    def plot_distribution_comparison(self, perform_t_test=True,
                                     stock_feature=GlobalConfig.MAX_MARGIN,
                                     twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                     reddit_feature=GlobalConfig.REDDIT_NUM_POS,
                                     stock_feature_min=0, stock_feature_max=1):

        # get all features
        sentiment_stock_feature_dict = self.extract_sentiment_and_stock_features(return_as_dict=True)

        # rescale stock features to range values between -1 and 1
        stock_feature_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[stock_feature],
                                                         lower_bound=stock_feature_min,
                                                         upper_bound=stock_feature_max)

        # rescale social media feature lists
        twitter_feature_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[twitter_feature],
                                                           lower_bound=stock_feature_min,
                                                           upper_bound=stock_feature_max)
        reddit_feature_list_rescaled = self.rescale_array(sentiment_stock_feature_dict[reddit_feature],
                                                          lower_bound=stock_feature_min,
                                                          upper_bound=stock_feature_max)

        # perform Kolmogorov Smirnov test to see if data comes from the same distribution
        statistic_twitter, pvalue_twitter = KolmogorovSmirnov(obs_array_1=stock_feature_list_rescaled,
                                                              obs_array_2=twitter_feature_list_rescaled).perform_test()
        statistic_reddit, pvalue_reddit = KolmogorovSmirnov(obs_array_1=stock_feature_list_rescaled,
                                                            obs_array_2=reddit_feature_list_rescaled).perform_test()

        # plot histograms
        plt.hist(stock_feature_list_rescaled, bins=50, alpha=0.5, color='grey', label=stock_feature)
        plt.hist(twitter_feature_list_rescaled, bins=50, alpha=0.5, color='blue', label=twitter_feature)
        plt.hist(reddit_feature_list_rescaled, bins=50, alpha=0.5, color='red', label=reddit_feature)
        plt.title('Distribution Comparison\npvalue_twitter: '+str(round(pvalue_twitter, 2))+\
                  ' - pvalue_reddit: '+str(round(pvalue_reddit, 2)))
        plt.xlabel('Bins (features rescaled)')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH, self.ticker,
                    str(stock_feature)+'_'+str(twitter_feature)+'_'+str(reddit_feature)+ \
                    '_distribution_comparison.png'))

        print(datetime.now(), ':', self.ticker, 'distribution_comparison_plot created.')






def prepare_stock_dict_from_json(dict_item):

    prepared_dict = dict()
    for date_key, date_dict in dict_item.items():

        # convert str key to datetime.date key
        prepared_dict.update({datetime.strptime(date_key, '%Y-%m-%d').date(): dict()})

        for feature_key, feature_value in date_dict.items():

            # convert str value to datetime.time value
            if feature_key == GlobalConfig.START_TIME or feature_key == GlobalConfig.END_TIME:
                prepared_dict.get(datetime.strptime(date_key, '%Y-%m-%d').date()).\
                                  update({feature_key: datetime.strptime(feature_value, '%H:%M:%S').time()})
            else:
                prepared_dict.get(datetime.strptime(date_key, '%Y-%m-%d').date()).\
                                  update({feature_key: feature_value})
    return prepared_dict


def prepare_social_media_dict_from_json(dict_item):

    prepared_dict = dict()
    for date_key, date_dict in dict_item.items():

        # convert str key to datetime.date key
        prepared_dict.update({datetime.strptime(date_key, '%Y-%m-%d').date(): dict()})

        for feature_key, feature_value in date_dict.items():

            # convert str value to datetime.time value
            prepared_dict.get(datetime.strptime(date_key, '%Y-%m-%d').date()).\
                              update({feature_key: feature_value})
    return prepared_dict