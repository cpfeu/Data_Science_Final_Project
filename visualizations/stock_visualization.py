import os
import sys
import plotly.offline as po
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from datetime import datetime
from configurations.global_config import GlobalConfig


class StockVisualizer:

    def __init__(self, single_stock_recording_list, stock_feature_dict=None):
        self.single_stock_recording_list = single_stock_recording_list
        self.stock_feature_dict = stock_feature_dict

    def plot_all_in_one_chart(self):
        time_stamp_list = []
        open_list = []
        close_list = []
        high_list = []
        low_list = []
        volume_list = []
        for single_stock_recording in self.single_stock_recording_list:
            time_stamp_list.append(single_stock_recording.time_stamp)
            open_list.append(single_stock_recording.open)
            high_list.append(single_stock_recording.high)
            close_list.append(single_stock_recording.close)
            volume_list.append(single_stock_recording.volume)
            low_list.append(single_stock_recording.low)

        # create traces
        open_trace = go.Scattergl(x=time_stamp_list, y=open_list,
                                  mode='lines', marker=dict(color=GlobalConfig.OPEN_COLOR),
                                  name=GlobalConfig.STOCK_PARAM_OPEN)
        close_trace = go.Scattergl(x=time_stamp_list, y=close_list,
                                   mode='lines', marker=dict(color=GlobalConfig.CLOSE_COLOR),
                                   name=GlobalConfig.STOCK_PARAM_CLOSE)
        high_trace = go.Scattergl(x=time_stamp_list, y=high_list,
                                  mode='lines', marker=dict(color=GlobalConfig.HIGH_COLOR),
                                  name=GlobalConfig.STOCK_PARAM_HIGH)
        low_trace = go.Scattergl(x=time_stamp_list, y=low_list,
                                 mode='lines', marker=dict(color=GlobalConfig.LOW_COLOR),
                                 name=GlobalConfig.STOCK_PARAM_LOW)
        volume_trace = go.Scattergl(x=time_stamp_list, y=volume_list,
                                    mode='lines', marker=dict(color=GlobalConfig.VOLUME_COLOR),
                                    name=GlobalConfig.STOCK_PARAM_VOLUME)

        # design layout
        layout = dict(title=self.single_stock_recording_list[0].name,
                      xaxis=dict(title='Time',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      yaxis=dict(title='Price [in $]',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                      hovermode='closest')


        # create and plot figure
        figure = dict(data=[open_trace, close_trace, high_trace, low_trace, volume_trace], layout=layout)
        po.plot(figure, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                              self.single_stock_recording_list[0].name, "All_in_one_plot.html"), auto_open=False)
        print(datetime.now(), ':', self.single_stock_recording_list[0].name, 'all_in_one_plot created.')




    def plot_stock_features(self):
        # check is stock_feature_dict argument was passed
        if self.stock_feature_dict is None:
            print('Passed stock_feature_dict is of type None.')
            sys.exit(0)


        # description parameter
        ticker = self.single_stock_recording_list[0].name
        start_time = list(self.stock_feature_dict.values())[0].get(GlobalConfig.START_TIME)
        end_time = list(self.stock_feature_dict.values())[0].get(GlobalConfig.END_TIME)


        # extract feature lists
        date_list = []
        abs_traded_volume_list = []
        vol_fluc_list = []
        max_margin_list = []
        over_night_diff_list = []
        abs_diff_list = []
        per_change_list = []
        price_fluc_list = []
        for key, dict_value in self.stock_feature_dict.items():
            date_list.append(key)
            abs_traded_volume_list.append(dict_value.get(GlobalConfig.ABS_VOL))
            vol_fluc_list.append(dict_value.get(GlobalConfig.VOL_FLUC))
            max_margin_list.append(dict_value.get(GlobalConfig.MAX_MARGIN))
            over_night_diff_list.append(dict_value.get(GlobalConfig.OVER_NIGHT_DIFF))
            abs_diff_list.append(dict_value.get(GlobalConfig.ABS_DIFFERENCE))
            per_change_list.append(dict_value.get(GlobalConfig.PER_CHANGE))
            price_fluc_list.append(dict_value.get(GlobalConfig.PRICE_FLUC))


        # create traces and layout for volume features and plot figure
        abs_traded_volume_trace = go.Scattergl(x=date_list, y=abs_traded_volume_list, mode='lines+markers',
                                               line=dict(color='grey', width=0.5),
                                               marker=dict(color='darkgrey', size=8, opacity=0.7, symbol='circle-open',
                                                           line=dict(color='grey', width=2)),
                                               name=GlobalConfig.ABS_VOL,
                                               text='Absolute trading volume between '+
                                                    str(start_time)+' and '+str(end_time),
                                               showlegend=True,
                                               hoverinfo='text',
                                               opacity=0.7,
                                               hoverlabel=dict(bgcolor='darkgrey'))
        vol_fluc_trace = go.Scattergl(x=date_list, y=vol_fluc_list, mode='lines+markers',
                                      line=dict(color='grey', width=0.5),
                                      marker=dict(color='slategrey', size=8, opacity=0.7, symbol='triangle-right',
                                                  line=dict(color='grey', width=2)),
                                      name=GlobalConfig.VOL_FLUC,
                                      text='Volume standard deviation between '+str(start_time)+' and '+str(end_time),
                                      showlegend=True,
                                      hoverinfo='text',
                                      opacity=0.7,
                                      hoverlabel=dict(bgcolor='slategrey'))
        layout_plot_1 = dict(title=ticker + '<br>Daily Volume Features between '+str(start_time)+' and '+str(end_time),
                             xaxis=dict(title='Date',
                                        titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                             yaxis=dict(title='Volume Feature Units',
                                        titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                             hovermode='closest')
        figure_plot_1 = dict(data=[abs_traded_volume_trace, vol_fluc_trace], layout=layout_plot_1)
        po.plot(figure_plot_1, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                              ticker, "volume_features.html"), auto_open=False)
        print(datetime.now(), ':', ticker, 'volume_feature_plot created.')


        # create traces and layout for remaining features and plot figure
        max_margin_trace = go.Scattergl(x=date_list, y=max_margin_list, mode='lines+markers',
                                        line=dict(color='green', width=0.5),
                                        marker=dict(color='darkgreen', size=8, opacity=0.7, symbol='square',
                                                    line=dict(color='green', width=2)),
                                        name=GlobalConfig.MAX_MARGIN,
                                        text='Maximum price difference between '+
                                             str(start_time)+' and '+str(end_time)+' in $',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='darkgreen'))
        over_night_diff_trace = go.Scattergl(x=date_list, y=over_night_diff_list, mode='lines+markers',
                                        line=dict(color='blue', width=0.5),
                                        marker=dict(color='deepskyblue', size=8, opacity=0.7, symbol='circle',
                                                    line=dict(color='blue', width=2)),
                                        name=GlobalConfig.OVER_NIGHT_DIFF,
                                        text='Price difference from close (yesterday) to open (today) in $',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='blue'))
        abs_diff_trace = go.Scattergl(x=date_list, y=abs_diff_list, mode='lines+markers',
                                        line=dict(color='mediumblue', width=0.5),
                                        marker=dict(color='midnightblue', size=8, opacity=0.7, symbol='diamond',
                                                    line=dict(color='mediumblue', width=2)),
                                        name=GlobalConfig.ABS_DIFFERENCE,
                                        text='Price difference between open at '+
                                             str(start_time)+' and close at '+str(end_time)+' in $',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='mediumblue'))
        per_change_trace = go.Scattergl(x=date_list, y=per_change_list, mode='lines+markers',
                                        line=dict(color='red', width=0.5),
                                        marker=dict(color='darkred', size=8, opacity=0.7, symbol='hexagram',
                                                    line=dict(color='red', width=2)),
                                        name=GlobalConfig.PER_CHANGE,
                                        text='Percentage price change between open at '+
                                             str(start_time)+' and close at '+str(end_time)+' in %',
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='red'))
        price_fluc_trace = go.Scattergl(x=date_list, y=price_fluc_list, mode='lines+markers',
                                        line=dict(color='orange', width=0.5),
                                        marker=dict(color='darkorange', size=8, opacity=0.7, symbol='pentagon',
                                                    line=dict(color='orange', width=2)),
                                        name=GlobalConfig.PRICE_FLUC,
                                        text='Price standard deviation between '+
                                             str(start_time)+' and '+str(end_time),
                                        showlegend=True,
                                        hoverinfo='text',
                                        opacity=0.7,
                                        hoverlabel=dict(bgcolor='orange'))
        layout_plot_2 = dict(title=ticker + '<br>Daily Stock Price Features between '+str(start_time)+' and '+str(end_time),
                             xaxis=dict(title='Date',
                                        titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                             yaxis=dict(title='Stock Price Feature Units',
                                        titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
                             hovermode='closest')
        figure_plot_1 = dict(data=[max_margin_trace, over_night_diff_trace,
                                   abs_diff_trace, per_change_trace, price_fluc_trace], layout=layout_plot_2)
        po.plot(figure_plot_1, filename=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                              ticker, "price_features.html"), auto_open=False)
        print(datetime.now(), ':', ticker, 'price_feature_plot created.')



    def plot_stock_feature_histogranms(self):
        # check is stock_feature_dict argument was passed
        if self.stock_feature_dict is None:
            print('Passed stock_feature_dict is of type None.')
            sys.exit(0)

        # description parameter
        ticker = self.single_stock_recording_list[0].name
        start_date = list(self.stock_feature_dict.keys())[0]
        end_date = list(self.stock_feature_dict.keys())[-1]

        # extract list of values for each feature
        over_night_diff_list = []
        max_margin_list = []
        abs_diff_list = []
        per_change_list = []
        abs_vol_list = []
        vol_fluc_list = []
        price_fluc_list = []
        for date_key, feature_dict in self.stock_feature_dict.items():
            if (date_key.weekday() == 5 or date_key.weekday() == 6):
                continue
            else:
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

        # create histogram plot for features
        figure, ((f11, f12, f13, f14), (f21, f22, f23, f24)) = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))
        figure.suptitle(ticker + ' Daily Feature Histograms \nTime: '+str(start_date)+' until '+str(end_date), size=16)

        f11.hist(over_night_diff_list, bins=30, color="blue", rwidth=0.95)
        # f11.set_xlabel("Feature bin", fontsize=8)
        # f11.set_ylabel("Quantity", fontsize=8)
        f11.set_title(GlobalConfig.OVER_NIGHT_DIFF + '\nfeature histogram')

        f12.hist(max_margin_list, bins=30, color="blue", rwidth=0.95)
        # f12.set_xlabel("Feature bin", fontsize=8)
        # f12.set_ylabel("Quantity", fontsize=8)
        f12.set_title(GlobalConfig.MAX_MARGIN + '\nfeature histogram')

        f13.hist(abs_diff_list, bins=30, color="blue", rwidth=0.95)
        # f13.set_xlabel("Feature bin", fontsize=8)
        # f13.set_ylabel("Quantity", fontsize=8)
        f13.set_title(GlobalConfig.ABS_DIFFERENCE + '\nfeature histogram')

        f14.hist(per_change_list, bins=30, color="blue", rwidth=0.95)
        # f14.set_xlabel("Feature bin", fontsize=8)
        # f14.set_ylabel("Quantity", fontsize=8)
        f14.set_title(GlobalConfig.PER_CHANGE + '\nfeature histogram')

        f21.hist(abs_vol_list, bins=30, color="blue", rwidth=0.95)
        # f21.set_xlabel("Feature bin", fontsize=8)
        f21.ticklabel_format(style='sci', axis='x')
        # f21.set_ylabel("Quantity", fontsize=8)
        f21.set_title(GlobalConfig.ABS_VOL + '\nfeature histogram')

        f22.hist(vol_fluc_list, bins=30, color="blue", rwidth=0.95)
        # f22.set_xlabel("Feature bin", fontsize=8)
        f22.ticklabel_format(style='sci', axis='x')
        # f22.set_ylabel("Quantity", fontsize=8)
        f22.set_title(GlobalConfig.VOL_FLUC + '\nfeature histogram')

        f23.hist(price_fluc_list, bins=30, color="blue", rwidth=0.95)
        # f23.set_xlabel("Feature bin", fontsize=8)
        # f23.set_ylabel("Quantity", fontsize=8)
        f23.set_title(GlobalConfig.PRICE_FLUC + '\nfeature histogram')

        plt.savefig(os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH,
                                 ticker, "feature_histograms.png"), dpi=400)
        plt.close()

        print(datetime.now(), ':', ticker, 'feature_histograms plot created.')


