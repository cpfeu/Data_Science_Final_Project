import os
import plotly.offline as po
import plotly.graph_objs as go
from datetime import datetime
from configurations.global_config import GlobalConfig


class StockVisualizer:

    def __init__(self, single_stock_recording_list):
        self.single_stock_recording_list = single_stock_recording_list


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


