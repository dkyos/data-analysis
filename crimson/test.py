#!/usr/bin/env python

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import date
#from flatten_json import flatten
from tqdm import tnrange as trange
from time import sleep

class CrimsonHexagonClient(object):
    """Interacts with the Crimson Hexagon API"""

    def __init__(self, username, password, monitor_id):
        self.username = username
        self.password = password
        self.monitor_id = monitor_id
        self.base = 'https://api.crimsonhexagon.com/api/monitor'
        self.session = requests.Session()
        self.ratelimit_refresh = 60
        self._auth()

    def _auth(self):
        """Authenticates a user using their username and password through the authenticate endpoint."""
        url = 'https://forsight.crimsonhexagon.com/api/authenticate?'

        payload = {
            'username': self.username,
            'password': self.password
        }

        r = self.session.get(url, params=payload)
        j_result = r.json()
        #print(j_result)

        if j_result['status'] == 'error':
            print('-- Not Authenticated --')
        else:
            # j_result['status'] == 'success'
            self.auth_token = j_result["auth"]
            print('-- Authenticated --')

        return

    def make_endpoint(self, endpoint):
        return '{}/{}?'.format(self.base, endpoint)

    def get_data_from_endpoint(self, from_, to_, endpoint):
        """Hits the designated endpoint (volume/posts) for a specified time period.
        The ratelimit is burned through ASAP and then backed off for one minute.
        """
        endpoint = self.make_endpoint(endpoint)
        #print (endpoint)

        from_, to_ = str(from_), str(to_)
        payload = {
            'auth': self.auth_token,
            'id': self.monitor_id,
            'start': from_,
            'end': to_,
            'aggregatebyday': 'true',
            'uselocaltime': 'false'
        }

        r = self.session.get(endpoint, params=payload)
        self.last_response = r

        ratelimit_remaining = r.headers['X-RateLimit-Remaining']

        # If the header is empty or 0 then wait for a ratelimit refresh.
        if (not ratelimit_remaining) or (float(ratelimit_remaining) < 1):
            print('Waiting for ratelimit refresh...')
            sleep(self.ratelimit_refresh)

        return r

    def plot_volume(self, r_volume):
        """Plots a time-series chart with two axes to show the daily and cumulative
        document count.
        """
        # Convert r to df, fix datetime, add cumulative sum.
        df_volume = pd.DataFrame(r_volume.json()['volumes'])
        df_volume['startDate'] = pd.to_datetime(df_volume['startDate'])
        df_volume['endDate'] = pd.to_datetime(df_volume['endDate'])
        df_volume['cumulative_sum'] = df_volume['numberOfDocuments'].cumsum()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        df_volume['numberOfDocuments'].plot(ax=ax1, style='b-')
        df_volume['cumulative_sum'].plot(ax=ax2, style='r-')

        print(df_volume)

        ax1.set_ylabel('Number of Documents')
        ax2.set_ylabel('Cumulative Sum')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=2)

        plt.show()

        return

    def make_data_pipeline(self, from_, to_):
        """Combines the functionsin this class to make a robust pipeline, that 
        loops through each day in a time period. Data is returned as a dataframe.
        """

        # Get the volume over time data.
        r_volume = self.get_data_from_endpoint(from_, to_, 'dayandtime')
        #print(r_volume.json())
        i#print('There are approximately {} documents.'.format(r_volume.json()['volumes'][0]['numberOfDocuments']))
        self.plot_volume(r_volume)

        return

if __name__ == "__main__":
    # Credentials.
    username = 'xxxx'
    password = 'xxxxx'

    # Monitor id - taken from URL on website.
    monitor_id = 'xxxxx'

    # Instantiate client.
    crimson_api = CrimsonHexagonClient(username, password, monitor_id)

    from_ = date(2018, 1, 1)
    to_   = date(2018, 5, 30)

    # Combine class functions into a typical workflow.
    crimson_api.make_data_pipeline(from_, to_)

