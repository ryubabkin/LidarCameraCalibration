import pandas as pd
import os


def convert_time(
        data: pd.DataFrame,
        date: str,
        tzone: str = 'US/Eastern'
) -> pd.DataFrame:
    data['timestamp'] = date + '_' + data['Timestamps'].astype(str)
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d%m%Y_%H%M%S.%f')
    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzone).dt.tz_localize(None).map(pd.Timestamp.timestamp)
    data.drop(columns=['Timestamps'], inplace=True)
    data['gnss_timestamp'] = data['timestamp']
    return data


def merge_gnss(
        association: pd.DataFrame,
        gnss_file: str,
        tzone: str = 'US/Eastern'
) -> pd.DataFrame:
    gnss_date = os.path.basename(gnss_file).split('.')[0][:8]
    gnss_data = pd.read_csv(gnss_file)
    gnss_data = convert_time(
        data=gnss_data,
        date=gnss_date,
        tzone=tzone
    )
    mix = pd.merge_asof(
        left=association,
        right=gnss_data,
        left_on="timestamp",
        right_on="timestamp",
        direction="nearest",
        allow_exact_matches=True
    )
    return mix
