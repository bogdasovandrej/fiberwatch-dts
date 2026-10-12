import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent if here.name == 'tools' else here))
import fiberwatch as fw

data_dir, run, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
REACH = 2 * fw.RADIUS
rows, forecast_rows = [], []
for path in sorted(data_dir.glob('[1-6].csv')):
    matrix = fw.remove_common_mode(fw.read_source(path)[0])[0]
    names = matrix.columns
    grid = pd.read_csv(run / f'scan_{path.stem}.csv.gz')
    grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
    grid['cnn2d_notify'] = grid['cnn2d_notify'].astype(str).eq('True')
    notified = fw.merge_events(grid, 'cnn2d_notify', list(grid.coordinate.drop_duplicates().sort_values()))
    table = fw.forecast_changes(matrix)
    forecasts = fw.merge_events(table, 'forecast', list(table.coordinate.drop_duplicates().sort_values()))
    for frame in (notified, forecasts):
        if len(frame):
            frame['first_alert'] = pd.to_datetime(frame['first_alert'])
            frame['from_pos'] = names.get_indexer(frame.coordinate_from, method='nearest') - REACH
            frame['to_pos'] = names.get_indexer(frame.coordinate_to, method='nearest') + REACH
    for e in notified.itertuples():
        before = forecasts[(forecasts.from_pos <= e.to_pos) & (forecasts.to_pos >= e.from_pos)
                           & (forecasts.first_alert < e.first_alert)
                           & (forecasts.first_alert >= e.first_alert - pd.Timedelta(minutes=60))] if len(forecasts) else forecasts
        lead = (e.first_alert - before.first_alert.min()).total_seconds() / 60 if len(before) else 0.
        rows.append({'file': path.name, 'lead_min': lead})
    for f in forecasts.itertuples():
        after = notified[(notified.from_pos <= f.to_pos) & (notified.to_pos >= f.from_pos)
                         & (notified.first_alert > f.first_alert)
                         & (notified.first_alert <= f.first_alert + pd.Timedelta(minutes=60))] if len(notified) else notified
        forecast_rows.append({'file': path.name, 'confirmed': len(after) > 0})
    print(path.name, len(notified), len(forecasts), flush=True)
events, forecasts = pd.DataFrame(rows), pd.DataFrame(forecast_rows)
summary = pd.DataFrame({
    'notified_events': events.groupby('file').size(),
    'warned_10_min_before': events.assign(x=events.lead_min.ge(10)).groupby('file').x.mean().round(3),
    'warned_20_min_before': events.assign(x=events.lead_min.ge(20)).groupby('file').x.mean().round(3),
    'forecasts': forecasts.groupby('file').size(),
    'forecasts_confirmed_within_60_min': forecasts.groupby('file').confirmed.mean().round(3),
})
total = {'notified_events': len(events), 'warned_10_min_before': round(events.lead_min.ge(10).mean(), 3),
         'warned_20_min_before': round(events.lead_min.ge(20).mean(), 3), 'forecasts': len(forecasts),
         'forecasts_confirmed_within_60_min': round(forecasts.confirmed.mean(), 3)}
summary.loc['all'] = total
summary.to_csv(output, encoding='utf-8-sig')
warned = events.loc[events.lead_min.gt(0), 'lead_min']
print(summary.to_string())
print('median lead, min:', warned.median() if len(warned) else 0)
