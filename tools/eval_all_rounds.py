import json
import math
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent if here.name == 'tools' else here))
import fiberwatch as fw

run, output = Path(sys.argv[1]), Path(sys.argv[2])
ROUNDS = [(1, 'review/cases/cases_private.csv', 'review/answers_final', 'review/review_corrections.csv'),
          (2, 'review/cases2/cases_private.csv', 'review/answers_round2', None),
          (3, 'review/cases3/cases_private.csv', 'review/answers_round3', None),
          (4, 'review/cases4/cases_private.csv', 'review/answers_round4', 'review/review_corrections4.csv')]
MAIN = ['amplitude_rule', 'random_forest', 'cnn2d']


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def stratum(frame):
    a, b = frame['cnn2d_notify'], frame['amplitude_rule_notify']
    other = frame['random_forest_notify']
    return pd.Series('N', index=frame.index).mask(other, 'D').mask(b & ~a, 'B').mask(a & ~b, 'C').mask(a & b, 'A')


cases = []
for number, cases_path, answers_dir, corrections in ROUNDS:
    frame = pd.read_csv(Path(cases_path))
    answers = {}
    for path in (Path(answers_dir)).rglob('*.json'):
        body = json.loads(path.read_text(encoding='utf-8'))
        body = body.get('data', body)
        answers[body.get('case_id', path.stem)] = body.get('decision')
    frame['decision'] = frame.case_id.map(answers)
    if corrections:
        for row in pd.read_csv(Path(corrections)).itertuples():
            frame.loc[frame.case_id.eq(row.case_id), 'decision'] = row.reviewed_decision
    cases.append(frame.assign(round=number))
cases = pd.concat(cases, ignore_index=True)
cases = cases.loc[cases.decision.isin(['event', 'normal', 'glitch']) & cases.group.ne('recheck')].reset_index(drop=True)
cases['human_event'] = cases.decision.eq('event')

grids, flags = {}, []
counts = {m: {} for m in MAIN}
for name in sorted(cases.file.unique()):
    grid = pd.read_csv(run / f'scan_{name.replace(".csv", "")}.csv.gz')
    grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
    for model in fw.MODELS:
        grid[model + '_notify'] = grid[model + '_notify'].astype(str).eq('True')
    grid['stratum'] = stratum(grid)
    grids[name] = grid
    cutoff = grid.end.min() + (grid.end.max() - grid.end.min()) * fw.COMMISSIONING_SHARE
    late = grid.loc[grid.end > cutoff]
    for model in MAIN:
        for key, size in late.loc[late[model + '_notify']].stratum.value_counts().items():
            counts[model][key] = counts[model].get(key, 0) + int(size)
for case in cases.itertuples():
    grid = grids[case.file]
    end = pd.Timestamp(case.end)
    if grid.end.dt.tz is not None and end.tz is None:
        end = end.tz_localize(grid.end.dt.tz)
    at = grid.loc[grid.coordinate.eq(grid.coordinate.iloc[(grid.coordinate - case.coordinate).abs().argsort().iloc[0]])]
    row = at.iloc[(at.end - end).abs().argsort().iloc[0]]
    flags.append({'stratum': row.stratum, **{m: bool(row[m + '_notify']) for m in MAIN}})
cases = cases.join(pd.DataFrame(flags))

share = cases.groupby('stratum').human_event.agg(['sum', 'size'])
share['p'] = share['sum'] / share['size']
print('слои (по всей разметке):')
print(share.to_string())
union = {}
for model in MAIN:
    for key, size in counts[model].items():
        union[key] = max(union.get(key, 0), size)
found_union = sum(size * share.p.get(key, 0) for key, size in union.items())
rows = []
for model in MAIN:
    windows = sum(counts[model].values())
    expected = sum(size * share.p.get(key, float('nan')) for key, size in counts[model].items())
    said = cases[model]
    tp, fp = int((said & cases.human_event).sum()), int((said & ~cases.human_event).sum())
    fn = int((~said & cases.human_event).sum())
    low, high = wilson(tp, tp + fp)
    rows.append({'model': fw.NAMES[model], 'windows_after_commissioning': windows,
                 'estimated_precision_full_record': expected / windows if windows else float('nan'),
                 'estimated_share_of_found_events': expected / found_union,
                 'labelled_TP': tp, 'labelled_FP': fp, 'labelled_FN': fn,
                 'labelled_precision': tp / max(tp + fp, 1), 'ci_low': low, 'ci_high': high,
                 'labelled_recall': tp / max(tp + fn, 1)})
table = pd.DataFrame(rows)
table.to_csv(output, index=False, encoding='utf-8-sig')
cases.to_csv(output.with_name(output.stem + '_cases.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(f'размечено случаев: {len(cases)}, событий {int(cases.human_event.sum())}')
print(table.round(3).to_string(index=False))
