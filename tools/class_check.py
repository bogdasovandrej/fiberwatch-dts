import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent if here.name == 'tools' else here))
import fiberwatch as fw

SETS = {
    'A': ['short', 'short', 'long', 'long', 'wide', 'wide', 'persistent'],
    'B': ['periodic', 'repeat', 'pulsating', 'same_time', 'same_time', 'same_time'],
}
ROUNDS = {f'{k}{n}': kinds for n in (1, 2) for k, kinds in SETS.items()}
CLEARANCE_M = 15.
SEPARATION = pd.Timedelta(minutes=50)


def trapezoid(minutes, onset, length, edge):
    t = np.asarray(minutes) - onset
    rise = np.clip(t / edge + .5, 0, 1)
    fall = np.clip((length - t) / edge + .5, 0, 1)
    return np.minimum(rise, fall)


def quiet_sites(matrix):
    valid = matrix.loc[:, matrix.notna().all() & matrix.gt(fw.INVALID_BELOW).all()]
    smooth = valid.rolling('20min', min_periods=3).median()
    quiet = (smooth.max() - smooth.min()).lt(2.5).reindex(matrix.columns, fill_value=False)
    names = matrix.columns.to_numpy()
    ok = []
    for i, c in enumerate(names):
        around = (names >= c - CLEARANCE_M) & (names <= c + CLEARANCE_M)
        if around.sum() > 3 and quiet.to_numpy()[around].all():
            ok.append(c)
    return ok


def free_moment(times, rng, taken, before, after, pulses, tries=4000):
    gaps = times.to_series().diff().gt(fw.MAX_GAP).to_numpy()
    gap_end = times[gaps]
    gap_start = times[np.flatnonzero(gaps) - 1]
    for _ in range(tries):
        t = times[rng.randrange(len(times))]
        low, high = t - before, t + after
        if low < times[0] or high > times[-1]:
            continue
        if ((gap_start < high) & (gap_end > low)).any():
            continue
        moments = [t + pd.Timedelta(minutes=p) for p in pulses]
        if all(abs(m - u) >= SEPARATION for m in moments for u in taken):
            return t
    return None


def plan(kind, rng, step_min):
    size = rng.uniform(5., 10.) * rng.choice([-1, 1])
    local = rng.uniform(1.5, 3.)
    if kind == 'short':
        length = rng.uniform(max(2 * step_min, 1.), 6.)
        return dict(size=size, fwhm=local, pulses=[0.], length=length, edge=min(.5, length / 4))
    if kind in ('long', 'wide'):
        return dict(size=size, fwhm=rng.uniform(8., 12.) if kind == 'wide' else local, pulses=[0.],
                    length=rng.uniform(30., 180.), edge=2.)
    if kind == 'persistent':
        return dict(size=size, fwhm=local, pulses=[0.], length=None, edge=2.)
    if kind == 'periodic':
        period = rng.uniform(120., 160.)
        return dict(size=size, fwhm=local, pulses=[k * period for k in range(4)], length=15., edge=2., period=period)
    if kind == 'repeat':
        return dict(size=size, fwhm=local, pulses=[0., rng.uniform(150., 300.)], length=20., edge=2.)
    if kind == 'pulsating':
        period = rng.uniform(15., 25.)
        return dict(size=size, fwhm=local, pulses=[k * period for k in range(4)], length=5., edge=1., merged=True)
    return dict(size=size, fwhm=local, pulses=[0.], length=30., edge=2.)


def check(data_dir, model_dir, output, files):
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    fitted = fw.load_final(model_dir)
    for f_number, name in enumerate(files):
        raw = fw.remove_common_mode(fw.read_source(data_dir / name)[0])[0]
        step_min = raw.index.to_series().diff().median() / pd.Timedelta(minutes=1)
        sites_all = quiet_sites(raw)
        for round_name, kinds in ROUNDS.items():
            rng = random.Random(1000 * sum(map(ord, name)) + sum(ord(ch) * 31 ** i for i, ch in enumerate(round_name)))
            matrix = raw.copy()
            minutes = (matrix.index - matrix.index[0]) / pd.Timedelta(minutes=1)
            free = list(sites_all)
            rng.shuffle(free)
            chosen, taken, scenarios = [], [], []
            group_onset = None
            for kind in kinds:
                site = next((c for c in free if all(abs(c - u) > 2 * CLEARANCE_M for u in chosen)), None)
                if site is None:
                    continue
                spec = plan(kind, rng, step_min)
                span = pd.Timedelta(minutes=(spec['pulses'][-1] + (spec['length'] or 60.) + 60.))
                if kind == 'same_time' and group_onset is not None:
                    onset = group_onset
                else:
                    onset = free_moment(matrix.index, rng, taken, pd.Timedelta(minutes=130), span, spec['pulses'])
                    if onset is None:
                        continue
                    if kind == 'same_time':
                        group_onset = onset
                    taken += [onset + pd.Timedelta(minutes=p) for p in spec['pulses']]
                chosen.append(site)
                sigma = spec['fwhm'] / 2.355
                columns = matrix.columns[(matrix.columns > site - 4 * sigma) & (matrix.columns < site + 4 * sigma)]
                weights = np.exp(-(columns.to_numpy() - site) ** 2 / (2 * sigma ** 2))
                start_min = (onset - matrix.index[0]) / pd.Timedelta(minutes=1)
                shape = np.zeros(len(matrix))
                for p in spec['pulses']:
                    if spec['length'] is None:
                        shape = np.maximum(shape, np.clip((np.asarray(minutes) - start_min) / spec['edge'] + .5, 0, 1))
                    else:
                        shape = np.maximum(shape, trapezoid(minutes, start_min + p, spec['length'], spec['edge']))
                matrix.loc[:, columns] = matrix[columns].to_numpy() + spec['size'] * np.outer(shape, weights)
                for k, p in enumerate(spec['pulses']):
                    if spec.get('merged') and k > 0:
                        break
                    scenarios.append({'file': name, 'round': round_name, 'kind': kind, 'site_m': site, 'pulse': k,
                                      'onset': onset + pd.Timedelta(minutes=p), 'size_c': spec['size'],
                                      'true_length_min': spec['length'] if not spec.get('merged') else None,
                                      'true_fwhm_m': spec['fwhm'], 'true_period_min': spec.get('period'),
                                      'pulses': len(spec['pulses'])})
            if not scenarios:
                continue
            grid, _ = fw.scan(matrix, fitted, max_columns=None)
            fw.calibrate(grid, fitted)
            events = fw.merge_events(grid, 'cnn2d_notify', list(grid.coordinate.drop_duplicates().sort_values()))
            table = pd.DataFrame(scenarios)
            near = np.zeros(len(events), dtype=bool)
            for s in table.itertuples() if len(events) else []:
                reach = max(2., s.true_fwhm_m)
                near |= (events.coordinate_to >= s.site_m - reach).to_numpy() & (events.coordinate_from <= s.site_m + reach).to_numpy()
            events = fw.classify_events(events.loc[near].reset_index(drop=True), matrix) if near.any() else events.iloc[:0]
            if len(events):
                events.assign(round=round_name).to_csv(output / f'events_{name}_{round_name}.csv', index=False, encoding='utf-8-sig')
            for s in table.itertuples():
                reach = max(2., s.true_fwhm_m)
                last = s.onset + pd.Timedelta(minutes=(5. if pd.isna(s.true_length_min) else s.true_length_min) + 50)
                match = events.loc[(events.coordinate_to >= s.site_m - reach) & (events.coordinate_from <= s.site_m + reach)
                                   & (pd.to_datetime(events.first_alert) >= s.onset - pd.Timedelta(minutes=10))
                                   & (pd.to_datetime(events.first_alert) <= last)] if len(events) else events
                row = s._asdict()
                row.pop('Index')
                row['detected'] = len(match) > 0
                if len(match):
                    e = match.sort_values('first_alert').iloc[0]
                    row.update({k: e[k] for k in ['first_alert', 'description', 'change_from_background_c', 'duration_min',
                                                  'width_m', 'bursts', 'duration_class', 'extent_class', 'repeats_at_place',
                                                  'period_min', 'places_at_same_time', 'classes']})
                rows.append(row)
            print(name, round_name, 'scenarios', len(table), 'detected', sum(r['detected'] for r in rows[-len(table):]), flush=True)

    return pd.DataFrame(rows)


def summarize(result, output):
    expect_duration = {'short': 'кратковременное', 'persistent': 'не вернулось к фону до конца данных'}
    result['expected_duration'] = result.kind.map(expect_duration).fillna('продолжительное')
    result['expected_extent'] = np.where(result.kind.eq('wide'), 'обширное', 'локальное')
    got = result.loc[result.detected].copy()
    checks = {
        'длительность: класс верный': (got.kind.ne('pulsating'), got.duration_class.eq(got.expected_duration)),
        'протяжённость: класс верный': (got.kind.ne('pulsating'), got.extent_class.eq(got.expected_extent)),
        'повтор на месте: найден': (got.kind.isin(['periodic', 'repeat']), got.repeats_at_place.ge(2)),
        'периодичность: найдена': (got.kind.eq('periodic'), got.period_min.notna()),
        'одновременность: найдена': (got.kind.eq('same_time'), got.places_at_same_time.ge(3)),
        'пульсация: найдена': (got.kind.eq('pulsating'), got.bursts.ge(3)),
        'нет ложного повтора': (got.kind.isin(['short', 'long', 'wide', 'persistent', 'same_time', 'pulsating']),
                                got.repeats_at_place.eq(1)),
        'нет ложной одновременности': (got.kind.ne('same_time'), got.places_at_same_time.eq(1)),
        'нет ложной пульсации': (got.kind.ne('pulsating'), got.bursts.lt(3)),
        'тип (нагрев/охлаждение) верный': (got.kind.ne('pulsating'),
                                           got.description.eq(np.where(got.size_c > 0, 'нагрев', 'охлаждение'))),
    }
    summary = [{'проверка': 'обнаружено событий', 'верно': int(result.detected.sum()), 'всего': len(result)}]
    for title, (mask, ok) in checks.items():
        summary.append({'проверка': title, 'верно': int(ok[mask].sum()), 'всего': int(mask.sum())})
    summary = pd.DataFrame(summary)
    summary['доля'] = (summary['верно'] / summary['всего']).round(3)
    timed = got.loc[got.kind.isin(['short', 'long', 'wide', 'periodic', 'repeat', 'same_time'])]
    error_d = (timed.duration_min - timed.true_length_min) / timed.true_length_min
    error_w = got.width_m - got.true_fwhm_m
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'class_check_cases.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output / 'class_check.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print('длительность: медиана относительной ошибки', round(float(error_d.abs().median()), 3),
          '| ширина: медиана ошибки, м', round(float(error_w.abs().median()), 2))
    print(result.loc[result.detected].groupby('kind')[['duration_min', 'true_length_min', 'width_m', 'true_fwhm_m']].median().round(1))


if sys.argv[1] == '--summary':
    summarize(pd.concat([pd.read_csv(f) for f in sys.argv[3:]], ignore_index=True), Path(sys.argv[2]))
else:
    files = sys.argv[4].split(',') if len(sys.argv) > 4 else ['2.csv', '3.csv', '4.csv', '5.csv', '6.csv']
    summarize(check(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), files), Path(sys.argv[3]))
