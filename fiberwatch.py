import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score

WINDOW = pd.Timedelta(minutes=40)
BIN = pd.Timedelta(minutes=2)
N_BINS = 20
MAX_GAP = pd.Timedelta(minutes=10)
INVALID_BELOW = -273.15
REFERENCE_SPAN = pd.Timedelta(minutes=6)
NOISE_FACTOR = 4.0
CHANNELS = ['median_relative_c', 'min_relative_c', 'max_relative_c', 'observed', 'noise_c']
EVENT_C = 3.0
GREY_MAX_C = 2.8
COMMON_MODE_MIN_C = 0.5
BACKGROUND_SPAN = pd.Timedelta(minutes=80)
COMMISSIONING_SHARE = .3
RADIUS = 4
SHORT_MIN = 10
WIDE_M = 5.
SAME_PLACE_M = 1.
SAME_TIME = pd.Timedelta(minutes=10)
PERIODIC_SPREAD = .25
MODELS = ['amplitude_rule', 'random_forest', 'cnn2d']
NAMES = {'amplitude_rule': 'Правило: отклонение медианы', 'random_forest': 'Случайный лес',
         'cnn2d': 'CNN время × координата'}
CNN_WIDTHS = (16, 32, 64)
MAX_EPOCHS = 150
POS_WEIGHT = 1.0
NOISE_AUGMENT_MAX = 1.0


def set_event_c(value):
    global EVENT_C, GREY_MAX_C
    EVENT_C, GREY_MAX_C = float(value), max(.5, float(value) - .2)


def read_source(path):
    source = pd.read_csv(path)
    if len(source[['dts_id', 'channel_id']].drop_duplicates()) != 1:
        raise ValueError(f'{path.name}: в файле смешаны приборы или каналы')
    times = pd.DatetimeIndex(pd.to_datetime(source['timestamp'], format='ISO8601'))
    profiles = [pd.DataFrame(json.loads(text), columns=['coordinate', 'temperature_c'])
                .set_index('coordinate')['temperature_c'] for text in source['points']]
    matrix = pd.concat(profiles, axis=1, keys=range(len(profiles))).T
    matrix.index = times
    matrix = matrix.sort_index(kind='stable')
    duplicates = int(matrix.index.duplicated(keep='last').sum())
    matrix = matrix.loc[~matrix.index.duplicated(keep='last')]
    matrix = matrix.sort_index(axis=1)
    step = matrix.index.to_series().diff().dt.total_seconds()
    invalid = matrix.lt(INVALID_BELOW)
    quality = {
        'file': path.name, 'dts_id': str(source.dts_id.iloc[0]), 'channel_id': int(source.channel_id.iloc[0]),
        'timezone': str(matrix.index.tz) if matrix.index.tz is not None else 'не указан',
        'profiles': len(matrix), 'coordinates': matrix.shape[1], 'duplicate_profiles_dropped': duplicates,
        'start': str(matrix.index[0]), 'end': str(matrix.index[-1]),
        'median_step_s': float(step.median()) if step.notna().any() else float('nan'), 'max_gap_min': float(step.max() / 60),
        'gaps_over_10min': int(step.gt(MAX_GAP.total_seconds()).sum()),
        'short_profiles': int(matrix.isna().any(axis=1).sum()),
        'invalid_values': int(invalid.sum().sum()),
        'always_invalid_coordinates': int(invalid.all(axis=0).sum()),
        'min_valid_c': float(matrix.where(~invalid).min().min()),
        'max_c': float(matrix.max().max()),
    }
    return matrix, quality


def remove_common_mode(matrix):
    valid = matrix.loc[:, matrix.notna().all() & matrix.ge(INVALID_BELOW).all()]
    deviation = valid - valid.rolling('20min', min_periods=3, closed='left').median()
    common = deviation.median(axis=1).fillna(0.)
    common = common.where(common.abs().ge(COMMON_MODE_MIN_C), 0.)
    corrected = matrix.sub(common, axis=0).where(matrix.ge(INVALID_BELOW), matrix)
    return corrected, common.loc[common.ne(0.)]


def window_ends(matrix):
    times = matrix.index.to_series()
    block = times.diff().gt(MAX_GAP).cumsum()
    block_start = times.groupby(block).transform('min')
    return times.loc[times - block_start >= WINDOW - pd.Timedelta(minutes=2)].index


def represent(rows, end):
    start = end - WINDOW
    rows = rows.loc[(rows.index > start) & (rows.index <= end)]
    if len(rows) < 3 or rows.index.to_series().diff().gt(MAX_GAP).any():
        return None
    position = (rows.index - start) / BIN
    bin_number = [int(-((-value) // 1)) for value in position]
    reference = reference_level(rows, start)
    relative = rows - reference
    grouped = relative.groupby(pd.Series(bin_number, index=relative.index))
    bins = list(range(1, N_BINS + 1))
    observed = grouped.count().T.reindex(columns=bins).fillna(0).gt(0)
    if observed.iloc[0].sum() < 3:
        return None
    parts = {}
    for name, frame in [('median', grouped.median()), ('min', grouped.min()), ('max', grouped.max())]:
        parts[name] = frame.T.reindex(columns=bins).ffill(axis=1).fillna(0.)
    parts['observed'] = observed.astype(float)
    parts['reference'] = reference
    parts['noise'] = noise_sigma(rows)
    return parts


def reference_level(rows, start):
    early = rows.loc[rows.index <= start + REFERENCE_SPAN]
    return (early if len(early) >= 3 else rows.iloc[:3]).median()


def noise_sigma(rows):
    return rows.diff().abs().median() * 1.4826 / math.sqrt(2)


def to_tensor(parts):
    stacked = [torch.tensor(parts[name].to_numpy(dtype='float32')) / (10. if name != 'observed' else 1.)
               for name in ['median', 'min', 'max', 'observed']]
    noise = torch.tensor(parts['noise'].to_numpy(dtype='float32')).unsqueeze(1).expand(-1, N_BINS) / 10.
    return torch.stack(stacked + [noise], dim=1)


def features(parts):
    observed = parts['observed'].astype(bool)
    median = parts['median'].where(observed)
    low, high = parts['min'].where(observed), parts['max'].where(observed)
    long = median.stack().rename('median').reset_index()
    long.columns = ['window', 'bin', 'median']
    by_window = long.groupby('window', sort=False)
    long['rate'] = by_window['median'].diff() / (by_window['bin'].diff() * 2.)
    first = by_window['median'].first()
    last = by_window['median'].last()
    spread = high - low
    table = pd.DataFrame({
        'range_c': high.max(axis=1) - low.min(axis=1),
        'median_range_c': median.max(axis=1) - median.min(axis=1),
        'std_c': median.std(axis=1, ddof=0),
        'net_change_c': last - first,
        'absolute_net_change_c': (last - first).abs(),
        'max_absolute_rate_c_per_min': long.groupby('window', sort=False)['rate'].apply(lambda r: r.abs().max()),
        'mean_absolute_rate_c_per_min': long.groupby('window', sort=False)['rate'].apply(lambda r: r.abs().mean()),
        'mean_bin_spread_c': spread.mean(axis=1),
        'max_bin_spread_c': spread.max(axis=1),
        'observed_fraction': parts['observed'].mean(axis=1),
        'noise_c': parts['noise'],
    })
    return table.loc[parts['median'].index].fillna(0.)


def amplitude(parts):
    level = parts['median'].abs().max(axis=1)
    spike = pd.concat([parts['max'].max(axis=1), -parts['min'].min(axis=1)], axis=1).max(axis=1)
    return level, spike


def subset(parts, columns):
    return {k: v.loc[columns] for k, v in parts.items()}


def neighbours(names, position):
    return [names[i] if 0 <= i < len(names) else None for i in range(position - RADIUS, position + RADIUS + 1)]


def patches(parts, neighbour_lists):
    full = to_tensor(parts)
    position = {c: i for i, c in enumerate(parts['median'].index)}
    index = torch.tensor([[position.get(c, 0) for c in row] for row in neighbour_lists])
    present = torch.tensor([[c in position for c in row] for row in neighbour_lists], dtype=torch.float32)
    out = full[index] * present[:, :, None, None]
    return out.permute(0, 2, 1, 3).contiguous()


def injected_shape(kind, minutes, onset, size, rng):
    t = (torch.tensor(minutes, dtype=torch.float64) - onset).clamp(min=0.)
    started = torch.tensor(minutes, dtype=torch.float64) >= onset
    if kind in ('step', 'small'):
        tau = rng.uniform(.2, 5.)
        shape = 1 - torch.exp(-t / tau)
    elif kind == 'ramp':
        shape = (t / rng.uniform(5., 25.)).clamp(max=1.)
    elif kind == 'pulse':
        rise, decay = rng.uniform(.3, 3.), rng.uniform(3., 20.)
        shape = (1 - torch.exp(-t / rise)) * torch.exp(-t / decay)
        shape = shape / shape.max().clamp(min=1e-9)
    elif kind == 'spike':
        shape = torch.zeros_like(t)
        first = int(started.nonzero()[0]) if started.any() else len(t) - 1
        shape[first:first + rng.choice([1, 1, 2])] = 1.
    elif kind == 'glitch':
        shape = torch.zeros_like(t)
        first = int(started.nonzero()[0]) if started.any() else len(t) - 1
        shape[first] = 1.
    else:
        raise ValueError(kind)
    return (size * shape * started).tolist()


def make_examples(matrix, quality, ends, columns_per_end, rng, excluded, purpose, augment_noise=False, keep_raw=False):
    names = list(matrix.columns)
    position = {c: i for i, c in enumerate(names)}
    xs, x2s, feature_tables, passports, raws = [], [], [], [], []
    for end in ends:
        start = end - WINDOW
        rows = matrix.loc[(matrix.index > start) & (matrix.index <= end)]
        valid = rows.notna().all() & rows.ge(INVALID_BELOW).all()
        blocked = set()
        for zone in excluded:
            if end > zone['start'] and start < zone['end']:
                blocked.update(zone['columns'])
        candidates = [i for i, c in enumerate(names) if valid[c] and c not in blocked]
        rng.shuffle(candidates)
        targets, taken = [], []
        for i in candidates:
            if all(abs(i - j) > 2 * RADIUS + 1 for j in taken):
                taken.append(i)
                targets.append(names[i])
                if len(targets) == columns_per_end:
                    break
        if len(targets) < columns_per_end:
            continue
        lists = {t: neighbours(names, position[t]) for t in targets}
        local = sorted({c for row in lists.values() for c in row if c is not None and valid[c]}, key=position.get)
        natural = represent(rows[local], end)
        if natural is None:
            continue
        natural_level, natural_spike = amplitude(subset(natural, targets))
        quiet = [t for t in targets if natural_level[t] < min(1.5, EVENT_C / 2) and natural_spike[t] < EVENT_C + 1.]
        minutes = ((rows.index - start) / pd.Timedelta(minutes=1)).tolist()
        delta = pd.DataFrame(0., index=rows.index, columns=local)
        added = pd.DataFrame(0., index=rows.index, columns=local)
        meta = []
        for target in quiet:
            members = [c for c in lists[target] if c is not None and valid[c]]
            noise_added = 0.
            if augment_noise and rng.random() < .5:
                noise_added = rng.uniform(.1, NOISE_AUGMENT_MAX)
                generator = torch.Generator().manual_seed(rng.randrange(2 ** 31))
                noise = torch.randn(len(rows), len(members), generator=generator, dtype=torch.float64) * noise_added
                added[members] = pd.DataFrame(noise.tolist(), index=rows.index, columns=members)
            sigma = float(noise_sigma((rows[target] + added[target]).to_frame()).iloc[0])
            event_threshold = max(EVENT_C, NOISE_FACTOR * sigma)
            kind = rng.choices(['none', 'small', 'step', 'ramp', 'pulse', 'spike', 'glitch'],
                               [26, 17, 13, 11, 11, 10, 12])[0]
            realized, width = 0., 0.
            if kind != 'none':
                low, high = (.3, GREY_MAX_C) if kind == 'small' else (EVENT_C, 12.)
                if kind not in ('small', 'glitch'):
                    low = min(event_threshold, high - .5)
                size = math.exp(rng.uniform(math.log(low), math.log(high))) * rng.choice([-1, 1])
                onset = rng.uniform(8., 39.5)
                values = pd.Series(injected_shape(kind, minutes, onset, size, rng), index=rows.index)
                realized = float((values - reference_level(values.to_frame(), start).iloc[0]).abs().max())
                if kind != 'glitch':
                    low_w, high_w = (.8, 3.) if kind == 'spike' else (.7, 15.)
                    width = math.exp(rng.uniform(math.log(low_w), math.log(high_w)))
                for c in members:
                    offset = position[c] - position[target]
                    weight = 1. if offset == 0 else (math.exp(-offset ** 2 / (2 * width ** 2)) if width > 0 else 0.)
                    delta[c] = delta[c] + values * weight
            label = 0 if kind == 'glitch' else int(realized >= event_threshold)
            meta.append({'target': target, 'column': target, 'position': position[target], 'kind': kind,
                         'realized_c': realized, 'spatial_width': width, 'label': label,
                         'noise_added_c': noise_added, 'noise_c': sigma, 'event_threshold_c': event_threshold})
        meta = pd.DataFrame(meta)
        if meta.empty:
            continue
        meta = meta.loc[meta.label.notna()]
        if meta.empty:
            continue
        kept = meta.target.tolist()
        injected = rows[local] + delta + added
        parts = represent(injected, end)
        centre = subset(parts, kept)
        xs.append(to_tensor(centre))
        x2s.append(patches(parts, [lists[t] for t in kept]))
        feature_tables.append(features(centre).reset_index(drop=True))
        level, spike = amplitude(centre)
        meta = meta.assign(file=quality['file'], device=quality['dts_id'], end=end, purpose=purpose,
                           level_c=level.loc[kept].to_list(), spike_c=spike.loc[kept].to_list(),
                           natural_level_c=natural_level.loc[kept].to_list())
        meta['label'] = meta.label.astype(int)
        passports.append(meta.drop(columns='target'))
        if keep_raw:
            raws += [injected[t] for t in kept]
    result = (torch.cat(xs), torch.cat(x2s), pd.concat(feature_tables, ignore_index=True),
              pd.concat(passports, ignore_index=True))
    return result + (raws,) if keep_raw else result


class SpatialCNN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv2d(len(CHANNELS), width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.output = nn.Linear(width * 3, 1)

    def forward(self, inputs):
        patterns = self.filters(inputs)
        centre = patterns[:, :, RADIUS, :]
        summary = torch.cat([patterns.mean(dim=(2, 3)), patterns.amax(dim=(2, 3)), centre.amax(dim=2)], dim=1)
        return self.output(summary).squeeze(1)


def train_cnn(x_train, y_train, x_val, y_val, width, seed, factory=SpatialCNN, max_epochs=None, patience=10):
    max_epochs = max_epochs or MAX_EPOCHS
    torch.manual_seed(seed)
    model = factory(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history, waited = float('inf'), None, [], 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for batch in order.split(256):
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_function(model(x_train), y_train))
            val_loss = float(loss_function(model(x_val), y_val))
        history.append({'epoch': epoch, 'width': width, 'model': factory.__name__,
                        'training_loss': train_loss, 'validation_loss': val_loss})
        if val_loss < best - 1e-4:
            best, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, history


def scores(name, model, x, table, x2=None):
    if name == 'amplitude_rule':
        return pd.Series(table['level_c'].values)
    if name == 'random_forest':
        return pd.Series(model.predict_proba(table[FEATURES])[:, 1])
    with torch.no_grad():
        return pd.Series(torch.sigmoid(model(x2)).tolist())


FEATURES = ['range_c', 'median_range_c', 'std_c', 'net_change_c', 'absolute_net_change_c',
            'max_absolute_rate_c_per_min', 'mean_absolute_rate_c_per_min', 'mean_bin_spread_c',
            'max_bin_spread_c', 'observed_fraction', 'noise_c']


def choose_threshold(labels, values):
    candidates = sorted(set(values.quantile([i / 200 for i in range(1, 200)]).round(6)))
    best = max(candidates, key=lambda t: (fbeta_score(labels, values.ge(t), beta=2, zero_division=0), t))
    return float(best)


def fit_all(x, x2, table, passport, seed):
    rng = random.Random(seed)
    slots = sorted(passport[['file', 'end']].drop_duplicates().itertuples(index=False, name=None))
    validation_slots = set(rng.sample(slots, max(1, len(slots) // 5)))
    is_val = pd.Series([(f, e) in validation_slots for f, e in zip(passport.file, passport.end)])
    train_idx = is_val.index[~is_val].tolist()
    val_idx = is_val.index[is_val].tolist()
    y = torch.tensor(passport.label.to_list(), dtype=torch.float32)
    labels_val = passport.label.iloc[val_idx].reset_index(drop=True)
    fitted, log = {}, []
    for name in MODELS:
        model, extra = None, {}
        if name == 'random_forest':
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)
            model.fit(table.loc[train_idx, FEATURES], passport.label.iloc[train_idx])
        elif name == 'cnn2d':
            inputs, factory = x2, SpatialCNN
            trials = [train_cnn(inputs[train_idx], y[train_idx], inputs[val_idx], y[val_idx], width, seed, factory)
                      for width in CNN_WIDTHS]
            model, best_loss, history = min(trials, key=lambda trial: trial[1])
            extra = {'width': model.filters[0].out_channels, 'validation_loss': best_loss,
                     'epochs_run': len(history), 'parameters': sum(p.numel() for p in model.parameters()),
                     'history': [row for trial in trials for row in trial[2]]}
        values = scores(name, model, x[val_idx], table.loc[val_idx].reset_index(drop=True), x2[val_idx])
        threshold = choose_threshold(labels_val, values)
        fitted[name] = (model, threshold)
        log.append({'model': name, 'threshold': threshold, **{k: v for k, v in extra.items() if k != 'history'}})
        if 'history' in extra:
            fitted.setdefault('_cnn_history', []).extend(extra['history'])
    return fitted, log


def evaluate(labels, flags):
    tn, fp, fn, tp = confusion_matrix(labels, flags, labels=[0, 1]).ravel()
    return {'n': len(labels), 'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn),
            'recall': recall_score(labels, flags, zero_division=0),
            'precision': precision_score(labels, flags, zero_division=0),
            'f2': fbeta_score(labels, flags, beta=2, zero_division=0)}


def load_cards(annotations_path, cases_dir):
    annotations = pd.read_csv(annotations_path).set_index('sample_id')
    cards = {}
    for sample_id in annotations.index:
        card = pd.read_csv(cases_dir / f'{sample_id}.csv', index_col=0)
        card.index = pd.to_datetime(card.index, format='ISO8601')
        cards[sample_id] = card[['temperature_c']]
    return annotations, cards


def card_inputs(cards, sample_ids, annotations, matrices):
    tensors, tensors2, tables = [], [], []
    for sample_id in sample_ids:
        matrix = matrices[annotations.loc[sample_id, 'source_file']]
        names = list(matrix.columns)
        centre = matrix.columns.get_indexer([annotations.loc[sample_id, 'coordinate']], method='nearest')[0]
        rows = matrix.loc[cards[sample_id].index]
        columns = [c for c in neighbours(names, centre) if c is not None]
        local = rows[columns].loc[:, rows[columns].notna().all() & rows[columns].ge(INVALID_BELOW).all()]
        parts = represent(local, rows.index[-1])
        target = names[centre]
        own = subset(parts, [target])
        tensors.append(to_tensor(own))
        tensors2.append(patches(parts, [neighbours(names, centre)]))
        table = features(own)
        table.index = [sample_id]
        table['level_c'], table['spike_c'] = [v.values for v in amplitude(own)]
        tables.append(table)
    return torch.cat(tensors), torch.cat(tensors2), pd.concat(tables)


def exclusion_zones(annotations, matrices):
    zones = {}
    for sample_id, row in annotations.iterrows():
        matrix = matrices[row.source_file]
        position = matrix.columns.get_indexer([row.coordinate], method='nearest')[0]
        start, end = pd.Timestamp(row.start), pd.Timestamp(row.end)
        if matrix.index.tz is None:
            start, end = start.tz_localize(None), end.tz_localize(None)
        zones.setdefault(row.source_file, []).append(
            {'start': start, 'end': end, 'columns': set(matrix.columns[max(0, position - 15):position + 16])})
    return zones


def scan(matrix, fitted, every_minutes=10, max_columns=1500):
    usable = matrix.columns[matrix.lt(INVALID_BELOW).sum().eq(0)]
    stride = 1 if max_columns is None else max(1, math.ceil(len(usable) / max_columns))
    columns = list(usable[::stride])
    names = list(matrix.columns)
    position = {c: i for i, c in enumerate(names)}
    ends, last = [], None
    for end in window_ends(matrix):
        if last is None or end - last >= pd.Timedelta(minutes=every_minutes):
            ends.append(end)
            last = end
    rows_out = []
    for end in ends:
        rows = matrix.loc[(matrix.index > end - WINDOW) & (matrix.index <= end), usable]
        rows = rows.loc[:, rows.notna().all() & rows.ge(INVALID_BELOW).all()]
        if rows.empty:
            continue
        parts = represent(rows, end)
        if parts is None:
            continue
        targets = [c for c in columns if c in parts['median'].index]
        own = subset(parts, targets)
        table = features(own)
        level, spike = amplitude(own)
        table['level_c'] = level
        x = to_tensor(own)
        x2 = patches(parts, [neighbours(names, position[c]) for c in targets])
        parts = own
        history = matrix.loc[(matrix.index > end - WINDOW - BACKGROUND_SPAN) & (matrix.index <= end - WINDOW), targets]
        background = history.where(history > INVALID_BELOW).median()
        from_background = (rows[targets].rolling(2).median() - background).abs().max()
        frame = pd.DataFrame({'end': end, 'coordinate': table.index, 'position': matrix.columns.get_indexer(table.index),
                              'level_c': level.values, 'spike_c': spike.values,
                              'limit_c': notify_limit(table).values,
                              'background_change_c': from_background.reindex(table.index).values,
                              'signed_level_c': parts['median'].apply(lambda r: r.loc[r.abs().idxmax()], axis=1).values})
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, x, table.reset_index(drop=True), x2)
            frame[name + '_score'] = values.values
            frame[name] = values.ge(threshold).values
        rows_out.append(frame)
    if not rows_out:
        return pd.DataFrame(), stride
    return pd.concat(rows_out, ignore_index=True), stride


def notify_limit(table):
    return (table['noise_c'] * NOISE_FACTOR).clip(lower=EVENT_C)


def measured_c(table):
    return table[['level_c', 'spike_c']].max(axis=1)


def rule_passes(grid):
    from_background = grid.background_change_c.fillna(measured_c(grid))
    return measured_c(grid).ge(grid.limit_c) & from_background.ge(grid.limit_c)


def calibrate(grid, fitted):
    passes = rule_passes(grid)
    for name in MODELS:
        grid[name + '_notify'] = grid[name] & passes
        grid[name + '_candidate'] = grid[name] & ~passes


def merge_events(grid, model, columns):
    flagged = grid.loc[grid[model]].copy()
    if flagged.empty:
        return pd.DataFrame()
    position = {c: i for i, c in enumerate(columns)}
    by_fibre = 'position' in flagged
    reach = 2 * RADIUS if by_fibre else 2
    moments = [pd.Timestamp(t) for t in sorted(grid.end.unique())]
    times = {t: i for i, t in enumerate(moments)}
    linked = {i: [j - i for j, u in enumerate(moments) if abs(u - t) <= WINDOW] for i, t in enumerate(moments)}
    flagged['ci'] = flagged.position if by_fibre else flagged.coordinate.map(position)
    flagged['ti'] = flagged.end.map(times)
    cells = {(r.ti, r.ci): i for i, r in enumerate(flagged.itertuples())}
    label = {}
    current = 0
    for cell in cells:
        if cell in label:
            continue
        current += 1
        stack = [cell]
        label[cell] = current
        while stack:
            ti, ci = stack.pop()
            for dt in linked[ti]:
                for dc in range(-reach, reach + 1):
                    neighbour = (ti + dt, ci + dc)
                    if neighbour in cells and neighbour not in label:
                        label[neighbour] = current
                        stack.append(neighbour)
    flagged['event_id'] = [label[(r.ti, r.ci)] for r in flagged.itertuples()]
    peak = flagged.loc[flagged.groupby('event_id').level_c.idxmax()].set_index('event_id')
    events = flagged.groupby('event_id').agg(first_alert=('end', 'min'), last_alert=('end', 'max'),
                                             coordinate_from=('coordinate', 'min'), coordinate_to=('coordinate', 'max'),
                                             windows=('end', 'size'))
    events['peak_change_c'] = peak.signed_level_c.round(2)
    events['peak_spike_c'] = peak.spike_c.round(2)
    events['peak_coordinate'] = peak.coordinate
    if 'eta_min' in flagged:
        events['eta_min'] = flagged.groupby('event_id').eta_min.min().round(0)
    kind = events.peak_change_c.gt(0).map({True: 'нагрев', False: 'охлаждение'})
    events['description'] = kind.where(events.peak_change_c.abs().ge(1.5), 'краткий выброс')
    events['status'] = 'требуется проверка'
    return events.reset_index()


def filled(flags):
    out = flags.copy()
    out[1:-1] |= flags[:-2] & flags[2:]
    return out


def run_bounds(flags, i, breaks):
    low = high = i
    while low > 0 and flags[low - 1] and not breaks[low]:
        low -= 1
    while high < len(flags) - 1 and flags[high + 1] and not breaks[high + 1]:
        high += 1
    return low, high


def count_bursts(series, high, low, rest=pd.Timedelta(minutes=2)):
    bursts, state, since = 0, 'out', None
    for t, v in series.items():
        if state == 'out':
            if v >= high:
                bursts, state = bursts + 1, 'in'
        elif v >= low:
            state = 'in'
        elif state == 'in':
            state, since = 'leaving', t
        elif t - since >= rest:
            state = 'out'
    return bursts


def event_shape(valid, event, step, spacing):
    start, last = pd.Timestamp(event.first_alert) - WINDOW, pd.Timestamp(event.last_alert)
    history = valid.loc[(valid.index > start - BACKGROUND_SPAN) & (valid.index <= start)]
    if len(history) < 3:
        history = valid.loc[(valid.index > start) & (valid.index <= start + REFERENCE_SPAN)]
    background = history.median()
    inside = valid.columns[(valid.columns >= event.coordinate_from - 1e-6) & (valid.columns <= event.coordinate_to + 1e-6)]
    change = (valid.loc[(valid.index > start) & (valid.index <= last), inside] - background[inside]).stack()
    if change.empty:
        return {}
    moment, place = change.abs().idxmax()
    peak = change.loc[(moment, place)]
    sign = 1. if peak > 0 else -1.
    half = abs(peak) / 2
    series = sign * (valid[place] - background[place])
    series = series.loc[series.index > start - BACKGROUND_SPAN]
    smooth = series.rolling('2min', center=True, min_periods=1).median()
    times = series.index
    breaks = np.r_[False, np.diff(times.values) > np.timedelta64(MAX_GAP)]
    flags = smooth.fillna(-np.inf).to_numpy() >= half
    flags[times.get_loc(moment)] = True
    low, high = run_bounds(filled(flags), times.get_loc(moment), breaks)
    span = (times > start) & (times <= last)
    noise = float(noise_sigma(series.loc[span].to_frame()).iloc[0])
    bursts = count_bursts(smooth.loc[span], half, half / 2) if half / 2 >= 2 * noise else 1
    profile = sign * (valid.loc[moment] - background)
    left, right = run_bounds(filled(profile.fillna(-np.inf).to_numpy() >= half), valid.columns.get_loc(place),
                             np.zeros(len(profile), dtype=bool))
    return {'start_time': times[low], 'peak_time': moment, 'peak_place_m': place,
            'change_from_background_c': round(float(peak), 2),
            'duration_min': round((times[high] - times[low] + step) / pd.Timedelta(minutes=1), 1),
            'returned': bool(high < len(times) - 1 and not breaks[high + 1]),
            'width_m': round(float(valid.columns[right] - valid.columns[left]) + spacing, 2),
            'extent_from_m': valid.columns[left], 'extent_to_m': valid.columns[right], 'bursts': bursts}


def groups(linked):
    label = np.full(len(linked), -1)
    for i in range(len(linked)):
        if label[i] >= 0:
            continue
        label[i] = i
        stack = [i]
        while stack:
            for k in np.flatnonzero(linked[stack.pop()]):
                if label[k] < 0:
                    label[k] = i
                    stack.append(k)
    return label


def plural(n, forms):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def number(value):
    text = f'{value:.1f}'.replace('.', ',')
    return text[:-2] if text.endswith(',0') else text


def lasting(minutes):
    if minutes < 1:
        return 'меньше минуты'
    return f'{minutes:.0f} мин' if minutes < 90 else f'{number(minutes / 60)} ч'


def operator_message(e):
    t = pd.Timestamp(e.first_alert)
    offset = None if t.tzinfo is None else t.utcoffset() / pd.Timedelta(hours=1)
    zone = '' if offset is None else (' UTC' if offset == 0 else f' UTC{offset:+g}')
    low, high = (e.extent_from_m, e.extent_to_m) if pd.notna(e.extent_from_m) else (e.coordinate_from, e.coordinate_to)
    digits = 0 if float(low).is_integer() and float(high).is_integer() else 1
    low, high = (f'{v:.{digits}f}'.replace('.', ',') for v in (low, high))
    place = f'{low} м' if low == high else f'{low}–{high} м'
    change = e.change_from_background_c if pd.notna(e.change_from_background_c) else e.peak_change_c
    parts = [f'{e.description} на {number(abs(change))} °C']
    if pd.notna(e.duration_min):
        parts.append(f'длилось {lasting(e.duration_min)}' if e.returned else 'держится до конца записи')
    if pd.notna(e.period_min):
        parts.append(f'повторяется здесь примерно каждые {lasting(e.period_min)}')
    elif e.repeats_at_place > 1:
        parts.append(f'на этом месте {e.repeats_at_place} {plural(e.repeats_at_place, ("раз", "раза", "раз"))} за запись')
    if e.places_at_same_time > 1:
        k = e.places_at_same_time - 1
        parts.append(f'одновременно ещё на {k} {"участке" if k % 10 == 1 and k % 100 != 11 else "участках"}')
    if e.bursts >= 3:
        parts.append(f'то усиливается, то спадает: {e.bursts:.0f} {plural(e.bursts, ("всплеск", "всплеска", "всплесков"))}')
    return f'{t:%d.%m %H:%M}{zone} — {place} — ' + ', '.join(parts) + ' — требуется проверка'


def classify_events(events, matrix):
    if events.empty:
        return events
    valid = matrix.where(matrix > INVALID_BELOW)
    step = matrix.index.to_series().diff().median()
    spacing = float(pd.Series(matrix.columns).diff().median())
    shapes = pd.DataFrame([event_shape(valid, e, step, spacing) for e in events.itertuples()], index=events.index,
                          columns=['start_time', 'peak_time', 'peak_place_m', 'change_from_background_c', 'duration_min',
                                   'returned', 'width_m', 'extent_from_m', 'extent_to_m', 'bursts'])
    events = events.join(shapes)
    measured = events.duration_min.notna()
    heating = events.change_from_background_c.fillna(events.peak_change_c).gt(0)
    events['description'] = events.description.where(events.description.eq('краткий выброс'),
                                                      heating.map({True: 'нагрев', False: 'охлаждение'}))
    events['duration_class'] = 'не определена'
    events.loc[measured, 'duration_class'] = events.duration_min.lt(SHORT_MIN).map(
        {True: 'кратковременное', False: 'продолжительное'})
    events.loc[measured & ~events.returned.fillna(True).astype(bool), 'duration_class'] = 'не вернулось к фону до конца данных'
    events['extent_class'] = events.width_m.ge(WIDE_M).map({True: 'обширное', False: 'локальное'}).where(measured, 'не определена')
    low = events.extent_from_m.fillna(events.coordinate_from).to_numpy(dtype=float)
    high = events.extent_to_m.fillna(events.coordinate_to).to_numpy(dtype=float)
    same_place = (low[:, None] <= high[None, :] + SAME_PLACE_M) & (low[None, :] <= high[:, None] + SAME_PLACE_M)
    start = pd.to_datetime(events.start_time.fillna(events.first_alert), format='ISO8601')
    moments = start.map(lambda t: t.value).to_numpy()
    same_time = np.abs(moments[:, None] - moments[None, :]) <= SAME_TIME.value
    place = groups(same_place)
    sign = heating.to_numpy()
    spot = groups(same_place & (sign[:, None] == sign[None, :]))
    repeats, periods = [], []
    for i in range(len(events)):
        here = np.sort(moments[spot == spot[i]])
        distinct = here[np.r_[True, np.diff(here) > SAME_TIME.value]]
        gaps = np.diff(distinct) / pd.Timedelta(minutes=1).value
        repeats.append(len(distinct))
        regular = len(gaps) >= 3 and gaps.std() <= PERIODIC_SPREAD * gaps.mean()
        periods.append(round(gaps.mean()) if regular else np.nan)
    events['place'] = pd.factorize(place)[0] + 1
    events['repeats_at_place'], events['period_min'] = repeats, periods
    events['places_at_same_time'] = [len(set(place[same_time[i]])) for i in range(len(events))]
    parts = [events.duration_class, events.extent_class,
             events.repeats_at_place.map(lambda n: f'повторяется: событий на этом месте {n}' if n > 1 else ''),
             events.places_at_same_time.map(lambda n: f'одновременно на участках: {n}' if n > 1 else ''),
             events.bursts.map(lambda n: f'пульсирующее: всплесков {int(n)}' if n >= 3 else '')]
    periodic = events.period_min.notna()
    parts[2] = parts[2].where(~periodic, 'периодическое: примерно каждые ' + events.period_min.astype('Int64').astype(str) + ' мин')
    events['classes'] = pd.concat(parts, axis=1).apply(lambda row: ', '.join(p for p in row if p), axis=1)
    events['message'] = [operator_message(e) for e in events.itertuples()]
    return events


def run(data_dir, annotations_path, cases_dir, output, seed=42, ends_per_file=110, columns_per_end=24):
    started = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Папка результата не пуста: прошлый эксперимент не перезаписываем')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    matrices, qualities = {}, []
    for path in sorted(data_dir.glob('[1-6].csv')):
        matrix, quality = read_source(path)
        matrices[path.name] = matrix
        qualities.append(quality)
        print(f"прочитан {path.name}: {quality['profiles']} профилей x {quality['coordinates']} точек", flush=True)
    quality = pd.DataFrame(qualities)
    quality.to_csv(output / 'data_quality.csv', index=False, encoding='utf-8-sig')
    device_of = dict(zip(quality.file, quality.dts_id))

    if annotations_path is not None and cases_dir is not None:
        annotations, cards = load_cards(annotations_path, cases_dir)
    else:
        print('Разметка не задана: обучаемся на искусственных событиях, слепой тест на ручных фрагментах пропускаем', flush=True)
        annotations, cards = pd.DataFrame(columns=['source_file', 'coordinate', 'start', 'end', 'label']), {}
    for sample_id, row in annotations.iterrows():
        matrix = matrices[row.source_file]
        column = matrix.columns[matrix.columns.get_indexer([row.coordinate], method='nearest')[0]]
        source = matrix.loc[cards[sample_id].index, column]
        assert (source.round(4) == cards[sample_id].temperature_c.round(4)).all(), sample_id
    zones = exclusion_zones(annotations, matrices)
    for name in matrices:
        matrices[name], shifts = remove_common_mode(matrices[name])
        quality.loc[quality.file.eq(name), 'common_mode_profiles_corrected'] = len(shifts)
    quality.to_csv(output / 'data_quality.csv', index=False, encoding='utf-8-sig')

    examples = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        rng = random.Random(seed + offset)
        ends = list(window_ends(matrix))
        chosen = sorted(rng.sample(ends, min(ends_per_file, len(ends))))
        examples[name] = make_examples(matrix, quality.set_index('file', drop=False).loc[name], chosen, columns_per_end,
                                       rng, zones.get(name, []), 'train', augment_noise=True)
        test_rng = random.Random(seed + 1000 + offset)
        test_ends = sorted(test_rng.sample(ends, min(ends_per_file // 2, len(ends))))
        examples[name + ':test'] = make_examples(matrix, quality.set_index('file', drop=False).loc[name], test_ends,
                                                 columns_per_end, test_rng, zones.get(name, []), 'test')
        print(f"{name}: обучающих окон {len(examples[name][3])}, тестовых {len(examples[name + ':test'][3])}", flush=True)
    pd.concat([examples[k][3] for k in examples]).to_csv(output / 'synthetic_passport.csv', index=False)

    binary = annotations.loc[annotations.label.isin(['событие', 'нет события'])].copy()
    binary['target'] = binary.label.eq('событие').astype(int)
    binary['device'] = binary.source_file.map(device_of)
    if len(binary):
        card_x, card_x2, card_table = card_inputs(cards, binary.index, annotations, matrices)
        card_table = card_table.loc[binary.index].reset_index(drop=True)

    devices = sorted(set(device_of.values()))
    synthetic_rows, card_rows, load_rows, fit_logs, histories, all_events = [], [], [], [], [], []
    for fold, device in enumerate(devices, 1):
        train_files = [f for f in matrices if device_of[f] != device]
        test_files = [f for f in matrices if device_of[f] == device]
        x = torch.cat([examples[f][0] for f in train_files])
        x2 = torch.cat([examples[f][1] for f in train_files])
        table = pd.concat([examples[f][2] for f in train_files], ignore_index=True)
        passport = pd.concat([examples[f][3] for f in train_files], ignore_index=True)
        table['level_c'] = passport.level_c.values
        assert not set(passport.device) & {device}
        print(f'fold {fold}: проверяем {test_files}, обучаем на {train_files} ({len(passport)} окон)', flush=True)
        fitted, log = fit_all(x, x2, table, passport, seed)
        histories += [dict(row, fold=fold) for row in fitted.pop('_cnn_history')]
        fit_logs += [dict(row, fold=fold, held_out=';'.join(test_files)) for row in log]

        tx = torch.cat([examples[f + ':test'][0] for f in test_files])
        tx2 = torch.cat([examples[f + ':test'][1] for f in test_files])
        ttable = pd.concat([examples[f + ':test'][2] for f in test_files], ignore_index=True)
        tpass = pd.concat([examples[f + ':test'][3] for f in test_files], ignore_index=True)
        ttable['level_c'] = tpass.level_c.values
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, tx, ttable, tx2)
            tpass[name + '_score'] = values.values
            tpass[name] = values.ge(threshold).astype(int).values

        mask = binary.device.eq(device).to_list()
        idx = [i for i, m in enumerate(mask) if m]
        for name in MODELS if idx else []:
            model, threshold = fitted[name]
            values = scores(name, model, card_x[idx], card_table.loc[idx].reset_index(drop=True), card_x2[idx])
            rule_ok = measured_c(card_table).ge(notify_limit(card_table))
            for i, sample_id, value in zip(idx, binary.index[idx], values):
                card_rows.append({'sample_id': sample_id, 'model': name, 'fold': fold, 'score': float(value),
                                  'file': binary.loc[sample_id, 'source_file'],
                                  'threshold': threshold, 'prediction': int(value >= threshold),
                                  'rule_ok': bool(rule_ok.iloc[i]),
                                  'target': int(binary.loc[sample_id, 'target'])})

        for name_file in test_files:
            grid, stride = scan(matrices[name_file], fitted)
            columns = list(grid.coordinate.drop_duplicates().sort_values())
            matrix = matrices[name_file]
            cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * COMMISSIONING_SHARE
            calibrate(grid, fitted)
            later = grid.loc[grid.end > cutoff].copy()
            later_hours = (matrix.index[-1] - cutoff).total_seconds() / 3600
            for name in MODELS:
                events = merge_events(later, name + '_notify', columns)
                load_rows.append({'file': name_file, 'model': name, 'windows': len(later), 'hours': later_hours,
                                  'flagged_share': float(later[name + '_notify'].mean()), 'events': len(events),
                                  'events_per_hour': len(events) / later_hours, 'coordinate_stride': stride})
                if len(events):
                    all_events.append(events.assign(file=name_file, model=name, fold=fold))
                in_file = tpass.file.eq(name_file)
                tpass.loc[in_file, name + '_notify'] = (tpass[name + '_score'].ge(fitted[name][1])
                                                        & measured_c(tpass).ge(notify_limit(ttable).values)).astype(int)
                tpass.loc[in_file, 'after_commissioning'] = tpass.end.gt(cutoff)
                for row in card_rows:
                    if row['file'] == name_file and row['model'] == name and row['fold'] == fold:
                        row['prediction_notify'] = int(row['prediction'] and row['rule_ok'])
                        row['card_in_commissioning'] = bool(cards[row['sample_id']].index[-1] <= cutoff)
            grid.to_csv(output / f'scan_{name_file.replace(".csv", "")}.csv.gz', index=False)
        synthetic_rows.append(tpass.assign(fold=fold))

    synthetic = pd.concat(synthetic_rows, ignore_index=True)
    synthetic.to_csv(output / 'synthetic_test_predictions.csv', index=False)
    card_predictions = pd.DataFrame(card_rows)
    card_predictions.to_csv(output / 'card_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(load_rows).to_csv(output / 'operator_load.csv', index=False)
    pd.DataFrame(fit_logs).to_csv(output / 'fold_settings.csv', index=False)
    pd.DataFrame(histories).to_csv(output / 'cnn_training_history.csv', index=False)
    if all_events:
        pd.concat(all_events).to_csv(output / 'events.csv', index=False, encoding='utf-8-sig')

    summary = []
    for name in MODELS:
        row = {'model': NAMES[name]}
        cards_part = card_predictions.loc[card_predictions.model.eq(name)] if len(card_predictions) else pd.DataFrame()
        if len(cards_part):
            row.update({f'cards_{k}': v for k, v in evaluate(cards_part.target, cards_part.prediction).items()})
            row['cards_auc'] = roc_auc_score(cards_part.target, cards_part.score)
            row['cards_notify_TP'] = int(cards_part.loc[cards_part.target.eq(1), 'prediction_notify'].sum())
            row['cards_notify_FP'] = int(cards_part.loc[cards_part.target.eq(0), 'prediction_notify'].sum())
        row.update({f'synthetic_{k}': v for k, v in evaluate(synthetic.label, synthetic[name]).items()})
        for kind in ['step', 'ramp', 'pulse', 'spike']:
            part = synthetic.loc[synthetic.kind.eq(kind) & synthetic.label.eq(1)]
            row[f'recall_{kind}'] = part[name].mean()
        late = synthetic.loc[synthetic.after_commissioning.eq(True)]
        row['notify_recall_events'] = late.loc[late.label.eq(1), name + '_notify'].mean()
        row['notify_false_alarm_quiet'] = late.loc[late.kind.eq('none'), name + '_notify'].mean()
        row['false_alarm_glitch'] = synthetic.loc[synthetic.kind.eq('glitch'), name].mean()
        row['false_alarm_quiet'] = synthetic.loc[synthetic.kind.eq('none'), name].mean()
        row['false_alarm_small'] = synthetic.loc[synthetic.kind.eq('small'), name].mean()
        load = pd.DataFrame(load_rows)
        part = load.loc[load.model.eq(name)]
        row['events_per_hour'] = part.events.sum() / part.hours.sum()
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')

    x = torch.cat([examples[f][0] for f in matrices])
    x2 = torch.cat([examples[f][1] for f in matrices])
    table = pd.concat([examples[f][2] for f in matrices], ignore_index=True)
    passport = pd.concat([examples[f][3] for f in matrices], ignore_index=True)
    table['level_c'] = passport.level_c.values
    fitted, log = fit_all(x, x2, table, passport, seed)
    fitted.pop('_cnn_history')
    (output / 'models').mkdir()
    for name in ('cnn2d',):
        network, network_threshold = fitted[name]
        torch.save({'state_dict': network.state_dict(), 'width': network.filters[0].out_channels,
                    'threshold': network_threshold, 'event_c': EVENT_C}, output / 'models' / f'final_{name}.pt')
    joblib.dump({name: fitted[name] for name in MODELS if name != 'cnn2d'}, output / 'models' / 'final_other.joblib')
    pd.DataFrame(log).to_csv(output / 'final_settings.csv', index=False)
    protocol = {'background_minutes': 80, 'grey_max_c': GREY_MAX_C, 'common_mode_min_c': COMMON_MODE_MIN_C, 'radius': RADIUS, 'cnn_widths': list(CNN_WIDTHS), 'max_epochs': MAX_EPOCHS, 'seed': seed, 'window_minutes': 40, 'bin_minutes': 2, 'event_c': EVENT_C,
                'evaluation': 'leave-one-device-out; human cards never used for training or thresholds',
                'pandas': pd.__version__, 'torch': torch.__version__,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'elapsed_seconds': time.perf_counter() - started}
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.round(3).to_string(index=False))


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_STYLE = ({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def injection_example(data_dir, out):
    matrix, _ = read_source(data_dir / '4.csv')
    ends = list(window_ends(matrix))
    end = ends[200]
    rows = matrix.loc[(matrix.index > end - WINDOW) & (matrix.index <= end)]
    column = rows.columns[400]
    minutes = ((rows.index - (end - WINDOW)) / pd.Timedelta(minutes=1)).tolist()
    rng = random.Random(3)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    for ax, (kind, size, title) in zip(axes, [('step', 5., 'ступень (экспонента)'), ('ramp', -4., 'плавное охлаждение'),
                                              ('pulse', 6., 'нагрев и остывание'), ('spike', -5., 'краткий выброс')]):
        delta = injected_shape(kind, minutes, 18., size, rng)
        ax.plot(minutes, rows[column].values, color='#8a8f98', lw=1.2, label='реальный фон')
        ax.plot(minutes, rows[column].values + pd.Series(delta).values, color='#c0392b', lw=1.5, label='фон + событие')
        ax.set_title(title)
        ax.set_xlabel('минуты окна')
    axes[0].set_ylabel('°C')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Обучающие примеры: к настоящим измерениям 4.csv добавлены изменения с известной амплитудой', y=1.03)
    fig.tight_layout()
    fig.savefig(out / 'fig2_injection_examples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def heatmap(run, data_dir, out, name='4.csv', model='cnn2d'):
    matrix, _ = read_source(data_dir / name)
    events = pd.read_csv(run / 'events.csv')
    events = events.loc[events.file.eq(name) & events.model.eq(model)].copy()
    for column in ['first_alert', 'last_alert']:
        events[column] = pd.to_datetime(events[column], format='ISO8601')
    shown = matrix.loc[:, matrix.lt(INVALID_BELOW).sum().eq(0)]
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = (shown.index - shown.index[0]) / pd.Timedelta(hours=1)
    image = ax.pcolormesh(shown.columns, hours, shown.values, cmap='inferno', shading='auto')
    fig.colorbar(image, ax=ax, label='°C')
    start = shown.index[0]
    for _, e in events.iterrows():
        y0 = (e.first_alert - start) / pd.Timedelta(hours=1) - 40 / 60
        y1 = (e.last_alert - start) / pd.Timedelta(hours=1)
        ax.add_patch(plt.Rectangle((e.coordinate_from - 1, y0), e.coordinate_to - e.coordinate_from + 2, y1 - y0,
                                   fill=False, edgecolor='#4fc3f7', lw=1.4))
    ax.axhline((shown.index[-1] - start) / pd.Timedelta(hours=1) * COMMISSIONING_SHARE, color='white', ls='--', lw=1)
    ax.set_xlabel('расстояние вдоль волокна, м')
    ax.set_ylabel(f'часы от начала записи ({shown.index[0]})')
    ax.set_title(f'{name}: голубые рамки — события {NAMES[model]} после пусконаладки (пунктир), {len(events)} шт.')
    fig.tight_layout()
    fig.savefig(out / f'fig7_heatmap_{name.replace(".csv", "")}.png', dpi=150)
    plt.close(fig)


def render_figures(run, data_dir, annotations_path, cases_dir):
    plt.rcParams.update(PLOT_STYLE)
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    injection_example(data_dir, out)
    heatmap(run, data_dir, out, '4.csv')
    tracks_figure(run / 'tracks', out)
    print('графики сохранены в', out)


def hot_points(matrix, rise=4.0, history='6min'):
    valid = matrix.loc[:, matrix.lt(INVALID_BELOW).sum().eq(0)]
    baseline = valid.rolling(history, min_periods=3, closed='left').median()
    excess = (valid - baseline).where((valid - baseline).ge(rise)).stack().rename('excess_c').reset_index()
    excess.columns = ['time', 'coordinate', 'excess_c']
    excess['minutes'] = (excess.time - matrix.index[0]).dt.total_seconds() / 60
    return excess


def heat_tracks(points, seconds_scale=10., eps=12., min_points=4, link_minutes=3., link_coordinates=60.):
    from sklearn.cluster import DBSCAN
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()
    scaled = pd.DataFrame({'t': points.minutes * seconds_scale, 'c': points.coordinate})
    points = points.assign(spot=DBSCAN(eps=eps, min_samples=min_points).fit_predict(scaled))
    spots = (points.loc[points.spot.ge(0)].groupby('spot')
             .agg(start_min=('minutes', 'min'), end_min=('minutes', 'max'), coordinate=('coordinate', 'median'),
                  coordinate_from=('coordinate', 'min'), coordinate_to=('coordinate', 'max'),
                  peak_excess_c=('excess_c', 'max'), points=('excess_c', 'size'), first_time=('time', 'min'))
             .sort_values('start_min').reset_index())
    tracks, track_of = [], []
    for spot in spots.itertuples():
        candidates = [i for i, track in enumerate(tracks)
                      if -2. <= spot.start_min - track[-1].end_min <= link_minutes
                      and abs(spot.coordinate - track[-1].coordinate) <= link_coordinates]
        if candidates:
            best = min(candidates, key=lambda i: abs(spot.coordinate - tracks[i][-1].coordinate))
            tracks[best].append(spot)
            track_of.append(best)
        else:
            tracks.append([spot])
            track_of.append(len(tracks) - 1)
    spots['track'] = track_of
    rows = []
    for number, track in enumerate(tracks):
        frame = pd.DataFrame(track)
        if len(frame) < 3:
            continue
        x, y = (frame.start_min + frame.end_min) / 2, frame.coordinate
        spread = ((x - x.mean()) ** 2).sum()
        speed = ((x - x.mean()) * (y - y.mean())).sum() / spread if spread > 0 else 0.
        r = ((x - x.mean()) * (y - y.mean())).sum() / (spread * ((y - y.mean()) ** 2).sum()) ** .5
        rows.append({'track': number, 'first_time': frame.first_time.min(), 'start_min': frame.start_min.min(),
                     'end_min': frame.end_min.max(), 'spots': len(frame),
                     'coordinate_start': frame.coordinate.iloc[0], 'coordinate_end': frame.coordinate.iloc[-1],
                     'speed_m_per_min': speed, 'linearity_r': r, 'peak_excess_c': frame.peak_excess_c.max(),
                     'direction': 'от начала волокна' if speed > 0 else 'к началу волокна',
                     'description': 'движущийся локальный нагрев' if abs(r) >= .8 else 'серия локальных нагревов',
                     'status': 'требуется проверка'})
    return spots, pd.DataFrame(rows)


def find_tracks(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(data_dir.glob('[1-6].csv')):
        matrix, quality = read_source(path)
        points = hot_points(matrix)
        spots, tracks = heat_tracks(points)
        points.to_csv(output / f'hot_points_{path.stem}.csv', index=False)
        if len(spots):
            spots.to_csv(output / f'hot_spots_{path.stem}.csv', index=False)
        if len(tracks):
            tracks.to_csv(output / f'tracks_{path.stem}.csv', index=False, encoding='utf-8-sig')
        summary.append({'file': path.name, 'median_step_s': quality['median_step_s'], 'hot_points': len(points),
                        'spots': len(spots), 'tracks': len(tracks),
                        'moving_tracks': int(tracks.description.eq('движущийся локальный нагрев').sum()) if len(tracks) else 0})
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(output / 'tracks_summary.csv', index=False)


def tracks_figure(output, out, name='5'):
    points = output / f'hot_points_{name}.csv'
    if not points.exists():
        return
    points = pd.read_csv(points)
    spots = pd.read_csv(output / f'hot_spots_{name}.csv')
    tracks = pd.read_csv(output / f'tracks_{name}.csv') if (output / f'tracks_{name}.csv').exists() else pd.DataFrame()
    episodes = [(points.minutes.min(), 80.), (285., 352.)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, (low, high) in zip(axes, episodes):
        part = points.loc[points.minutes.between(low, high)]
        image = ax.scatter(part.minutes, part.coordinate, c=part.excess_c, s=5, cmap='inferno_r', vmin=4, vmax=12)
        for track in tracks.itertuples():
            if track.description == 'движущийся локальный нагрев' and low <= track.start_min <= high:
                members = spots.loc[spots.track.eq(track.track)]
                ax.plot((members.start_min + members.end_min) / 2, members.coordinate, color='#2462a8', lw=1.6)
                ax.annotate(f'{track.speed_m_per_min:+.0f} м/мин', ((track.start_min + track.end_min) / 2,
                            (track.coordinate_start + track.coordinate_end) / 2), color='#2462a8', fontsize=9,
                            xytext=(6, 0), textcoords='offset points')
        ax.set_xlabel('минуты от начала записи 5.csv')
        ax.set_ylabel('расстояние вдоль волокна, м')
    fig.colorbar(image, ax=axes, label='на сколько °C выше своего прошлого (6 мин)')
    fig.suptitle('5.csv: краткие локальные нагревы +4…+14 °C и их перемещение вдоль волокна (синие линии - треки)')
    fig.savefig(out / 'fig9_moving_heat_5.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def slow_changes(matrix, hours=2, every='10min', max_columns=1500):
    usable = matrix.columns[matrix.notna().all() & matrix.ge(INVALID_BELOW).all()]
    stride = max(1, math.ceil(len(usable) / max_columns))
    smooth = matrix[usable[::stride]].rolling('20min', min_periods=3).median()
    grid = smooth.resample(every).last().dropna(how='all')
    delta = grid - grid.shift(freq=pd.Timedelta(hours=hours)).reindex(grid.index)
    table = delta.stack().rename('signed_level_c').reset_index()
    table.columns = ['end', 'coordinate', 'signed_level_c']
    table['level_c'] = table.signed_level_c.abs()
    table['spike_c'] = table.level_c
    table['slow'] = table.level_c.ge(EVENT_C)
    return table


def forecast_changes(matrix, every='10min', max_columns=1500, start_c=1.0, look_minutes=30, horizon_minutes=30):
    usable = matrix.columns[matrix.notna().all() & matrix.ge(INVALID_BELOW).all()]
    stride = max(1, math.ceil(len(usable) / max_columns))
    columns = usable[::stride]
    smooth = matrix[columns].rolling('20min', min_periods=3).median()
    grid = smooth.resample(every).last()
    steps = int(pd.Timedelta(look_minutes, 'min') / pd.Timedelta(every))
    background = grid.rolling(5, min_periods=3).median().shift(8)
    change = grid - background
    speed = (grid - grid.shift(steps)) / look_minutes
    same_way = (change * speed).gt(0)
    eta = ((EVENT_C - change.abs()) / speed.abs()).where(same_way & speed.ne(0))
    flag = change.abs().ge(start_c) & change.abs().lt(EVENT_C) & eta.le(horizon_minutes)
    table = pd.DataFrame({'signed_level_c': change.stack(), 'eta_min': eta.stack(), 'forecast': flag.stack()}).reset_index()
    table.columns = ['end', 'coordinate', 'signed_level_c', 'eta_min', 'forecast']
    table = table.dropna(subset=['signed_level_c'])
    table['level_c'] = table.signed_level_c.abs()
    table['spike_c'] = table.level_c
    table['position'] = matrix.columns.get_indexer(table.coordinate)
    return table


def glitch_points(matrix, jump=4.0, calm=1.5, history='6min'):
    valid = matrix.loc[:, matrix.notna().all() & matrix.lt(INVALID_BELOW).sum().eq(0)]
    deviation = valid - valid.rolling(history, min_periods=3, closed='left').median()
    isolated = (deviation.abs().ge(jump) & deviation.shift(1, axis=1).abs().lt(calm)
                & deviation.shift(-1, axis=1).abs().lt(calm) & deviation.shift(-1).abs().lt(calm))
    found = deviation.where(isolated).stack().rename('deviation_c').reset_index()
    found.columns = ['time', 'coordinate', 'deviation_c']
    return found


def load_final(model_dir):
    fitted = joblib.load(model_dir / 'final_other.joblib')
    for name, factory in (('cnn2d', SpatialCNN),):
        saved = torch.load(model_dir / f'final_{name}.pt', map_location='cpu', weights_only=True)
        set_event_c(saved.get('event_c', 3.0))
        network = factory(saved['width'])
        network.load_state_dict(saved['state_dict'])
        network.eval()
        fitted[name] = (network, float(saved['threshold']))
    return fitted


def detect(path, model_dir, output, model='cnn2d', event_c=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Папка результата не пуста')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fitted = load_final(model_dir)
    trained_c = EVENT_C
    if event_c is not None and event_c != trained_c:
        print(f'Порог уведомления {event_c} °C вместо {trained_c} °C, на который учили модели. '
              + ('Для понижения порога модели нужно переобучить (--event-c при обучении).' if event_c < trained_c else ''))
        set_event_c(event_c)
    matrix, quality = read_source(path)
    quality['event_c'] = EVENT_C
    matrix, shifts = remove_common_mode(matrix)
    grid, stride = scan(matrix, fitted, max_columns=None)
    if grid.empty:
        message = {'type': 'мало данных', 'description': 'нет ни одного окна 40 минут без разрывов больше 10 минут; '
                                                          'анализ событий не выполнялся', 'status': 'нужна более длинная запись'}
        pd.DataFrame([message]).to_csv(output / 'data_quality_messages.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame().to_csv(output / 'events.csv', index=False, encoding='utf-8-sig')
        (output / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{path.name}: {message['description']}")
        return
    cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * COMMISSIONING_SHARE
    calibrate(grid, fitted)
    columns = list(grid.coordinate.drop_duplicates().sort_values())
    early = merge_events(grid.loc[grid.end <= cutoff], model + '_notify', columns)
    later = merge_events(grid.loc[grid.end > cutoff], model + '_notify', columns)
    events = pd.concat([early.assign(period='пусконаладка'),
                        later.assign(period='работа')], ignore_index=True)
    events = classify_events(events, matrix)
    invalid = matrix.columns[matrix.lt(INVALID_BELOW).any()]
    messages = []
    if len(invalid):
        messages.append({'type': 'недостоверные данные', 'coordinate_from': float(invalid.min()),
                         'coordinate_to': float(invalid.max()), 'coordinates': len(invalid),
                         'description': f'значения ниже {INVALID_BELOW} °C физически невозможны', 'status': 'требуется проверка'})
    if quality['duplicate_profiles_dropped']:
        messages.append({'type': 'повтор профиля', 'description': f"{quality['duplicate_profiles_dropped']} профилей с уже "
                         'встречавшимся временем; оставлен последний', 'status': 'требуется проверка данных'})
    if len(shifts):
        messages.append({'type': 'общий сдвиг всего волокна', 'time': str(shifts.index.min()), 'until': str(shifts.index.max()),
                         'description': f'{len(shifts)} профилей сдвинуты целиком на {shifts.abs().min():.1f}-{shifts.abs().max():.1f} °C; '
                                        'вероятный эффект прибора, вычтен до анализа', 'status': 'требуется проверка прибора'})
    gaps = matrix.index.to_series().diff()
    for stamp, gap in gaps.loc[gaps.gt(MAX_GAP)].items():
        messages.append({'type': 'разрыв данных', 'time': str(stamp - gap), 'until': str(stamp),
                         'description': f'нет профилей {gap}', 'status': 'анализ не выполнялся'})
    glitches = glitch_points(matrix)
    if len(glitches):
        messages.append({'type': 'вероятный сбой измерения', 'coordinates': glitches.coordinate.nunique(),
                         'description': f'{len(glitches)} одноточечных однозамерных выбросов >= 4 °C; '
                                        'соседи и следующий замер в норме', 'status': 'требуется проверка данных'})
        glitches.to_csv(output / 'probable_glitches.csv', index=False, encoding='utf-8-sig')
    border = merge_events(grid.loc[grid.end > cutoff], model + '_candidate', columns)
    if len(border):
        border = border.assign(status='кандидат: изменение от фона меньше порога, без уведомления')
    border.to_csv(output / 'candidates.csv', index=False, encoding='utf-8-sig')
    slow = slow_changes(matrix)
    slow_events = merge_events(slow, 'slow', list(slow.coordinate.drop_duplicates().sort_values()))
    if len(slow_events):
        kind = slow_events.peak_change_c.gt(0).map({True: 'медленный нагрев', False: 'медленное охлаждение'})
        slow_events = slow_events.assign(description=kind, status='журнал: медленное изменение, без уведомления')
    slow_events.to_csv(output / 'slow_changes.csv', index=False, encoding='utf-8-sig')
    forecast = forecast_changes(matrix)
    forecasts = merge_events(forecast, 'forecast', list(forecast.coordinate.drop_duplicates().sort_values()))
    if len(forecasts):
        kind = forecasts.peak_change_c.gt(0).map({True: 'рост', False: 'снижение'})
        forecasts = forecasts.assign(description=kind + ' температуры: при той же скорости превысит порог',
                                     status='прогноз, без уведомления')
    forecasts.to_csv(output / 'forecast.csv', index=False, encoding='utf-8-sig')
    points = hot_points(matrix)
    spots, tracks = heat_tracks(points)
    events.to_csv(output / 'events.csv', index=False, encoding='utf-8-sig')
    tracks.to_csv(output / 'moving_heat.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(messages).to_csv(output / 'data_quality_messages.csv', index=False, encoding='utf-8-sig')
    (output / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'file': path.name, 'model': model, 'events': len(events),
                      'events_after_commissioning': len(later), 'moving_heat_tracks': len(tracks),
                      'slow_changes_journal': len(slow_events), 'forecasts': len(forecasts), 'candidates': len(border),
                      'data_quality_messages': len(messages), 'output': str(output)}, ensure_ascii=False, indent=2))
    if len(events):
        first = events.sort_values('change_from_background_c', key=abs, ascending=False).iloc[0]
        print(f'Пример: {first.message}')


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description='FiberWatch: обнаружение температурных событий по данным DTS')
    parser.add_argument('--data-dir', type=Path, help='Папка с 1.csv ... 6.csv')
    parser.add_argument('--figures-only', action='store_true', help='Только построить графики по готовой папке --output')
    parser.add_argument('--tracks', action='store_true', help='Найти краткие и движущиеся нагревы (папка --output/tracks)')
    parser.add_argument('--detect', type=Path, help='Применить итоговую модель к CSV прибора; нужна --model-dir')
    parser.add_argument('--model-dir', type=Path, help='Папка models/ обученного прогона')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--cases-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ends-per-file', type=int, default=110, help='Сколько моментов времени брать из каждого файла')
    parser.add_argument('--model', choices=MODELS, default='cnn2d', help='Модель для --detect')
    parser.add_argument('--event-c', type=float, help='Порог уведомления, °C (по умолчанию 3). При обучении меняет и метки; '
                        'при --detect по умолчанию берётся из моделей')
    parser.add_argument('--quick', action='store_true', help='Быстрая проверка запуска (~3 мин): мало окон, короткое обучение')
    args = parser.parse_args()
    if args.quick:
        global CNN_WIDTHS, MAX_EPOCHS
        CNN_WIDTHS, MAX_EPOCHS, args.ends_per_file = (16,), 15, 12
    if args.event_c is not None and not args.detect:
        set_event_c(args.event_c)
    if args.detect:
        if args.model_dir is None:
            parser.error('--detect требует --model-dir')
        detect(args.detect, args.model_dir, args.output, args.model, args.event_c)
        return
    if args.data_dir is None:
        parser.error('для обучения нужна --data-dir (папка с 1.csv ... 6.csv); разметка - по желанию')
    if args.tracks:
        find_tracks(args.data_dir, args.output / 'tracks')
    elif not args.figures_only:
        run(args.data_dir, args.annotations, args.cases_dir, args.output, args.seed, args.ends_per_file)
        find_tracks(args.data_dir, args.output / 'tracks')
    render_figures(args.output, args.data_dir, args.annotations, args.cases_dir)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        sys.exit(2)
