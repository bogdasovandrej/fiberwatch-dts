"""FiberWatch v4: обнаружение температурных событий в данных DTS МГТС.

Один файл: чтение -> проверка качества -> окна -> признаки -> обучение -> оценка -> события.
Таблицы читает и преобразует Pandas. PyTorch нужен для CNN, scikit-learn для леса и метрик.

Главная идея v4. Ручной разметки мало (28 фрагментов), и она почти разделяется одним
порогом. Поэтому модели учатся на тысячах окон НАСТОЯЩЕГО фона из исходных CSV, в которые
мы добавляем синтетические нагревы/охлаждения/выбросы с известной амплитудой. Ручные
карточки в обучение не попадают вообще и служат слепой проверкой. Оценка идёт
leave-one-device-out: прибор, на котором проверяем, не участвует в обучении и подборе порога.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
WINDOW = pd.Timedelta(minutes=40)       # сколько истории видит модель
BIN = pd.Timedelta(minutes=2)           # шаг агрегирования: делает файлы с разной частотой сопоставимыми
N_BINS = 20
MAX_GAP = pd.Timedelta(minutes=10)      # больший разрыв = окно не строим (не склеиваем ненаблюдаемое)
INVALID_BELOW = -273.15                 # ниже абсолютного нуля температура физически невозможна
CHANNELS = ['median_relative_c', 'min_relative_c', 'max_relative_c', 'observed']
EVENT_C = 3.0      # политика команды: изменение >= 3 °C -> «требуется проверка»
SMALL_C = 1.5      # изменение <= 1.5 °C -> обычное поведение; между 1.5 и 3 -> пример не используем
ZONE_COORDINATES = 200   # зона = 200 соседних координат: порог настраивается под её обычное поведение
COMMISSIONING_SHARE = .3 # первые 30% записи прибора = «пусконаладка»: только наблюдаем, порог по зонам
MODELS = ['amplitude_rule', 'range_rule', 'random_forest', 'cnn']
NAMES = {'amplitude_rule': 'Правило: отклонение медианы', 'range_rule': 'Правило: размах (v3)',
         'random_forest': 'Случайный лес', 'cnn': 'CNN'}


# ---------------------------------------------------------------- 1. Чтение и качество

def read_source(path):
    """CSV МГТС -> таблица «время x координата». Одна строка CSV = один профиль вдоль волокна."""
    source = pd.read_csv(path)
    if len(source[['dts_id', 'channel_id']].drop_duplicates()) != 1:
        raise ValueError(f'{path.name}: в файле смешаны приборы или каналы')
    times = pd.DatetimeIndex(pd.to_datetime(source['timestamp'], format='ISO8601'))
    profiles = [pd.DataFrame(json.loads(text), columns=['coordinate', 'temperature_c'])
                .set_index('coordinate')['temperature_c'] for text in source['points']]
    matrix = pd.concat(profiles, axis=1, keys=range(len(profiles))).T
    matrix.index = times
    matrix = matrix.sort_index()
    if not matrix.index.is_unique:
        raise ValueError(f'{path.name}: повторяющееся время профиля')
    matrix = matrix.sort_index(axis=1)
    step = matrix.index.to_series().diff().dt.total_seconds()
    invalid = matrix.lt(INVALID_BELOW)
    quality = {
        'file': path.name, 'dts_id': str(source.dts_id.iloc[0]), 'channel_id': int(source.channel_id.iloc[0]),
        'timezone': str(matrix.index.tz) if matrix.index.tz is not None else 'не указан',
        'profiles': len(matrix), 'coordinates': matrix.shape[1],
        'start': str(matrix.index[0]), 'end': str(matrix.index[-1]),
        'median_step_s': float(step.median()), 'max_gap_min': float(step.max() / 60),
        'gaps_over_10min': int(step.gt(MAX_GAP.total_seconds()).sum()),
        'short_profiles': int(matrix.isna().any(axis=1).sum()),
        'invalid_values': int(invalid.sum().sum()),
        'always_invalid_coordinates': int(invalid.all(axis=0).sum()),
        'min_valid_c': float(matrix.where(~invalid).min().min()),
        'max_c': float(matrix.max().max()),
    }
    return matrix, quality


def window_ends(matrix):
    """Моменты, для которых есть полные 40 минут истории без разрывов больше 10 минут."""
    times = matrix.index.to_series()
    block = times.diff().gt(MAX_GAP).cumsum()
    block_start = times.groupby(block).transform('min')
    return times.loc[times - block_start >= WINDOW - pd.Timedelta(minutes=2)].index


# ---------------------------------------------------------------- 2. Представление окна

def represent(rows, end):
    """Окно (время x ряды) -> 20 интервалов по 2 минуты относительно опорного уровня.

    Совпадает с represent_card из temperature_model.py (v3), но считает сразу много рядов.
    median - уровень, min/max - короткие выбросы, observed - был ли в интервале замер.
    Опорный уровень = медиана первых трёх замеров окна (не «норма», а точка отсчёта).
    """
    start = end - WINDOW
    rows = rows.loc[(rows.index > start) & (rows.index <= end)]
    if len(rows) < 3 or rows.index.to_series().diff().gt(MAX_GAP).any():
        return None
    position = (rows.index - start) / BIN
    bin_number = [int(-((-value) // 1)) for value in position]   # округление вверх: интервал (k-1, k]
    reference = rows.iloc[:3].median()
    relative = rows - reference
    grouped = relative.groupby(bin_number)
    bins = list(range(1, N_BINS + 1))
    observed = grouped.count().T.reindex(columns=bins).fillna(0).gt(0)
    if observed.iloc[0].sum() < 3:
        return None
    parts = {}
    for name, frame in [('median', grouped.median()), ('min', grouped.min()), ('max', grouped.max())]:
        parts[name] = frame.T.reindex(columns=bins).ffill(axis=1).fillna(0.)
    parts['observed'] = observed.astype(float)
    parts['reference'] = reference
    return parts


def to_tensor(parts):
    """[число окон, 4 канала, 20 интервалов]; температуры в единицах 10 °C, как в v3."""
    stacked = [torch.tensor(parts[name].to_numpy(dtype='float32')) / (10. if name != 'observed' else 1.)
               for name in ['median', 'min', 'max', 'observed']]
    return torch.stack(stacked, dim=1)


def features(parts):
    """Признаки для леса: те же 10 чисел, что extract_features в v3, но сразу для многих окон."""
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
    })
    return table.loc[parts['median'].index].fillna(0.)


def amplitude(parts):
    """Наибольшее отклонение медианы от опорного уровня и наибольший выброс min/max, °C."""
    level = parts['median'].abs().max(axis=1)
    spike = pd.concat([parts['max'].max(axis=1), -parts['min'].min(axis=1)], axis=1).max(axis=1)
    return level, spike


# ---------------------------------------------------------------- 3. Синтетические события на реальном фоне

def injected_shape(kind, minutes, onset, size, rng):
    """Форма добавляемого изменения температуры (°C) в моменты замеров.

    Тепло передаётся к волокну с задержкой, поэтому ступень - экспонента
    1 - exp(-t/tau), а не мгновенный скачок. Параметры - инженерные диапазоны для обучения,
    а не характеристики конкретного объекта МГТС.
    """
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
    else:
        raise ValueError(kind)
    return (size * shape * started).tolist()


def make_examples(matrix, quality, ends, columns_per_end, rng, excluded, purpose):
    """Окна реального фона + синтетические события. Возвращает тензор, признаки и паспорт окон.

    Метка задаётся физически измеримой реализованной амплитудой добавки:
    >= 3 °C -> 1 («требуется проверка»), <= 1.5 °C -> 0, промежуточные отбрасываем.
    Фон берём только «спокойный», чтобы случайное настоящее событие не стало ответом 0.
    """
    tensors, feature_tables, passports = [], [], []
    for end in ends:
        start = end - WINDOW
        rows = matrix.loc[(matrix.index > start) & (matrix.index <= end)]
        usable = rows.columns[rows.notna().all() & rows.ge(INVALID_BELOW).all()]
        blocked = set()
        for zone in excluded:
            if end > zone['start'] and start < zone['end']:
                blocked.update(zone['columns'])
        usable = [c for c in usable if c not in blocked]
        if len(usable) < columns_per_end:
            continue
        chosen = rng.sample(usable, columns_per_end)
        natural = represent(rows[chosen], end)
        if natural is None:
            continue
        level, spike = amplitude(natural)
        natural_level = level
        quiet = [c for c in chosen if level[c] < 1.5 and spike[c] < 4.]
        minutes = ((rows.index - start) / pd.Timedelta(minutes=1)).tolist()
        delta = pd.DataFrame(0., index=rows.index, columns=quiet)
        meta = []
        for column in quiet:
            kind = rng.choices(['none', 'small', 'step', 'ramp', 'pulse', 'spike'], [35, 15, 15, 12, 12, 11])[0]
            realized = 0.
            if kind != 'none':
                low, high = (.3, SMALL_C) if kind == 'small' else (EVENT_C, 12.)
                size = math.exp(rng.uniform(math.log(low), math.log(high))) * rng.choice([-1, 1])
                onset = rng.uniform(8., 39.5)
                values = pd.Series(injected_shape(kind, minutes, onset, size, rng), index=rows.index)
                realized = float((values - values.iloc[:3].median()).abs().max())
                delta[column] = values
            label = 1 if realized >= EVENT_C else (0 if realized <= SMALL_C else None)
            meta.append({'column': column, 'position': matrix.columns.get_loc(column), 'kind': kind, 'realized_c': realized, 'label': label})
        meta = pd.DataFrame(meta)
        meta = meta.loc[meta.label.notna()]
        if meta.empty:
            continue
        kept = meta.column.tolist()
        parts = represent(rows[kept] + delta[kept], end)
        tensors.append(to_tensor(parts))
        feature_tables.append(features(parts).reset_index(drop=True))
        level, spike = amplitude(parts)
        meta = meta.assign(file=quality['file'], device=quality['dts_id'], end=end, purpose=purpose,
                           level_c=level.loc[kept].to_list(), spike_c=spike.loc[kept].to_list(),
                           natural_level_c=natural_level.loc[kept].to_list())
        meta['label'] = meta.label.astype(int)
        passports.append(meta)
    return torch.cat(tensors), pd.concat(feature_tables, ignore_index=True), pd.concat(passports, ignore_index=True)


# ---------------------------------------------------------------- 4. Модели

class TemperatureCNN(nn.Module):
    """Та же архитектура, что в v3: две свёртки по времени, затем среднее и максимум по окну."""
    def __init__(self, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv1d(4, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(width, width, 3, padding=1), nn.ReLU())
        self.output = nn.Linear(width * 2, 1)

    def forward(self, inputs):
        patterns = self.filters(inputs)
        return self.output(torch.cat([patterns.mean(dim=2), patterns.amax(dim=2)], dim=1)).squeeze(1)


def train_cnn(x_train, y_train, x_val, y_val, width, seed, max_epochs=80, patience=8):
    """Мини-пакеты по 256 окон; лучшая эпоха выбирается по ошибке на проверочной части обучения."""
    torch.manual_seed(seed)
    model = TemperatureCNN(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
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
        history.append({'epoch': epoch, 'width': width, 'training_loss': train_loss, 'validation_loss': val_loss})
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


def scores(name, model, x, table):
    if name == 'amplitude_rule':
        return pd.Series(table['level_c'].values)
    if name == 'range_rule':
        return pd.Series(table['range_c'].values)
    if name == 'random_forest':
        return pd.Series(model.predict_proba(table[FEATURES])[:, 1])
    with torch.no_grad():
        return pd.Series(torch.sigmoid(model(x)).tolist())


FEATURES = ['range_c', 'median_range_c', 'std_c', 'net_change_c', 'absolute_net_change_c',
            'max_absolute_rate_c_per_min', 'mean_absolute_rate_c_per_min', 'mean_bin_spread_c',
            'max_bin_spread_c', 'observed_fraction']


def choose_threshold(labels, values):
    """Порог с максимальным F2 (полнота важнее точности - решение команды «меньше пропусков»)."""
    candidates = sorted(set(values.quantile([i / 200 for i in range(1, 200)]).round(6)))
    best = max(candidates, key=lambda t: (fbeta_score(labels, values.ge(t), beta=2, zero_division=0), t))
    return float(best)


def fit_all(x, table, passport, seed):
    """Обучение всех моделей на окнах обучающих приборов; 20% окон (по времени) - для порога и эпох."""
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
        elif name == 'cnn':
            # Ширину выбираем по ошибке на проверочной части, а не по тестовому прибору.
            trials = [train_cnn(x[train_idx], y[train_idx], x[val_idx], y[val_idx], width, seed)
                      for width in (8, 16, 32)]
            model, best_loss, history = min(trials, key=lambda trial: trial[1])
            extra = {'width': model.output.in_features // 2, 'validation_loss': best_loss,
                     'epochs_run': len(history), 'parameters': sum(p.numel() for p in model.parameters()),
                     'history': [row for trial in trials for row in trial[2]]}
        values = scores(name, model, x[val_idx], table.loc[val_idx].reset_index(drop=True))
        threshold = choose_threshold(labels_val, values)
        fitted[name] = (model, threshold)
        log.append({'model': name, 'threshold': threshold, **{k: v for k, v in extra.items() if k != 'history'}})
        if 'history' in extra:
            fitted['_cnn_history'] = extra['history']
    return fitted, log


def evaluate(labels, flags):
    tn, fp, fn, tp = confusion_matrix(labels, flags, labels=[0, 1]).ravel()
    return {'n': len(labels), 'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn),
            'recall': recall_score(labels, flags, zero_division=0),
            'precision': precision_score(labels, flags, zero_division=0),
            'f2': fbeta_score(labels, flags, beta=2, zero_division=0)}


# ---------------------------------------------------------------- 5. Ручные карточки (слепой тест)

def load_cards(annotations_path, cases_dir):
    annotations = pd.read_csv(annotations_path).set_index('sample_id')
    cards = {}
    for sample_id in annotations.index:
        card = pd.read_csv(cases_dir / f'{sample_id}.csv', index_col=0)
        card.index = pd.to_datetime(card.index, format='ISO8601')
        cards[sample_id] = card[['temperature_c']]
    return annotations, cards


def card_inputs(cards, sample_ids):
    tensors, tables, levels = [], [], []
    for sample_id in sample_ids:
        card = cards[sample_id].rename(columns={'temperature_c': sample_id})
        parts = represent(card, card.index[-1])
        tensors.append(to_tensor(parts))
        table = features(parts)
        table['level_c'] = amplitude(parts)[0]
        tables.append(table)
    return torch.cat(tensors), pd.concat(tables)


def exclusion_zones(annotations, matrices):
    """Места ручных карточек (+-15 координат) не используются как фон для обучения."""
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


# ---------------------------------------------------------------- 6. Сплошной просмотр и объединение в события

def scan(matrix, fitted, every_minutes=10, max_columns=1500):
    """Применить модели ко всей записи: окна каждые ~10 минут, координаты с шагом stride."""
    usable = matrix.columns[matrix.lt(INVALID_BELOW).sum().eq(0)]   # участок -300 идёт в отчёт качества, не в модель
    stride = max(1, math.ceil(len(usable) / max_columns))
    columns = list(usable[::stride])
    ends, last = [], None
    for end in window_ends(matrix):
        if last is None or end - last >= pd.Timedelta(minutes=every_minutes):
            ends.append(end)
            last = end
    rows_out = []
    for end in ends:
        rows = matrix.loc[(matrix.index > end - WINDOW) & (matrix.index <= end), columns]
        rows = rows.loc[:, rows.notna().all() & rows.ge(INVALID_BELOW).all()]
        parts = represent(rows, end)
        if parts is None or rows.empty:
            continue
        table = features(parts)
        level, spike = amplitude(parts)
        table['level_c'] = level
        x = to_tensor(parts)
        frame = pd.DataFrame({'end': end, 'coordinate': table.index, 'position': matrix.columns.get_indexer(table.index),
                              'level_c': level.values, 'spike_c': spike.values,
                              'signed_level_c': parts['median'].apply(lambda r: r.loc[r.abs().idxmax()], axis=1).values})
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, x, table.reset_index(drop=True))
            frame[name + '_score'] = values.values
            frame[name] = values.ge(threshold).values
        rows_out.append(frame)
    return pd.concat(rows_out, ignore_index=True), stride


def calibrate(grid, fitted, cutoff):
    """Порог каждой зоны = max(общий порог, 99-й перцентиль её оценок за пусконаладку).

    Так зона, где колебания на несколько градусов - обычное дело (1.csv), или шумный
    прибор (5.csv) не засыпают оператора. Цена: в таких зонах слабые события ловятся хуже,
    а событие, случившееся во время пусконаладки, поднимет порог своей зоны.
    Решение принимается только по прошлому (до cutoff) - будущее не используется.
    """
    early = grid.loc[grid.end <= cutoff]
    zone = early.position // ZONE_COORDINATES
    thresholds = {}
    for name in MODELS:
        quantile = early.groupby(zone)[name + '_score'].quantile(.99)
        thresholds[name] = quantile.clip(lower=fitted[name][1])
        grid[name + '_zones'] = grid[name + '_score'].ge(
            (grid.position // ZONE_COORDINATES).map(thresholds[name]).fillna(fitted[name][1]))
    return thresholds


def merge_events(grid, model, columns):
    """Соседние срабатывания (по координате и по времени) -> одно физическое событие."""
    flagged = grid.loc[grid[model]].copy()
    if flagged.empty:
        return pd.DataFrame()
    position = {c: i for i, c in enumerate(columns)}
    times = {t: i for i, t in enumerate(sorted(grid.end.unique()))}
    flagged['ci'] = flagged.coordinate.map(position)
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
            for dt in (-1, 0, 1):
                for dc in (-2, -1, 0, 1, 2):
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
    kind = events.peak_change_c.gt(0).map({True: 'нагрев', False: 'охлаждение'})
    events['description'] = kind.where(events.peak_change_c.abs().ge(1.5), 'краткий выброс')
    events['status'] = 'требуется проверка'
    events['cause'] = 'неизвестна'
    return events.reset_index()


# ---------------------------------------------------------------- 7. Эксперимент целиком

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
        print(f"прочитан {path.name}: {quality['profiles']} профилей x {quality['coordinates']} координат", flush=True)
    quality = pd.DataFrame(qualities)
    quality.to_csv(output / 'data_quality.csv', index=False, encoding='utf-8-sig')
    device_of = dict(zip(quality.file, quality.dts_id))

    annotations, cards = load_cards(annotations_path, cases_dir)
    # Контроль: карточки действительно вырезаны из этих исходников (те же числа).
    for sample_id, row in annotations.iterrows():
        matrix = matrices[row.source_file]
        column = matrix.columns[matrix.columns.get_indexer([row.coordinate], method='nearest')[0]]
        source = matrix.loc[cards[sample_id].index, column]
        assert (source.round(4) == cards[sample_id].temperature_c.round(4)).all(), sample_id
    zones = exclusion_zones(annotations, matrices)

    examples = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        rng = random.Random(seed + offset)
        ends = list(window_ends(matrix))
        chosen = sorted(rng.sample(ends, min(ends_per_file, len(ends))))
        examples[name] = make_examples(matrix, quality.set_index('file', drop=False).loc[name], chosen, columns_per_end,
                                       rng, zones.get(name, []), 'train')
        test_rng = random.Random(seed + 1000 + offset)
        test_ends = sorted(test_rng.sample(ends, min(ends_per_file // 2, len(ends))))
        examples[name + ':test'] = make_examples(matrix, quality.set_index('file', drop=False).loc[name], test_ends,
                                                 columns_per_end, test_rng, zones.get(name, []), 'test')
        print(f"{name}: обучающих окон {len(examples[name][2])}, тестовых {len(examples[name + ':test'][2])}", flush=True)
    pd.concat([examples[k][2] for k in examples]).to_csv(output / 'synthetic_passport.csv', index=False)

    binary = annotations.loc[annotations.label.isin(['событие', 'нет события'])].copy()
    binary['target'] = binary.label.eq('событие').astype(int)
    binary['device'] = binary.source_file.map(device_of)
    card_x, card_table = card_inputs(cards, binary.index)
    card_table = card_table.loc[binary.index].reset_index(drop=True)

    devices = sorted(set(device_of.values()))
    synthetic_rows, card_rows, load_rows, fit_logs, histories, all_events = [], [], [], [], [], []
    for fold, device in enumerate(devices, 1):
        train_files = [f for f in matrices if device_of[f] != device]
        test_files = [f for f in matrices if device_of[f] == device]
        x = torch.cat([examples[f][0] for f in train_files])
        table = pd.concat([examples[f][1] for f in train_files], ignore_index=True)
        passport = pd.concat([examples[f][2] for f in train_files], ignore_index=True)
        table['level_c'] = passport.level_c.values
        assert not set(passport.device) & {device}
        print(f'fold {fold}: проверяем {test_files}, обучаем на {train_files} ({len(passport)} окон)', flush=True)
        fitted, log = fit_all(x, table, passport, seed)
        histories += [dict(row, fold=fold) for row in fitted.pop('_cnn_history')]
        fit_logs += [dict(row, fold=fold, held_out=';'.join(test_files)) for row in log]

        # (а) синтетические события на фоне проверяемого прибора
        tx = torch.cat([examples[f + ':test'][0] for f in test_files])
        ttable = pd.concat([examples[f + ':test'][1] for f in test_files], ignore_index=True)
        tpass = pd.concat([examples[f + ':test'][2] for f in test_files], ignore_index=True)
        ttable['level_c'] = tpass.level_c.values
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, tx, ttable)
            tpass[name + '_score'] = values.values
            tpass[name] = values.ge(threshold).astype(int).values

        # (б) ручные карточки проверяемого прибора - слепой тест
        mask = binary.device.eq(device).to_list()
        idx = [i for i, m in enumerate(mask) if m]
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, card_x[idx], card_table.loc[idx].reset_index(drop=True))
            for sample_id, value in zip(binary.index[idx], values):
                card_rows.append({'sample_id': sample_id, 'model': name, 'fold': fold, 'score': float(value),
                                  'file': binary.loc[sample_id, 'source_file'],
                                  'threshold': threshold, 'prediction': int(value >= threshold),
                                  'target': int(binary.loc[sample_id, 'target'])})

        # (в) нагрузка на оператора: вся запись проверяемого прибора
        for name_file in test_files:
            grid, stride = scan(matrices[name_file], fitted)
            columns = list(grid.coordinate.drop_duplicates().sort_values())
            hours = (matrices[name_file].index[-1] - matrices[name_file].index[0]).total_seconds() / 3600
            matrix = matrices[name_file]
            cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * COMMISSIONING_SHARE
            zone_thresholds = calibrate(grid, fitted, cutoff)
            later = grid.loc[grid.end > cutoff].copy()
            later_hours = (matrix.index[-1] - cutoff).total_seconds() / 3600
            for name in MODELS:
                for mode, column in [('global', name), ('zones', name + '_zones')]:
                    events = merge_events(later, column, columns)
                    load_rows.append({'file': name_file, 'model': name, 'threshold_mode': mode, 'windows': len(later),
                                      'hours': later_hours, 'flagged_share': float(later[column].mean()),
                                      'events': len(events), 'events_per_hour': len(events) / later_hours,
                                      'coordinate_stride': stride})
                    if len(events):
                        all_events.append(events.assign(file=name_file, model=name, threshold_mode=mode, fold=fold))
                # Чувствительность после калибровки: синтетика этого файла после пусконаладки.
                in_file = tpass.file.eq(name_file)
                zone_of = tpass.position // ZONE_COORDINATES
                limit = zone_of.map(zone_thresholds[name]).fillna(fitted[name][1])
                tpass.loc[in_file, name + '_zones'] = tpass[name + '_score'].ge(limit).astype(int)
                tpass.loc[in_file, 'after_commissioning'] = tpass.end.gt(cutoff)
                for row in card_rows:
                    if row['file'] == name_file and row['model'] == name and row['fold'] == fold:
                        card = cards[row['sample_id']]
                        position = matrix.columns.get_indexer([annotations.loc[row['sample_id'], 'coordinate']], method='nearest')[0]
                        row['zone_threshold'] = float(zone_thresholds[name].get(position // ZONE_COORDINATES, row['threshold']))
                        row['prediction_zones'] = int(row['score'] >= row['zone_threshold'])
                        row['card_in_commissioning'] = bool(card.index[-1] <= cutoff)
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
        cards_part = card_predictions.loc[card_predictions.model.eq(name)]
        row = {'model': NAMES[name], **{f'cards_{k}': v for k, v in evaluate(cards_part.target, cards_part.prediction).items()},
               'cards_auc': roc_auc_score(cards_part.target, cards_part.score)}
        row.update({f'synthetic_{k}': v for k, v in evaluate(synthetic.label, synthetic[name]).items()})
        for kind in ['step', 'ramp', 'pulse', 'spike']:
            part = synthetic.loc[synthetic.kind.eq(kind) & synthetic.label.eq(1)]
            row[f'recall_{kind}'] = part[name].mean()
        late = synthetic.loc[synthetic.after_commissioning.eq(True)]
        row['zones_recall_events'] = late.loc[late.label.eq(1), name + '_zones'].mean()
        row['zones_recall_same_windows_global'] = late.loc[late.label.eq(1), name].mean()
        row['zones_false_alarm_quiet'] = late.loc[late.kind.eq('none'), name + '_zones'].mean()
        row['cards_zones_TP'] = int(cards_part.loc[cards_part.target.eq(1), 'prediction_zones'].sum())
        row['cards_zones_FP'] = int(cards_part.loc[cards_part.target.eq(0), 'prediction_zones'].sum())
        row['false_alarm_quiet'] = synthetic.loc[synthetic.kind.eq('none'), name].mean()
        row['false_alarm_small'] = synthetic.loc[synthetic.kind.eq('small'), name].mean()
        load = pd.DataFrame(load_rows)
        for mode in ['global', 'zones']:
            part = load.loc[load.model.eq(name) & load.threshold_mode.eq(mode)]
            row[f'events_per_hour_{mode}'] = part.events.sum() / part.hours.sum()
            row[f'flagged_share_{mode}'] = (part.flagged_share * part.windows).sum() / part.windows.sum()
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')

    # Финальные модели для применения: все приборы. Их ответы на обучающих данных не являются оценкой.
    x = torch.cat([examples[f][0] for f in matrices])
    table = pd.concat([examples[f][1] for f in matrices], ignore_index=True)
    passport = pd.concat([examples[f][2] for f in matrices], ignore_index=True)
    table['level_c'] = passport.level_c.values
    fitted, log = fit_all(x, table, passport, seed)
    fitted.pop('_cnn_history')
    (output / 'models').mkdir()
    cnn, cnn_threshold = fitted['cnn']
    torch.save({'state_dict': cnn.state_dict(), 'width': cnn.output.in_features // 2, 'threshold': cnn_threshold},
               output / 'models' / 'final_cnn.pt')
    joblib.dump({name: fitted[name] for name in MODELS if name != 'cnn'}, output / 'models' / 'final_other.joblib')
    pd.DataFrame(log).to_csv(output / 'final_settings.csv', index=False)
    protocol = {'seed': seed, 'window_minutes': 40, 'bin_minutes': 2, 'event_c': EVENT_C, 'small_c': SMALL_C,
                'evaluation': 'leave-one-device-out; human cards never used for training or thresholds',
                'pandas': pd.__version__, 'torch': torch.__version__,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'elapsed_seconds': time.perf_counter() - started}
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='FiberWatch v4: синтетика на реальном фоне + слепая проверка')
    parser.add_argument('--data-dir', type=Path, required=True, help='Папка с 1.csv ... 6.csv')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ends-per-file', type=int, default=110, help='Сколько моментов времени брать из каждого файла')
    args = parser.parse_args()
    run(args.data_dir, args.annotations, args.cases_dir, args.output, args.seed, args.ends_per_file)


if __name__ == '__main__':
    main()
