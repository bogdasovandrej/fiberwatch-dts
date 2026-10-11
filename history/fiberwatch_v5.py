"""FiberWatch v5: обнаружение температурных событий в данных DTS МГТС.

v5 относительно v4.1: свёрточная сеть по времени И координате (cnn2d) видит 9 соседних точек
волокна. Настоящее изменение температуры захватывает соседние точки (тепло распространяется,
прибор усредняет по длине), а шум и сбой измерения обычно затрагивают одну точку. Одноточечный
однозамерный выброс (как F5-02) считается вероятным сбоем измерения, а не температурным событием.
Обучение работает и без ручной разметки: она нужна только для слепой проверки.

v4.1 относительно v4.0: опорный уровень по первым 6 минутам окна, оценка шума окна как
отдельный вход, шумовая аугментация обучающего фона и метка события с учётом шума
(изменение >= max(3 °C, 4 x шум)). Причина правки - ложные тревоги v4.0 на шумном приборе 5.csv.

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
import sys
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
REFERENCE_SPAN = pd.Timedelta(minutes=6)  # опорный уровень: медиана первых 6 минут (не менее 3 замеров)
NOISE_FACTOR = 4.0                         # событие должно быть больше 4 «шумов» окна
CHANNELS = ['median_relative_c', 'min_relative_c', 'max_relative_c', 'observed', 'noise_c']
EVENT_C = 3.0      # политика команды: изменение >= 3 °C -> «требуется проверка»
SMALL_C = 1.5      # изменение <= 1.5 °C -> обычное поведение; между 1.5 и 3 -> пример не используем
ZONE_COORDINATES = 200   # зона = 200 соседних координат: порог настраивается под её обычное поведение
COMMISSIONING_SHARE = .3 # первые 30% записи прибора = «пусконаладка»: только наблюдаем, порог по зонам
RADIUS = 4               # cnn2d видит точку и по 4 соседа с каждой стороны: 9 координат
CNN_WIDTHS = (16, 32)    # ширины сети, из которых выбираем по проверочной части обучения
MAX_EPOCHS = 150
MODELS = ['amplitude_rule', 'noise_rule', 'range_rule', 'random_forest', 'cnn', 'cnn2d']
NAMES = {'amplitude_rule': 'Правило: отклонение медианы', 'noise_rule': 'Правило с учётом шума',
         'range_rule': 'Правило: размах (v3)', 'random_forest': 'Случайный лес', 'cnn': 'CNN по времени',
         'cnn2d': 'CNN время × координата'}


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

    median - уровень, min/max - короткие выбросы, observed - был ли в интервале замер.
    Опорный уровень = медиана замеров первых 6 минут окна, но не менее трёх замеров
    (не «норма», а точка отсчёта). При шаге 2 минуты это ровно три замера, как в v3;
    при шаге 10 секунд - 36 замеров вместо 20 секунд, поэтому шум меньше сдвигает опору.
    noise - оценка шума окна по соседним разностям (робастная, выбросы на неё почти не влияют).
    """
    start = end - WINDOW
    rows = rows.loc[(rows.index > start) & (rows.index <= end)]
    if len(rows) < 3 or rows.index.to_series().diff().gt(MAX_GAP).any():
        return None
    position = (rows.index - start) / BIN
    bin_number = [int(-((-value) // 1)) for value in position]   # округление вверх: интервал (k-1, k]
    reference = reference_level(rows, start)
    relative = rows - reference
    # Series, а не список: список pandas принял бы за названия столбцов, если координаты - целые
    # числа 1..20 (так и есть в 5.csv), и сгруппировал бы не по времени.
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
    """Медиана замеров первых 6 минут окна (если их меньше трёх - первых трёх замеров)."""
    early = rows.loc[rows.index <= start + REFERENCE_SPAN]
    return (early if len(early) >= 3 else rows.iloc[:3]).median()


def noise_sigma(rows):
    """Шум окна, °C: 1.4826 * медиана |разности соседних замеров| / sqrt(2).

    Для белого шума это оценка его стандартного отклонения. Медиана вместо среднего -
    чтобы сам нагрев или одиночный выброс почти не увеличивал оценку шума.
    """
    return rows.diff().abs().median() * 1.4826 / math.sqrt(2)


def to_tensor(parts):
    """[число окон, 5 каналов, 20 интервалов]; температуры в единицах 10 °C, как в v3.

    Пятый канал - шум окна, повторённый по всем 20 интервалам: свёртке так проще его учесть.
    """
    stacked = [torch.tensor(parts[name].to_numpy(dtype='float32')) / (10. if name != 'observed' else 1.)
               for name in ['median', 'min', 'max', 'observed']]
    noise = torch.tensor(parts['noise'].to_numpy(dtype='float32')).unsqueeze(1).expand(-1, N_BINS) / 10.
    return torch.stack(stacked + [noise], dim=1)


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
        'noise_c': parts['noise'],
    })
    return table.loc[parts['median'].index].fillna(0.)


def amplitude(parts):
    """Наибольшее отклонение медианы от опорного уровня и наибольший выброс min/max, °C."""
    level = parts['median'].abs().max(axis=1)
    spike = pd.concat([parts['max'].max(axis=1), -parts['min'].min(axis=1)], axis=1).max(axis=1)
    return level, spike


def subset(parts, columns):
    """Представление только для выбранных координат (признаки леса считаем не для всех)."""
    return {k: v.loc[columns] for k, v in parts.items()}


def neighbours(names, position):
    """Метки координат position-RADIUS..position+RADIUS; за краем волокна - None."""
    return [names[i] if 0 <= i < len(names) else None for i in range(position - RADIUS, position + RADIUS + 1)]


def patches(parts, neighbour_lists):
    """[окна, 5 каналов, 9 соседних координат, 20 интервалов] для cnn2d.

    Сосед, которого нет (край волокна, пропуск, -300), остаётся нулём с observed = 0:
    модель видит, что там нет данных, а не «температура не менялась».
    """
    full = to_tensor(parts)
    position = {c: i for i, c in enumerate(parts['median'].index)}
    index = torch.tensor([[position.get(c, 0) for c in row] for row in neighbour_lists])
    present = torch.tensor([[c in position for c in row] for row in neighbour_lists], dtype=torch.float32)
    out = full[index] * present[:, :, None, None]
    return out.permute(0, 2, 1, 3).contiguous()


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
    elif kind == 'glitch':
        shape = torch.zeros_like(t)
        first = int(started.nonzero()[0]) if started.any() else len(t) - 1
        shape[first] = 1.
    else:
        raise ValueError(kind)
    return (size * shape * started).tolist()


def make_examples(matrix, quality, ends, columns_per_end, rng, excluded, purpose, augment_noise=False, keep_raw=False):
    """Окна реального фона + синтетические события. Возвращает (x, x2, признаки, паспорт[, сырые ряды]).

    Метка задаётся физически измеримой реализованной амплитудой добавки в центральной точке:
    >= max(3 °C, 4 x шум окна) -> 1 («требуется проверка»), <= 1.5 °C -> 0, промежуточные
    отбрасываем. Фон берём только «спокойный», чтобы настоящее событие не стало ответом 0.
    Изменение захватывает соседние точки по колоколу exp(-d^2 / 2w^2) случайной ширины w.
    glitch - одна точка и один замер: вероятный сбой измерения, метка 0 (не температура).
    augment_noise (только для обучения): к половине окрестностей добавляем белый шум 0.1-1 °C,
    свой в каждой точке - так выглядит шум прибора.
    """
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
        for i in candidates:     # центры не ближе 2*RADIUS+2: добавки соседних центров не пересекаются
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
        quiet = [t for t in targets if natural_level[t] < 1.5 and natural_spike[t] < 4.]
        minutes = ((rows.index - start) / pd.Timedelta(minutes=1)).tolist()
        delta = pd.DataFrame(0., index=rows.index, columns=local)
        added = pd.DataFrame(0., index=rows.index, columns=local)
        meta = []
        for target in quiet:
            members = [c for c in lists[target] if c is not None and valid[c]]
            noise_added = 0.
            if augment_noise and rng.random() < .5:
                noise_added = rng.uniform(.1, 1.)
                generator = torch.Generator().manual_seed(rng.randrange(2 ** 31))
                noise = torch.randn(len(rows), len(members), generator=generator, dtype=torch.float64) * noise_added
                added[members] = pd.DataFrame(noise.tolist(), index=rows.index, columns=members)
            sigma = float(noise_sigma((rows[target] + added[target]).to_frame()).iloc[0])
            event_threshold = max(EVENT_C, NOISE_FACTOR * sigma)
            kind = rng.choices(['none', 'small', 'step', 'ramp', 'pulse', 'spike', 'glitch'],
                               [30, 13, 13, 11, 11, 10, 12])[0]
            realized, width = 0., 0.
            if kind != 'none':
                low, high = (.3, SMALL_C) if kind == 'small' else (EVENT_C, 12.)
                if kind not in ('small', 'glitch'):
                    low = min(event_threshold, high - .5)
                size = math.exp(rng.uniform(math.log(low), math.log(high))) * rng.choice([-1, 1])
                onset = rng.uniform(8., 39.5)
                values = pd.Series(injected_shape(kind, minutes, onset, size, rng), index=rows.index)
                realized = float((values - reference_level(values.to_frame(), start).iloc[0]).abs().max())
                if kind != 'glitch':
                    low_w, high_w = (.8, 6.) if kind == 'spike' else (.7, 15.)
                    width = math.exp(rng.uniform(math.log(low_w), math.log(high_w)))
                for c in members:
                    offset = position[c] - position[target]
                    weight = 1. if offset == 0 else (math.exp(-offset ** 2 / (2 * width ** 2)) if width > 0 else 0.)
                    delta[c] = delta[c] + values * weight
            if kind == 'glitch':
                label = 0
            else:
                label = 1 if realized >= event_threshold else (0 if realized <= SMALL_C else None)
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


# ---------------------------------------------------------------- 4. Модели

class TemperatureCNN(nn.Module):
    """Та же архитектура, что в v3: две свёртки по времени, затем среднее и максимум по окну."""
    def __init__(self, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv1d(len(CHANNELS), width, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(width, width, 3, padding=1), nn.ReLU())
        self.output = nn.Linear(width * 2, 1)

    def forward(self, inputs):
        patterns = self.filters(inputs)
        return self.output(torch.cat([patterns.mean(dim=2), patterns.amax(dim=2)], dim=1)).squeeze(1)


class SpatialCNN(nn.Module):
    """Свёртки 3x3 по координате и времени: видит, затронуло ли изменение соседние точки.

    Вход [окна, 5, 9, 20]. Итог: среднее и максимум откликов по всему участку плюс
    максимум по центральной точке - ответ относится именно к ней.
    """
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


def train_cnn(x_train, y_train, x_val, y_val, width, seed, factory=TemperatureCNN, max_epochs=None, patience=10):
    """Мини-пакеты по 256 окон; лучшая эпоха выбирается по ошибке на проверочной части обучения."""
    max_epochs = max_epochs or MAX_EPOCHS
    torch.manual_seed(seed)
    model = factory(width)
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
    if name == 'noise_rule':
        # Отклонение медианы в долях порога max(3 °C, 4 x шум): 1.0 = ровно на пороге.
        return pd.Series((table['level_c'] / (table['noise_c'] * NOISE_FACTOR).clip(lower=EVENT_C)).values)
    if name == 'range_rule':
        return pd.Series(table['range_c'].values)
    if name == 'random_forest':
        return pd.Series(model.predict_proba(table[FEATURES])[:, 1])
    with torch.no_grad():
        return pd.Series(torch.sigmoid(model(x2 if name == 'cnn2d' else x)).tolist())


FEATURES = ['range_c', 'median_range_c', 'std_c', 'net_change_c', 'absolute_net_change_c',
            'max_absolute_rate_c_per_min', 'mean_absolute_rate_c_per_min', 'mean_bin_spread_c',
            'max_bin_spread_c', 'observed_fraction', 'noise_c']


def choose_threshold(labels, values):
    """Порог с максимальным F2 (полнота важнее точности - решение команды «меньше пропусков»)."""
    candidates = sorted(set(values.quantile([i / 200 for i in range(1, 200)]).round(6)))
    best = max(candidates, key=lambda t: (fbeta_score(labels, values.ge(t), beta=2, zero_division=0), t))
    return float(best)


def fit_all(x, x2, table, passport, seed):
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
        elif name in ('cnn', 'cnn2d'):
            # Ширину выбираем по ошибке на проверочной части, а не по тестовому прибору.
            inputs, factory = (x, TemperatureCNN) if name == 'cnn' else (x2, SpatialCNN)
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


# ---------------------------------------------------------------- 5. Ручные карточки (слепой тест)

def load_cards(annotations_path, cases_dir):
    annotations = pd.read_csv(annotations_path).set_index('sample_id')
    cards = {}
    for sample_id in annotations.index:
        card = pd.read_csv(cases_dir / f'{sample_id}.csv', index_col=0)
        card.index = pd.to_datetime(card.index, format='ISO8601')
        cards[sample_id] = card[['temperature_c']]
    return annotations, cards


def card_inputs(cards, sample_ids, annotations, matrices):
    """Ручные фрагменты в виде входов моделей; соседи для cnn2d берутся из исходного файла."""
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
        table['level_c'] = amplitude(own)[0].values
        tables.append(table)
    return torch.cat(tensors), torch.cat(tensors2), pd.concat(tables)


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
        parts = represent(rows, end)        # все точки: соседи нужны cnn2d
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
        frame = pd.DataFrame({'end': end, 'coordinate': table.index, 'position': matrix.columns.get_indexer(table.index),
                              'level_c': level.values, 'spike_c': spike.values,
                              'signed_level_c': parts['median'].apply(lambda r: r.loc[r.abs().idxmax()], axis=1).values})
        for name in MODELS:
            model, threshold = fitted[name]
            values = scores(name, model, x, table.reset_index(drop=True), x2)
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

    if annotations_path is not None and cases_dir is not None:
        annotations, cards = load_cards(annotations_path, cases_dir)
    else:
        print('Разметка не задана: обучаемся на синтетике, слепой тест на ручных фрагментах пропускаем', flush=True)
        annotations, cards = pd.DataFrame(columns=['source_file', 'coordinate', 'start', 'end', 'label']), {}
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

        # (а) синтетические события на фоне проверяемого прибора
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

        # (б) ручные карточки проверяемого прибора - слепой тест
        mask = binary.device.eq(device).to_list()
        idx = [i for i, m in enumerate(mask) if m]
        for name in MODELS if idx else []:
            model, threshold = fitted[name]
            values = scores(name, model, card_x[idx], card_table.loc[idx].reset_index(drop=True), card_x2[idx])
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
        row = {'model': NAMES[name]}
        cards_part = card_predictions.loc[card_predictions.model.eq(name)] if len(card_predictions) else pd.DataFrame()
        if len(cards_part):
            row.update({f'cards_{k}': v for k, v in evaluate(cards_part.target, cards_part.prediction).items()})
            row['cards_auc'] = roc_auc_score(cards_part.target, cards_part.score)
            row['cards_zones_TP'] = int(cards_part.loc[cards_part.target.eq(1), 'prediction_zones'].sum())
            row['cards_zones_FP'] = int(cards_part.loc[cards_part.target.eq(0), 'prediction_zones'].sum())
        row.update({f'synthetic_{k}': v for k, v in evaluate(synthetic.label, synthetic[name]).items()})
        for kind in ['step', 'ramp', 'pulse', 'spike']:
            part = synthetic.loc[synthetic.kind.eq(kind) & synthetic.label.eq(1)]
            row[f'recall_{kind}'] = part[name].mean()
        late = synthetic.loc[synthetic.after_commissioning.eq(True)]
        row['zones_recall_events'] = late.loc[late.label.eq(1), name + '_zones'].mean()
        row['zones_recall_same_windows_global'] = late.loc[late.label.eq(1), name].mean()
        row['zones_false_alarm_quiet'] = late.loc[late.kind.eq('none'), name + '_zones'].mean()
        row['false_alarm_glitch'] = synthetic.loc[synthetic.kind.eq('glitch'), name].mean()
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
    x2 = torch.cat([examples[f][1] for f in matrices])
    table = pd.concat([examples[f][2] for f in matrices], ignore_index=True)
    passport = pd.concat([examples[f][3] for f in matrices], ignore_index=True)
    table['level_c'] = passport.level_c.values
    fitted, log = fit_all(x, x2, table, passport, seed)
    fitted.pop('_cnn_history')
    (output / 'models').mkdir()
    for name in ('cnn', 'cnn2d'):
        network, network_threshold = fitted[name]
        torch.save({'state_dict': network.state_dict(), 'width': network.filters[0].out_channels,
                    'threshold': network_threshold}, output / 'models' / f'final_{name}.pt')
    joblib.dump({name: fitted[name] for name in MODELS if name not in ('cnn', 'cnn2d')}, output / 'models' / 'final_other.joblib')
    pd.DataFrame(log).to_csv(output / 'final_settings.csv', index=False)
    protocol = {'version': '5.0', 'radius': RADIUS, 'cnn_widths': list(CNN_WIDTHS), 'max_epochs': MAX_EPOCHS, 'seed': seed, 'window_minutes': 40, 'bin_minutes': 2, 'event_c': EVENT_C, 'small_c': SMALL_C,
                'evaluation': 'leave-one-device-out; human cards never used for training or thresholds',
                'pandas': pd.__version__, 'torch': torch.__version__,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'elapsed_seconds': time.perf_counter() - started}
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.round(3).to_string(index=False))


# ---------------------------------------------------------------- 8. Графики для отчёта

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'amplitude_rule': '#8a8f98', 'noise_rule': '#7b5ea7', 'range_rule': '#c9a227', 'random_forest': '#2f7d4f',
          'cnn': '#6fa0e0', 'cnn2d': '#1d4f91'}
PLOT_STYLE = ({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def separability(cards_dir, annotations, out):
    """Почему v3 нельзя было честно сравнить: 28 карточек делятся одним числом."""
    rows = []
    for sample_id, row in annotations.iterrows():
        if row.label not in ('событие', 'нет события'):
            continue
        card = pd.read_csv(cards_dir / f'{sample_id}.csv', index_col=0)
        card.index = pd.to_datetime(card.index, format='ISO8601')
        parts = represent(card[['temperature_c']], card.index[-1])
        level, spike = amplitude(parts)
        rows.append({'id': sample_id, 'level': float(level.iloc[0]), 'spike': float(spike.iloc[0]),
                     'event': row.label == 'событие'})
    table = pd.DataFrame(rows).sort_values('level')
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for event, color, name in [(False, '#8a8f98', 'нет события'), (True, '#c0392b', 'событие')]:
        part = table.loc[table.event.eq(event)]
        ax.scatter(part.level, [1 if event else 0] * len(part), s=60, color=color, label=name, zorder=3)
        for _, r in part.iterrows():
            ax.annotate(r.id, (r.level, 1 if event else 0), textcoords='offset points',
                        xytext=(0, 9 if event else -14), ha='center', fontsize=7, rotation=60 if event else -60)
    ax.axvspan(1.35, 2.43, color='#2f7d4f', alpha=.12)
    ax.text(1.89, .5, 'любой порог\nв этой полосе\nдаёт 27/28', ha='center', va='center', fontsize=9, color='#2f7d4f')
    ax.set_xscale('log')
    ax.set_yticks([0, 1], ['нет события', 'событие'])
    ax.set_ylim(-.9, 1.9)
    ax.set_xlabel('Наибольшее отклонение медианы от опорного уровня за 40 минут, °C (лог. шкала)')
    ax.set_title('28 размеченных фрагментов почти делятся одним числом (исключение — провал F5-02)')
    fig.tight_layout()
    fig.savefig(out / 'fig1_cards_separability.png', dpi=150)
    plt.close(fig)


def injection_example(data_dir, out):
    """Как делается обучающий пример: настоящий фон + добавленное изменение."""
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


def model_comparison(run, out):
    summary = pd.read_csv(run / 'summary.csv')
    names = [NAMES[m] for m in MODELS]
    summary = summary.set_index('model').loc[names]
    colors = [COLORS[m] for m in MODELS]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [
        (summary.cards_TP, f'Слепой тест: найдено событий\nиз {int(summary.cards_TP.iloc[0] + summary.cards_FN.iloc[0])} размеченных', '{:.0f}'),
        (summary.cards_FP, f'Слепой тест: ложных срабатываний\nна {int(summary.cards_TN.iloc[0] + summary.cards_FP.iloc[0])} спокойных', '{:.0f}'),
        (summary.synthetic_recall * 100, 'Синтетика на чужом приборе:\nнайдено событий, %', '{:.1f}'),
        (summary.false_alarm_quiet * 100, 'Синтетика на чужом приборе:\nложные тревоги на спокойном фоне, %', '{:.1f}'),
    ]
    for ax, (values, title, fmt) in zip(axes, panels):
        ax.barh(range(len(names)), values.values, color=colors)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [''] * len(names))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(values.values):
            ax.text(v, i, ' ' + fmt.format(v), va='center', fontsize=9)
        ax.set_xlim(0, max(values.max() * 1.25, 1))
    fig.suptitle('Сравнение на приборе, который модель не видела (leave-one-device-out)', y=1.04)
    fig.tight_layout()
    fig.savefig(out / 'fig3_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def recall_by_amplitude(run, out):
    synthetic = pd.read_csv(run / 'synthetic_test_predictions.csv')
    events = synthetic.loc[synthetic.label.eq(1)].copy()
    events['bin'] = pd.cut(events.realized_c, [3, 4, 5, 7, 13], right=False,
                           labels=['3–4 °C', '4–5 °C', '5–7 °C', '7–12 °C'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, kinds, title in [(axes[0], ['step', 'ramp', 'pulse'], 'Нагревы и охлаждения'),
                             (axes[1], ['spike'], 'Краткие выбросы (1–2 замера)')]:
        part = events.loc[events.kind.isin(kinds)]
        table = part.groupby('bin', observed=False)[MODELS].mean() * 100
        width = .8 / len(MODELS)
        for i, model in enumerate(MODELS):
            ax.bar([x + (i - (len(MODELS) - 1) / 2) * width for x in range(len(table))], table[model], width,
                   color=COLORS[model], label=NAMES[model])
        ax.set_xticks(range(len(table)), table.index)
        ax.set_title(title)
        ax.set_xlabel('амплитуда добавленного изменения')
    axes[0].set_ylabel('найдено, %')
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('Чувствительность по амплитуде (синтетика на приборе, которого не было в обучении)', y=1.03)
    fig.tight_layout()
    fig.savefig(out / 'fig4_recall_by_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def operator_load(run, out):
    load = pd.read_csv(run / 'operator_load.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    files = sorted(load.file.unique())
    for ax, mode, title in [(axes[0], 'global', 'Единый порог для всех объектов'),
                            (axes[1], 'zones', 'Порог по зонам после пусконаладки')]:
        part = load.loc[load.threshold_mode.eq(mode)].pivot(index='file', columns='model', values='events_per_hour').loc[files]
        width = .8 / len(MODELS)
        for i, model in enumerate(MODELS):
            ax.bar([x + (i - (len(MODELS) - 1) / 2) * width for x in range(len(files))], part[model], width,
                   color=COLORS[model], label=NAMES[model])
        ax.set_xticks(range(len(files)), files)
        ax.set_title(title)
        ax.set_yscale('symlog', linthresh=1)
    axes[0].set_ylabel('событий в час на всё волокно')
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle('Нагрузка на оператора на сплошной записи (последние 70% времени каждого файла)', y=1.03)
    fig.tight_layout()
    fig.savefig(out / 'fig5_operator_load.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def learning_curves(run, out):
    history = pd.read_csv(run / 'cnn_training_history.csv')
    settings = pd.read_csv(run / 'fold_settings.csv')
    chosen = settings.loc[settings.model.eq('cnn2d')].set_index('fold').width
    files = settings.loc[settings.model.eq('cnn2d')].set_index('fold').held_out
    history = history.loc[history.model.eq('SpatialCNN')]
    fig, axes = plt.subplots(1, len(chosen), figsize=(14, 3), sharey=True)
    for ax, (fold, width) in zip(axes, chosen.items()):
        part = history.loc[history.fold.eq(fold) & history.width.eq(width)]
        ax.plot(part.epoch, part.training_loss, label='обучение', color='#2462a8')
        ax.plot(part.epoch, part.validation_loss, label='проверка', color='#c0392b')
        ax.set_title(f'без {files[fold]}\nширина {int(width)}', fontsize=9)
        ax.set_xlabel('эпоха')
    axes[0].set_ylabel('ошибка (BCE)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig6_cnn_learning.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def heatmap(run, data_dir, out, name='4.csv', model='cnn2d'):
    """Карта «координата x время» и найденные события - то, что увидит оператор."""
    matrix, _ = read_source(data_dir / name)
    events = pd.read_csv(run / 'events.csv')
    events = events.loc[events.file.eq(name) & events.model.eq(model) & events.threshold_mode.eq('zones')].copy()
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
    ax.set_xlabel('координата вдоль волокна (единица не подтверждена)')
    ax.set_ylabel(f'часы от начала записи ({shown.index[0]})')
    ax.set_title(f'{name}: голубые рамки — события {NAMES[model]} после пусконаладки (пунктир), {len(events)} шт.')
    fig.tight_layout()
    fig.savefig(out / f'fig7_heatmap_{name.replace(".csv", "")}.png', dpi=150)
    plt.close(fig)




def v3_vs_v4(table_path, out):
    """До и после дообучения на одних и тех же окнах чужих приборов (результат compare_v3.py)."""
    table = pd.read_csv(table_path).set_index('model')
    order = ['cnn_v3', 'random_forest_v3', 'policy_3c', 'random_forest', 'cnn', 'cnn2d']
    labels = ['CNN v3 (28 карточек)', 'Лес v3 (28 карточек)', 'Правило «3 °C»', 'Лес v5',
              'CNN v5 по времени', 'CNN v5 время × координата']
    colors = ['#9fb7d9', '#9fc9ae', '#8a8f98', '#2f7d4f', '#6fa0e0', '#1d4f91']
    table = table.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, column, title in [(axes[0], 'recall', 'найдено событий, %'),
                              (axes[1], 'precision', 'верных среди тревог, %'),
                              (axes[2], 'false_alarm_glitch', 'тревоги на одноточечных сбоях, %')]:
        values = table[column] * 100
        ax.barh(range(len(order)), values, color=colors)
        ax.set_yticks(range(len(order)), labels)
        ax.invert_yaxis()
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.1f}', va='center', fontsize=9)
        ax.set_xlim(0, 118)
    fig.suptitle('До и после дообучения: одни и те же окна приборов, не участвовавших в обучении', y=1.03)
    fig.tight_layout()
    fig.savefig(out / 'fig8_v3_vs_v4.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_figures(run, data_dir, annotations_path, cases_dir, v3_table=None):
    """Все графики отчёта по сохранённым результатам run (обучение не повторяется)."""
    plt.rcParams.update(PLOT_STYLE)
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    if annotations_path is not None and cases_dir is not None:
        separability(cases_dir, pd.read_csv(annotations_path).set_index('sample_id'), out)
    injection_example(data_dir, out)
    if (run / 'card_predictions.csv').exists() and (run / 'card_predictions.csv').stat().st_size > 10:
        model_comparison(run, out)
    recall_by_amplitude(run, out)
    operator_load(run, out)
    learning_curves(run, out)
    for name in ['4.csv', '1.csv', '5.csv']:
        heatmap(run, data_dir, out, name)
    if v3_table is not None and Path(v3_table).exists():
        v3_vs_v4(v3_table, out)
    tracks_figure(run / 'tracks', out)
    print('графики сохранены в', out)


# ---------------------------------------------------------------- 9. Движущиеся и точечные нагревы

def hot_points(matrix, rise=4.0, history='6min'):
    """Замеры, которые на >= rise °C выше медианы СВОЕЙ ЖЕ координаты за прошлые 6 минут.

    Это быстрый детектор кратковременных локальных нагревов (секунды-минуты): они видны
    только в записях с частым шагом. Используется только прошлое (closed='left').
    """
    valid = matrix.loc[:, matrix.lt(INVALID_BELOW).sum().eq(0)]
    baseline = valid.rolling(history, min_periods=3, closed='left').median()
    excess = (valid - baseline).where((valid - baseline).ge(rise)).stack().rename('excess_c').reset_index()
    excess.columns = ['time', 'coordinate', 'excess_c']
    excess['minutes'] = (excess.time - matrix.index[0]).dt.total_seconds() / 60
    return excess


def heat_tracks(points, seconds_scale=10., eps=12., min_points=4, link_minutes=3., link_coordinates=60.):
    """Горячие точки -> пятна (DBSCAN по «время x координата») -> треки (цепочки пятен).

    Пятно - один локальный нагрев. Трек - пятна, которые следуют друг за другом не дальше
    link_minutes по времени и link_coordinates по координате. Для трека из >= 3 пятен
    считаем скорость (координат в минуту) линейной аппроксимацией: знак = направление.
    Единица координаты не подтверждена, поэтому скорость - в «координатах/мин».
    """
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
                      # -2: пятна одной «пары» (волокно дважды проходит рядом с источником) идут одновременно
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
                     'speed_coord_per_min': speed, 'linearity_r': r, 'peak_excess_c': frame.peak_excess_c.max(),
                     'direction': 'к большим координатам' if speed > 0 else 'к меньшим координатам',
                     'description': 'движущийся локальный нагрев' if abs(r) >= .8 else 'серия локальных нагревов',
                     'status': 'требуется проверка', 'cause': 'неизвестна'})
    return spots, pd.DataFrame(rows)


def find_tracks(data_dir, output):
    """Поиск кратких и движущихся нагревов во всех файлах; таблицы в output/tracks_*.csv."""
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
                ax.annotate(f'{track.speed_coord_per_min:+.0f} коорд./мин', ((track.start_min + track.end_min) / 2,
                            (track.coordinate_start + track.coordinate_end) / 2), color='#2462a8', fontsize=9,
                            xytext=(6, 0), textcoords='offset points')
        ax.set_xlabel('минуты от начала записи 5.csv')
        ax.set_ylabel('координата вдоль волокна')
    fig.colorbar(image, ax=axes, label='на сколько °C выше своего прошлого (6 мин)')
    fig.suptitle('5.csv: краткие локальные нагревы +4…+14 °C и их перемещение вдоль волокна (синие линии - треки)')
    fig.savefig(out / 'fig9_moving_heat_5.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def glitch_points(matrix, jump=4.0, calm=1.5, history='6min'):
    """Вероятные сбои измерения: одна точка на один замер отскочила на >= jump °C от своей
    медианы за 6 минут, оба соседа по волокну остались в пределах calm °C, и следующий замер
    вернулся. Тепловая инерция кабеля не даёт реальной температуре так себя вести.
    Следующий замер - это задержка на один профиль, не просмотр будущего записи целиком.
    """
    valid = matrix.loc[:, matrix.notna().all() & matrix.lt(INVALID_BELOW).sum().eq(0)]
    deviation = valid - valid.rolling(history, min_periods=3, closed='left').median()
    isolated = (deviation.abs().ge(jump) & deviation.shift(1, axis=1).abs().lt(calm)
                & deviation.shift(-1, axis=1).abs().lt(calm) & deviation.shift(-1).abs().lt(calm))
    found = deviation.where(isolated).stack().rename('deviation_c').reset_index()
    found.columns = ['time', 'coordinate', 'deviation_c']
    return found


# ---------------------------------------------------------------- 10. Применение к новому файлу

def load_final(model_dir):
    """Итоговые модели (обучены на всех приборах) и их пороги."""
    fitted = joblib.load(model_dir / 'final_other.joblib')
    for name, factory in (('cnn', TemperatureCNN), ('cnn2d', SpatialCNN)):
        saved = torch.load(model_dir / f'final_{name}.pt', map_location='cpu', weights_only=True)
        network = factory(saved['width'])
        network.load_state_dict(saved['state_dict'])
        network.eval()
        fitted[name] = (network, float(saved['threshold']))
    return fitted


def detect(path, model_dir, output, model='cnn2d'):
    """CSV прибора -> события для оператора.

    Первые 30% записи - пусконаладка: события ищутся с общим порогом и помечаются,
    по ним же настраиваются пороги зон. Дальше - пороги зон. Отдельно: качество данных
    (-300 и разрывы) и краткие/движущиеся нагревы.
    """
    if output.exists() and any(output.iterdir()):
        raise ValueError('Папка результата не пуста')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fitted = load_final(model_dir)
    matrix, quality = read_source(path)
    grid, stride = scan(matrix, fitted)
    cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * COMMISSIONING_SHARE
    calibrate(grid, fitted, cutoff)
    columns = list(grid.coordinate.drop_duplicates().sort_values())
    early = merge_events(grid.loc[grid.end <= cutoff], model, columns)
    later = merge_events(grid.loc[grid.end > cutoff], model + '_zones', columns)
    events = pd.concat([early.assign(period='пусконаладка: общий порог'),
                        later.assign(period='работа: порог зоны')], ignore_index=True)
    invalid = matrix.columns[matrix.lt(INVALID_BELOW).any()]
    messages = []
    if len(invalid):
        messages.append({'type': 'недостоверные данные', 'coordinate_from': float(invalid.min()),
                         'coordinate_to': float(invalid.max()), 'coordinates': len(invalid),
                         'description': f'значения ниже {INVALID_BELOW} °C физически невозможны', 'status': 'требуется проверка'})
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
    points = hot_points(matrix)
    spots, tracks = heat_tracks(points)
    events.to_csv(output / 'events.csv', index=False, encoding='utf-8-sig')
    tracks.to_csv(output / 'moving_heat.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(messages).to_csv(output / 'data_quality_messages.csv', index=False, encoding='utf-8-sig')
    (output / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'file': path.name, 'model': model, 'events': len(events),
                      'events_after_commissioning': len(later), 'moving_heat_tracks': len(tracks),
                      'data_quality_messages': len(messages), 'output': str(output)}, ensure_ascii=False, indent=2))
    if len(events):
        first = events.sort_values('peak_change_c', key=abs, ascending=False).iloc[0]
        print(f"Пример: {first.first_alert} — координаты {first.coordinate_from:.1f}–{first.coordinate_to:.1f} — "
              f"{first.description} {first.peak_change_c:+.1f} °C — {first.status}; причина {first.cause}")


def main():
    # Консоль Windows может не уметь печатать отдельные символы: заменяем их, а не падаем.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description='FiberWatch v5: обнаружение температурных событий по данным DTS')
    parser.add_argument('--data-dir', type=Path, help='Папка с 1.csv ... 6.csv')
    parser.add_argument('--figures-only', action='store_true', help='Только построить графики по готовой папке --output')
    parser.add_argument('--tracks', action='store_true', help='Найти краткие и движущиеся нагревы (папка --output/tracks)')
    parser.add_argument('--detect', type=Path, help='Применить итоговую модель к CSV прибора; нужна --model-dir')
    parser.add_argument('--model-dir', type=Path, help='Папка models/ обученного прогона')
    parser.add_argument('--v3-table', type=Path, help='v3_vs_v4_synthetic.csv для графика «до/после»')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--cases-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ends-per-file', type=int, default=110, help='Сколько моментов времени брать из каждого файла')
    parser.add_argument('--model', choices=MODELS, default='cnn2d', help='Модель для --detect')
    parser.add_argument('--quick', action='store_true', help='Быстрая проверка запуска (~3 мин): мало окон, короткое обучение')
    args = parser.parse_args()
    if args.quick:
        global CNN_WIDTHS, MAX_EPOCHS
        CNN_WIDTHS, MAX_EPOCHS, args.ends_per_file = (16,), 15, 12
    if args.detect:
        if args.model_dir is None:
            parser.error('--detect требует --model-dir')
        detect(args.detect, args.model_dir, args.output, args.model)
        return
    if args.data_dir is None:
        parser.error('для обучения нужна --data-dir (папка с 1.csv ... 6.csv); разметка - по желанию')
    if args.tracks:
        find_tracks(args.data_dir, args.output / 'tracks')
    elif not args.figures_only:
        run(args.data_dir, args.annotations, args.cases_dir, args.output, args.seed, args.ends_per_file)
        find_tracks(args.data_dir, args.output / 'tracks')
    render_figures(args.output, args.data_dir, args.annotations, args.cases_dir, args.v3_table)


if __name__ == '__main__':
    main()
