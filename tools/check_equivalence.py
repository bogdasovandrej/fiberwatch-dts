"""Проверки fiberwatch.py (v4.1) против temperature_model.py (v3) и на утечку данных.

1. При шаге 2 минуты опорный уровень v4.1 = три первых замера, как в v3 -> представление и
   признаки обязаны совпасть точно.
2. При более частом шаге отличается только опорный уровень: после выравнивания опоры
   представления снова совпадают.
3. Окно зависит только от прошлого: изменение будущих замеров не меняет представление.
4. Оценка шума: на синтетическом белом шуме sigma=0.5 °C получается около 0.5.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

here = Path(__file__).resolve().parent
root = here.parent if here.name == 'tools' else here   # корень проекта: fiberwatch.py, history/
sys.path[:0] = [str(root), str(root / 'history')]

def locate(*candidates):
    """Первый существующий путь: скрипт работает и в папке v4, и в репозитории (pipeline_v4/)."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('не найдено: ' + ', '.join(str(c) for c in candidates))

V3_SOURCE = locate(root / 'history' / 'temperature_model.py', here.parent / 'temperature_model.py', here.parent / 'fiberwatch' / 'temperature_model.py')
PRIVATE = locate(root.parent / 'private_learning', here.parent / 'private_learning', here.parent.parent / 'private_learning')
import fiberwatch as fw

spec = importlib.util.spec_from_file_location('v3', V3_SOURCE)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
cases = PRIVATE / 'model_v3_bundle' / 'data' / 'cases'

exact, shifted = [], []
for path in sorted(cases.glob('*.csv')):
    card = pd.read_csv(path, index_col=0)
    card.index = pd.to_datetime(card.index, format='ISO8601')
    card = card[['temperature_c']]
    try:
        old, old_reference = v3.represent_card(card)
    except ValueError:
        continue                                  # -300: v3 и v4.1 оба отказываются классифицировать
    parts = fw.represent(card.rename(columns={'temperature_c': 'x'}), card.index[-1])
    shift = float(parts['reference'].iloc[0]) - old_reference
    new = pd.DataFrame({c: parts[k].loc['x'].values for c, k in
                        zip(v3.CHANNELS, ['median', 'min', 'max', 'observed'])}, index=old.index)
    aligned = new.copy()
    aligned[v3.CHANNELS[:3]] = aligned[v3.CHANNELS[:3]] + shift
    error = float((aligned - old).abs().max().max())
    step = card.index.to_series().diff().dt.total_seconds().median()
    if abs(step - 120) < 5:
        f_old = pd.Series(v3.extract_features(old))
        f_new = fw.features(parts).loc['x', f_old.index]
        exact.append(max(abs(shift), error, float((f_old - f_new).abs().max())))
    else:
        shifted.append(error)
print(f'шаг 2 мин: {len(exact)} карточек, макс. расхождение {max(exact):.2e}')
print(f'частый шаг: {len(shifted)} карточек, расхождение после выравнивания опоры {max(shifted):.2e}')
assert max(exact) < 1e-9 and max(shifted) < 1e-9

times = pd.date_range('2026-01-01', periods=300, freq='10s')
generator = torch.Generator().manual_seed(1)
noise = pd.DataFrame({'x': (torch.randn(300, generator=generator, dtype=torch.float64) * .5 + 20).tolist()}, index=times)
estimate = float(fw.noise_sigma(noise).iloc[0])
print(f'шум 0.50 °C -> оценка {estimate:.3f}')
assert .4 < estimate < .6

end = times[250]
before = fw.represent(noise, end)
future = noise.copy()
future.loc[future.index > end, 'x'] += 50.
after = fw.represent(future, end)
assert all(before[k].equals(after[k]) for k in ['median', 'min', 'max', 'observed'])
print('будущие замеры не влияют на окно: OK')
