"""До/после дообучения: модели v3 (28 карточек) и v5 на ОДНИХ И ТЕХ ЖЕ тестовых окнах.

Тестовые окна пересобираются кодом fiberwatch.py с теми же seed, что в прогоне, поэтому
совпадают с synthetic_test_predictions.csv. Для моделей v3 то же окно подаётся в их
собственном представлении (опора - три первых замера, 4 канала, 10 признаков).
Оговорка: модели v3 обучены на карточках ВСЕХ приборов - для них это не leave-one-device-out,
условия даже чуть выгоднее, чем у v4.1.
Запуск: python compare_v3.py <папка с CSV> <папка прогона, например run_v41>
"""
import importlib.util
import random
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
import fiberwatch_v4 as v40   # представление v4.0 = представление v3 (проверено check_equivalence)

spec = importlib.util.spec_from_file_location('v3', V3_SOURCE)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

data_dir, run = Path(sys.argv[1]), Path(sys.argv[2])
v3_dir = PRIVATE / 'real_classification_v3_compact'
annotations = pd.read_csv(PRIVATE / 'review_training_v3' / 'annotations_v3.csv').set_index('sample_id')
cnn3, cnn3_threshold = v3.load_model('cnn', v3_dir / 'models' / 'final_cnn')
rf3, rf3_threshold = v3.load_model('random_forest', v3_dir / 'models' / 'final_random_forest')

matrices = {p.name: fw.read_source(p) for p in sorted(data_dir.glob('[1-6].csv'))}
zones = fw.exclusion_zones(annotations, {k: m for k, (m, q) in matrices.items()})
parts = []
for offset, (name, (matrix, quality)) in enumerate(matrices.items()):
    if hasattr(fw, 'remove_common_mode'):   # v5.1: та же коррекция общего сдвига волокна, что в прогоне
        matrix = fw.remove_common_mode(matrix)[0]
    ends = list(fw.window_ends(matrix))
    test_rng = random.Random(42 + 1000 + offset)
    test_ends = sorted(test_rng.sample(ends, min(55, len(ends))))
    # keep_raw: те же окна с добавленными событиями в виде сырых рядов центральной точки.
    x, x2, table, passport, raws = fw.make_examples(matrix, pd.Series(quality), test_ends, 24, test_rng,
                                                     zones.get(name, []), 'test', keep_raw=True)
    old_x, old_table = [], []
    for raw in raws:
        old = v40.represent(raw.to_frame(), raw.index[-1])
        old_x.append(v40.to_tensor(old))
        old_table.append(v40.features(old).reset_index(drop=True))
    old_x, old_table = torch.cat(old_x), pd.concat(old_table, ignore_index=True)
    assert len(old_x) == len(passport)
    with torch.no_grad():
        passport['cnn_v3'] = (torch.sigmoid(cnn3(old_x)) >= cnn3_threshold).int().tolist()
    passport['random_forest_v3'] = (rf3.predict_proba(old_table[v40.FEATURES])[:, 1] >= rf3_threshold).astype(int)
    passport['policy_3c'] = passport.level_c.ge(3.).astype(int)
    parts.append(passport)
    print(name, len(passport), flush=True)

mine = pd.concat(parts, ignore_index=True)
reference = pd.read_csv(run / 'synthetic_test_predictions.csv')
key = ['file', 'end', 'position', 'kind']
mine['end'] = mine['end'].astype(str)
reference['end'] = reference['end'].astype(str)
merged = reference.merge(mine[key + ['cnn_v3', 'random_forest_v3', 'policy_3c']], on=key, how='inner', validate='one_to_one')
assert len(merged) == len(reference), (len(merged), len(reference))

rows = []
for model in ['cnn_v3', 'random_forest_v3', 'policy_3c', 'amplitude_rule', 'noise_rule', 'random_forest', 'cnn', 'cnn2d']:
    row = {'model': model, **fw.evaluate(merged.label, merged[model])}
    row['false_alarm_glitch'] = merged.loc[merged.kind.eq('glitch'), model].mean()
    for kind in ['step', 'ramp', 'pulse', 'spike']:
        row[f'recall_{kind}'] = merged.loc[merged.kind.eq(kind) & merged.label.eq(1), model].mean()
    row['false_alarm_quiet'] = merged.loc[merged.kind.eq('none'), model].mean()
    row['false_alarm_small'] = merged.loc[merged.kind.eq('small'), model].mean()
    for fold, part in merged.groupby('fold'):
        row[f'fa_quiet_fold{fold}'] = part.loc[part.kind.eq('none'), model].mean()
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(run / 'v3_vs_v4_synthetic.csv', index=False)
print(table.round(3).to_string(index=False))

load = []
for name, (matrix, quality) in matrices.items():
    grid = pd.read_csv(run / f'scan_{name.replace(".csv", "")}.csv.gz')
    grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
    cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * fw.COMMISSIONING_SHARE
    later = grid.loc[grid.end > cutoff].copy()
    later['policy_3c'] = later.level_c.ge(3.)
    events = fw.merge_events(later, 'policy_3c', list(later.coordinate.drop_duplicates().sort_values()))
    hours = (matrix.index[-1] - cutoff).total_seconds() / 3600
    load.append({'file': name, 'flagged_share': later.policy_3c.mean(), 'events_per_hour': len(events) / hours})
pd.DataFrame(load).to_csv(run / 'policy_3c_operator_load.csv', index=False)
print(pd.DataFrame(load).round(4).to_string(index=False))
