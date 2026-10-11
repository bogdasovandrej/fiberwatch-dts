"""Набор реальных случаев для слепой разметки (честная оценка на сплошной записи).

Источник - scan_*.csv.gz прогона: каждый файл просмотрен моделями, которые его прибор не видели.
Группы (в странице разметки скрыты, порядок перемешан):
  alert      - случайные события cnn2d после пусконаладки: доля настоящих = реальная точность;
  disagree   - cnn2d и CNN по времени ответили по-разному;
  rule_only  - правило уведомляет, cnn2d нет (возможные пропуски cnn2d);
  near       - cnn2d чуть ниже своего порога (возможные пропуски);
  gated      - cnn2d «да», но измеренное изменение ниже правила (возможные пропуски из-за правила, v5.2);
  recheck    - повторный показ ранее спорных случаев под новым номером (в точность не входит);
  раунд 4 (режим disagreement) - окна, где модели РАСХОДЯТСЯ после правила 3 °C:
  all_agree (все 4 «да»), rule_only / forest_only (они «да», cnn2d «нет»), cnn2d_only (cnn2d «да», правило «нет»),
  journal (cnn2d «да», правило от фона «нет» - журнал «около порога»), control (все «нет»);
  glitch     - вероятные одноточечные сбои;
  control    - случайные окна без уведомлений (контроль).
Запуск: python build_review.py <папка с CSV> <папка прогона> <папка результата>
         [<показанные ранее случаи.csv> <первый номер> [<seed> [<спорные для повтора.csv>]]]
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).resolve().parent
root = here.parent if here.name == 'tools' else here   # корень проекта: fiberwatch.py, history/
sys.path[:0] = [str(root), str(root / 'history')]
import fiberwatch as fw

data_dir, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
# Второй раунд: свежие случаи, которых разметчик не видел (исключаем показанные ранее).
second_round = len(sys.argv) > 4
shown = pd.read_csv(sys.argv[4]) if second_round else pd.DataFrame(columns=['file', 'coordinate', 'end'])
first_number = int(sys.argv[5]) if second_round else 1
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 11 if second_round else 7)
third_round = len(sys.argv) > 6
disagreement = len(sys.argv) > 8 and sys.argv[8] == 'disagreement'
recheck = pd.read_csv(sys.argv[7]) if len(sys.argv) > 7 else pd.DataFrame()
PLAN = ({'all_agree': 2, 'rule_only': 5, 'forest_only': 5, 'cnn2d_only': 5, 'journal': 2, 'control': 1} if disagreement
        else {'alert': 8, 'near': 1, 'gated': 1} if third_round
        else {'alert': 5, 'near': 1, 'control': 1} if second_round
        else {'alert': 5, 'disagree': 2, 'rule_only': 1, 'near': 2, 'control': 1})


def pick(frame, n):
    rows = list(frame.index)
    rng.shuffle(rows)
    return frame.loc[rows[:n]]


def draw(matrix, coordinate, end, path):
    names = list(matrix.columns)
    centre = int(matrix.columns.get_indexer([coordinate], method='nearest')[0])   # CSV чуть округляет координату
    coordinate = names[centre]
    span = matrix.loc[(matrix.index > end - pd.Timedelta(minutes=60)) & (matrix.index <= end + pd.Timedelta(minutes=20))]
    near = [names[i] for i in range(centre - fw.RADIUS, centre + fw.RADIUS + 1) if 0 <= i < len(names) and i != centre]
    wide = names[max(0, centre - 20):centre + 21]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1.1, 1]})
    minutes = (span.index - end) / pd.Timedelta(minutes=1)
    top.axvspan(-40, 0, color='#c9ced3', alpha=.45, lw=0)
    for column in near:
        top.plot(minutes, span[column].where(span[column] > fw.INVALID_BELOW), color='#9fb4cc', lw=.8)
    top.plot(minutes, span[coordinate].where(span[coordinate] > fw.INVALID_BELOW), color='#18212b', lw=1.8)
    top.set_ylabel('°C')
    top.set_xlabel('минуты относительно конца окна (серая полоса — 40 минут, которые видела модель)')
    top.grid(alpha=.25)
    shown = span[wide].where(span[wide] > fw.INVALID_BELOW)
    image = bottom.pcolormesh(range(len(wide)), minutes, shown.values, cmap='inferno', shading='auto')
    fig.colorbar(image, ax=bottom, label='°C')
    position = wide.index(coordinate)
    bottom.add_patch(plt.Rectangle((position - fw.RADIUS - .5, -40), 2 * fw.RADIUS + 1, 40, fill=False, edgecolor='#4fc3f7', lw=1.6))
    bottom.set_xticks([0, position, len(wide) - 1], [f'{wide[0]:.1f}', f'{coordinate:.1f}', f'{wide[-1]:.1f}'])
    bottom.set_xlabel('координата вдоль волокна (единица не подтверждена)')
    bottom.set_ylabel('минуты')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


cases = []
for path in sorted(data_dir.glob('[1-6].csv')):
    matrix, _ = fw.read_source(path)
    grid = pd.read_csv(run / f'scan_{path.stem}.csv.gz')
    grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
    cutoff = matrix.index[0] + (matrix.index[-1] - matrix.index[0]) * fw.COMMISSIONING_SHARE
    fitted_thresholds = pd.read_csv(run / 'fold_settings.csv')
    fold = fitted_thresholds.loc[fitted_thresholds.held_out.str.contains(path.name, regex=False)]
    threshold = float(fold.loc[fold.model.eq('cnn2d'), 'threshold'].iloc[0])
    late = grid.loc[grid.end > cutoff].copy()
    for column in [m + '_zones' for m in fw.MODELS] + (['cnn2d_border'] if 'cnn2d_border' in late else []):
        late[column] = late[column].astype(str).eq('True')
    columns = list(late.coordinate.drop_duplicates().sort_values())
    events = fw.merge_events(late, 'cnn2d_zones', columns)
    groups = {
        'alert': events.rename(columns={'peak_coordinate': 'coordinate'}).assign(end=None),
        'disagree': late.loc[late.cnn2d_zones != late.cnn_zones],
        'rule_only': late.loc[late.amplitude_rule_zones & ~late.cnn2d_zones],
        'near': late.loc[~late.cnn2d_zones & late.cnn2d_score.between(threshold * .5, threshold)].nlargest(20, 'cnn2d_score'),
        'control': late.loc[~late.cnn2d_zones & ~late.cnn_zones & ~late.amplitude_rule_zones],
    }
    if disagreement:
        cnn2d, rule, forest = late.cnn2d_zones, late.amplitude_rule_zones, late.random_forest_zones
        groups.update({'all_agree': late.loc[cnn2d & rule & forest & late.cnn_zones],
                       'rule_only': late.loc[rule & ~cnn2d], 'forest_only': late.loc[forest & ~cnn2d],
                       'cnn2d_only': late.loc[cnn2d & ~rule], 'journal': late.loc[late.cnn2d_border],
                       'control': late.loc[~cnn2d & ~rule & ~forest & ~late.cnn_zones]})
    if 'limit_c' in late:   # v5.2: модель «да», правило «нет»
        model_yes = late.cnn2d.astype(str).eq('True')
        groups['gated'] = late.loc[model_yes & late.level_c.lt(late.limit_c)]
    seen = shown.loc[shown.file.eq(path.name)]
    seen_times = pd.to_datetime(seen.end, format='ISO8601') if len(seen) else pd.Series(dtype='datetime64[ns]')

    def fresh(frame):
        if not len(seen):
            return frame
        times = frame['end'] if frame['end'].notna().all() else frame['first_alert']
        keep = []
        for coordinate, time in zip(frame['coordinate'], times):
            close = [(abs(coordinate - c) <= 10 * (columns[1] - columns[0])) and abs((pd.Timestamp(time) - t).total_seconds()) <= 7200
                     for c, t in zip(seen.coordinate, seen_times)]
            keep.append(not any(close))
        return frame.loc[keep]

    for group, frame in groups.items():
        if group not in PLAN:
            continue
        for _, row in pick(fresh(frame), PLAN[group]).iterrows():
            end = row['end']
            if group == 'alert':   # момент пика события: окно, где отклонение в пиковой точке наибольшее
                inside = late.loc[late.coordinate.eq(row['coordinate']) & late.end.between(row['first_alert'], row['last_alert'])]
                end = inside.loc[inside.level_c.idxmax(), 'end'] if len(inside) else row['first_alert']
            cases.append({'group': group, 'file': path.name, 'coordinate': float(row['coordinate']), 'end': pd.Timestamp(end)})
    glitches = fw.glitch_points(matrix)
    for _, row in pick(glitches, 0 if second_round else (2 if path.name == '5.csv' else 1)).iterrows():
        cases.append({'group': 'glitch', 'file': path.name, 'coordinate': float(row['coordinate']),
                      'end': pd.Timestamp(row['time']) + pd.Timedelta(minutes=10)})
    print(path.name, len(cases), flush=True)

for row in recheck.itertuples():
    cases.append({'group': 'recheck', 'file': row.file, 'coordinate': float(row.coordinate),
                  'end': pd.Timestamp(row.end), 'source_case': row.case_id})
cases = pd.DataFrame(cases).drop_duplicates(['file', 'coordinate', 'end']).reset_index(drop=True)
order = list(range(len(cases)))
rng.shuffle(order)
cases['order'] = order
cases = cases.sort_values('order').reset_index(drop=True)
cases['case_id'] = [f'case{i + first_number:03d}' for i in range(len(cases))]
cases['order'] = cases['order'] + first_number - 1
matrices = {}
for case in cases.itertuples():
    if case.file not in matrices:
        matrices[case.file] = fw.read_source(data_dir / case.file)[0]
    draw(matrices[case.file], case.coordinate, case.end, out / f'{case.case_id}.png')
cases.to_csv(out / 'cases_private.csv', index=False, encoding='utf-8-sig')
print(cases.group.value_counts().to_string(), '\nвсего', len(cases))
