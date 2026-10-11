"""Как мерить изменение, чтобы оно совпадало с тем, как его видит человек.

Гипотеза разметчика (комментарий к case106): изменение надо отсчитывать от обычного фона
перед событием, а не от первых минут окна - случайный «горб» в начале окна завышает перепад.
Для каждого размеченного случая (раунды 1-3, без повторов) в центральной точке считаем:
  current   - как в v5.2: max(отклонение медианы, выброс min/max) от первых 6 минут окна;
  sustained - фон = медиана первых 20 минут окна, изменение = наибольшее отклонение
              скользящей 6-минутной медианы во второй половине окна;
  peak      - то же, но скользящая медиана по 2 замерам (краткий выброс, не одиночный сбой).
Выбор способа - по раундам 1-2; раунд 3 - проверка.
Запуск: python measure_study.py <папка с CSV> <прогон> <выход.csv>
"""
import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
root = here.parent if here.name == 'tools' else here   # корень проекта: fiberwatch.py, history/
sys.path[:0] = [str(root), str(root / 'history')]
import fiberwatch as fw

data_dir, run, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROUNDS = [(1, 'review/cases/cases_private.csv', 'review/answers_final', 'review/review_corrections.csv'),
          (2, 'review/cases2/cases_private.csv', 'review/answers_round2', None),
          (3, 'review/cases3/cases_private.csv', 'review/answers_round3', None)]
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

matrices, grids, rows = {}, {}, []
for case in cases.itertuples():
    if case.file not in matrices:
        matrices[case.file] = fw.remove_common_mode(fw.read_source(data_dir / case.file)[0])[0]
        grid = pd.read_csv(run / f'scan_{case.file.replace(".csv", "")}.csv.gz')
        grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
        grids[case.file] = grid
    matrix, grid = matrices[case.file], grids[case.file]
    column = matrix.columns[matrix.columns.get_indexer([case.coordinate], method='nearest')[0]]
    end = pd.Timestamp(case.end)
    if matrix.index.tz is not None and end.tz is None:
        end = end.tz_localize(matrix.index.tz)
    history = matrix.loc[(matrix.index > end - 3 * fw.WINDOW) & (matrix.index <= end), column]
    history = history.where(history > fw.INVALID_BELOW).dropna()
    series = history.loc[history.index > end - fw.WINDOW]
    measures = {}
    for label, start, stop in [('half', fw.WINDOW, fw.WINDOW / 2), ('pre20', 1.5 * fw.WINDOW, fw.WINDOW),
                               ('pre40', 2 * fw.WINDOW, fw.WINDOW), ('pre80', 3 * fw.WINDOW, fw.WINDOW)]:
        background = history.loc[(history.index > end - start) & (history.index <= end - stop)].median()
        tail = series if label != 'half' else series.loc[series.index > end - fw.WINDOW / 2]
        measures[f'sustained_{label}'] = (tail.rolling('6min').median() - background).abs().max()
        measures[f'peak_{label}'] = (tail.rolling(2).median() - background).abs().max()
    at_column = grid.loc[grid.coordinate.eq(grid.coordinate.iloc[(grid.coordinate - case.coordinate).abs().argsort().iloc[0]])]
    scan_row = at_column.iloc[(at_column.end - end).abs().argsort().iloc[0]]
    rows.append({'current': max(scan_row.level_c, scan_row.spike_c), 'limit': scan_row.limit_c,
                 **measures, 'score': scan_row.cnn2d_score,
                 'model_yes': str(scan_row.cnn2d_zones) == 'True'})
cases = cases.join(pd.DataFrame(rows))
cases['human_event'] = cases.decision.eq('event')
cases.to_csv(output, index=False, encoding='utf-8-sig')


def agreement(part, measure):
    said = part[measure].ge(part.limit)
    return (said == part.human_event).mean(), int((said & ~part.human_event).sum()), int((~said & part.human_event).sum())


for name, part in [('раунды 1-2 (выбор)', cases.loc[cases['round'].le(2)]), ('раунд 3 (проверка)', cases.loc[cases['round'].eq(3)])]:
    print(name, 'случаев', len(part), 'событий', int(part.human_event.sum()))
    for measure in ['current'] + [c for c in cases.columns if c.startswith(('sustained_', 'peak_'))]:
        share, fp, fn = agreement(part, measure)
        print(f'  {measure:16s} совпадение {share:.3f}  лишние {fp}  пропуски {fn}')
