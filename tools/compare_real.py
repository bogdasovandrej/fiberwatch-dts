"""Сравнение всех моделей двух прогонов на реальных данных: разметка человека + нагрузка.

Для каждого размеченного случая берётся ответ каждой модели в той же точке и в тот же момент
(ближайшая строка сплошного просмотра, решение с порогом зоны - как у оператора).
Выборка случаев не случайная (много спорных), поэтому это сравнение моделей между собой;
несмещённая точность - только по случайным уведомлениям (evaluate_review.py).
Повторно показанные случаи (recheck) не учитываются: это проверка согласия человека с собой.
Запуск: python compare_real.py <выход.csv> <прогон> [<прогон> ...] -- <cases.csv> <ответы> [<поправки.csv>] ...
"""
import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
root = here.parent if here.name == 'tools' else here   # корень проекта: fiberwatch.py, history/
sys.path[:0] = [str(root), str(root / 'history')]
import fiberwatch as fw

args = sys.argv[1:]
output, runs, rounds = Path(args[0]), args[1:args.index('--')], args[args.index('--') + 1:]
labelled = []
for spec in rounds:
    parts = spec.split(',')
    cases = pd.read_csv(parts[0])
    answers = {}
    for path in Path(parts[1]).rglob('*.json'):
        body = json.loads(path.read_text(encoding='utf-8'))
        body = body.get('data', body)
        answers[body.get('case_id', path.stem)] = body.get('decision')
    cases['decision'] = cases.case_id.map(answers)
    if len(parts) > 2:
        for row in pd.read_csv(parts[2]).itertuples():
            cases.loc[cases.case_id.eq(row.case_id), 'decision'] = row.reviewed_decision
    labelled.append(cases)
labelled = pd.concat(labelled, ignore_index=True)
labelled = labelled.loc[labelled.decision.isin(['event', 'normal', 'glitch']) & labelled.group.ne('recheck')].copy()
labelled['human_event'] = labelled.decision.eq('event')

rows = []
for run in runs:
    grids = {}
    answers = []
    for case in labelled.itertuples():
        if case.file not in grids:
            grid = pd.read_csv(Path(run) / f'scan_{case.file.replace(".csv", "")}.csv.gz')
            grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
            grids[case.file] = grid
        grid = grids[case.file]
        end = pd.Timestamp(case.end)
        if grid.end.dt.tz is not None and end.tz is None:
            end = end.tz_localize(grid.end.dt.tz)
        column = grid.coordinate.iloc[(grid.coordinate - case.coordinate).abs().argsort().iloc[0]]
        at_column = grid.loc[grid.coordinate.eq(column)]
        row = at_column.iloc[(at_column.end - end).abs().argsort().iloc[0]]
        answers.append({m: str(row[f'{m}_zones']) == 'True' for m in fw.MODELS})
    answers = pd.DataFrame(answers, index=labelled.index)
    load = pd.read_csv(Path(run) / 'operator_load.csv')
    load = load.loc[load.threshold_mode.eq('zones')]
    for model in fw.MODELS:
        said = answers[model]
        truth = labelled.human_event
        tp, fp = int((said & truth).sum()), int((said & ~truth).sum())
        fn, tn = int((~said & truth).sum()), int((~said & ~truth).sum())
        part = load.loc[load.model.eq(model)].set_index('file').events_per_hour
        rows.append({'run': run, 'model': fw.NAMES[model], 'cases': len(labelled), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                     'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), 'agreement': (tp + tn) / len(labelled),
                     **{f'per_hour_{f.replace(".csv", "")}': v for f, v in part.items()}})
table = pd.DataFrame(rows)
table.to_csv(output, index=False, encoding='utf-8-sig')
print(f'случаев с решением человека: {len(labelled)} (событий {int(labelled.human_event.sum())})')
pd.set_option('display.width', 250)
print(table.round(3).to_string(index=False))
