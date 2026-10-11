"""Честная оценка на реальных данных по слепой разметке человека.

Вход: cases_private.csv (что за случай и откуда, build_review.py), папка с ответами
(JSON-документы коллекции answers страницы разметки) и прогон со scan_*.csv.gz.
Выход: real_review_summary.csv и печать главных чисел.

- Реальная точность cnn2d: доля «да — событие» среди СЛУЧАЙНЫХ уведомлений cnn2d (группа alert).
  Это несмещённая оценка: уведомления выбраны случайно, ответ модели разметчику не показан.
- Согласие каждой модели с человеком на всех размеченных случаях (выборка не случайная:
  в ней специально много спорных - поэтому это сравнение моделей между собой, а не их точность).
- «Не уверен» не превращается в правильный ответ и исключается.
Доверительный интервал - Уилсона, 95%.
Запуск: python evaluate_review.py <cases_private.csv> <папка ответов> <папка прогона>
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
root = here.parent if here.name == 'tools' else here   # корень проекта: fiberwatch.py, history/
sys.path[:0] = [str(root), str(root / 'history')]
import fiberwatch as fw

cases_path, answers_dir, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
# Необязательно: поправки после сверки с утверждённой политикой (каждая - с причиной).
corrections = pd.read_csv(sys.argv[4]).set_index('case_id') if len(sys.argv) > 4 else pd.DataFrame()
suffix = '_reviewed' if len(corrections) else ''


def wilson(successes, total, z=1.96):
    if total == 0:
        return float('nan'), float('nan')
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


cases = pd.read_csv(cases_path).set_index('case_id')
answers = {}
for path in answers_dir.rglob('*.json'):
    body = json.loads(path.read_text(encoding='utf-8'))
    body = body.get('data', body)
    answers[body.get('case_id', path.stem)] = body
cases['decision'] = [answers.get(c, {}).get('decision') for c in cases.index]
cases['kind'] = [answers.get(c, {}).get('kind') for c in cases.index]
cases['comment'] = [answers.get(c, {}).get('comment') for c in cases.index]
for case_id, fix in corrections.iterrows():
    assert cases.loc[case_id, 'decision'] == fix.human_decision, case_id
    cases.loc[case_id, 'decision'] = fix.reviewed_decision
labelled = cases.loc[cases.decision.notna()].copy()
print(f'размечено {len(labelled)} из {len(cases)}')
print(labelled.groupby(['group', 'decision']).size().unstack(fill_value=0).to_string())

# Ответы моделей в той же точке и в тот же момент (ближайшая строка просмотра).
rows = []
for case_id, case in labelled.iterrows():
    grid = pd.read_csv(run / f'scan_{case.file.replace(".csv", "")}.csv.gz')
    grid['end'] = pd.to_datetime(grid['end'], format='ISO8601')
    end = pd.Timestamp(case.end)
    if grid.end.dt.tz is not None and end.tz is None:
        end = end.tz_localize(grid.end.dt.tz)
    column = grid.coordinate.iloc[(grid.coordinate - case.coordinate).abs().argsort().iloc[0]]
    at_column = grid.loc[grid.coordinate.eq(column)]
    row = at_column.iloc[(at_column.end - end).abs().argsort().iloc[0]]
    flags = {f'{m}_zones': str(row.get(f'{m}_zones')) == 'True' for m in fw.MODELS}
    rows.append({'case_id': case_id, 'cnn2d_score': row['cnn2d_score'], **flags})
labelled = labelled.join(pd.DataFrame(rows).set_index('case_id'))
decided = labelled.loc[labelled.decision.isin(['event', 'normal', 'glitch'])].copy()
decided['human_event'] = decided.decision.eq('event')
rechecked = decided.loc[decided.group.eq('recheck')]
decided = decided.loc[decided.group.ne('recheck')]   # повтор спорных - проверка согласия человека с собой

summary = []
alerts = decided.loc[decided.group.eq('alert')]
low, high = wilson(int(alerts.human_event.sum()), len(alerts))
summary.append({'measure': 'реальная точность уведомлений cnn2d (случайные уведомления)',
                'value': alerts.human_event.mean(), 'n': len(alerts), 'ci_low': low, 'ci_high': high})
near = decided.loc[decided.group.isin(['near', 'rule_only'])]
summary.append({'measure': 'доля настоящих событий среди окон ниже порога cnn2d (возможные пропуски)',
                'value': near.human_event.mean(), 'n': len(near), 'ci_low': wilson(int(near.human_event.sum()), len(near))[0],
                'ci_high': wilson(int(near.human_event.sum()), len(near))[1]})
gated = decided.loc[decided.group.eq('gated')]
if len(gated):
    summary.append({'measure': 'доля настоящих событий среди отсечённых правилом 3 °C (пропуски из-за правила)',
                    'value': gated.human_event.mean(), 'n': len(gated),
                    'ci_low': wilson(int(gated.human_event.sum()), len(gated))[0],
                    'ci_high': wilson(int(gated.human_event.sum()), len(gated))[1]})
if len(rechecked) and 'source_case' in rechecked:
    print('повторный показ спорных случаев:')
    print(rechecked[['source_case', 'decision']].to_string())
for model in fw.MODELS:
    flags = decided[f'{model}_zones']
    tp = int((flags & decided.human_event).sum())
    fp = int((flags & ~decided.human_event).sum())
    fn = int((~flags & decided.human_event).sum())
    summary.append({'measure': f'согласие с человеком: {fw.NAMES[model]}', 'value': (tp + int((~flags & ~decided.human_event).sum())) / len(decided),
                    'n': len(decided), 'TP': tp, 'FP': fp, 'FN': fn})
summary = pd.DataFrame(summary)
summary.to_csv(run / f'real_review_summary{suffix}.csv', index=False, encoding='utf-8-sig')
labelled.to_csv(run / f'real_review_cases{suffix}.csv', encoding='utf-8-sig')
print(summary.round(3).to_string(index=False))
