"""Классификация температурных фрагментов МГТС: обучение и применение.

Таблицы читает и преобразует Pandas. Один ответ относится ко всей карточке.
Это не детектор начала события и не оценка промышленного качества.
Исходные измерения и результаты обучения сохраняются локально.
"""
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'data'
DATA = ROOT / 'data' / 'raw'
ANNOTATIONS = PACK / 'annotations.csv'
CASES = PACK / 'cases'
CHANNELS = ['median_relative_c', 'min_relative_c', 'max_relative_c', 'observed']
FAMILIES = ['range_rule', 'logistic', 'random_forest', 'extra_trees', 'boosting', 'cnn']
CONFIGS = {
    'range_rule': [{}],
    'logistic': [{'C': .1}, {'C': 1.}, {'C': 10.}],
    'random_forest': [{'max_depth': 3, 'min_samples_leaf': 1}, {'max_depth': None, 'min_samples_leaf': 2}],
    'extra_trees': [{'max_depth': 3, 'min_samples_leaf': 1}, {'max_depth': None, 'min_samples_leaf': 2}],
    'boosting': [{'max_iter': 100, 'max_depth': 2, 'min_samples_leaf': 2}],
    # Компактную архитектуру фиксируем; порог выбираем внутри обучающих приборов.
    # Широкий подбор эпох на малой разметке оказался чувствителен к одной метке.
    'cnn': [{'width': 8, 'epochs': 40}],
}


def represent_card(card):
    """Последние 40 минут завершённой карточки: 20 интервалов по 2 минуты.

    Медиана даёт уровень, минимум/максимум сохраняют короткие отклонения.
    Пустой интервал переносит последние агрегаты, observed=0 показывает пропуск.
    Это не новые измерения и не линейная интерполяция между редкими точками.
    """
    temperatures = card['temperature_c']
    if not isinstance(card.index, pd.DatetimeIndex):
        raise ValueError('Нужен DatetimeIndex')
    if len(card) < 3 or not card.index.is_unique or not card.index.is_monotonic_increasing:
        raise ValueError('Нужны минимум 3 измерения с уникальным возрастающим временем')
    if temperatures.isna().any() or temperatures.isin([float('inf'), -float('inf')]).any():
        raise ValueError('Пропущенные или бесконечные температуры требуют отдельной обработки')
    if temperatures.lt(-273.15).any():
        raise ValueError('Недостоверная температура: требуется проверка, классификатор не вызываем')
    end = card.index[-1]
    start = end - pd.Timedelta(minutes=40)
    temperatures = temperatures.loc[(temperatures.index > start) & (temperatures.index <= end)]
    if len(temperatures) < 3:
        raise ValueError('Недостаточно истории за 40 минут')
    if temperatures.index.to_series().diff().gt(pd.Timedelta(minutes=10)).any():
        raise ValueError('Разрыв больше 10 минут: карточку надо разделить')
    interval_ends = pd.date_range(start=start+pd.Timedelta(minutes=2), end=end, freq='2min')
    aggregate = temperatures.resample('2min', origin=start, closed='right', label='right').agg(['median','min','max','count'])
    aggregate = aggregate.reindex(interval_ends)
    observed = aggregate['count'].fillna(0).gt(0)
    if observed.sum() < 3:
        raise ValueError('Меньше 3 заполненных двухминутных интервалов')
    # Это опорный уровень фрагмента, а НЕ заведомо нормальная температура.
    reference = float(temperatures.iloc[:3].median())
    relative = aggregate[['median','min','max']].sub(reference)
    relative = relative.ffill().fillna(0.)
    relative.columns = CHANNELS[:3]
    relative['observed'] = observed.astype(float)
    relative.index.name = 'interval_end'
    assert relative.shape == (20,4) and not relative.isna().any().any()
    return relative, reference


def extract_features(representation):
    """Лес получает численные характеристики того же представления, что CNN."""
    measured = representation.loc[representation.observed.eq(1)]
    median = measured['median_relative_c']
    difference = median.diff()
    minutes_between = median.index.to_series().diff().dt.total_seconds().div(60)
    rate = difference.div(minutes_between)
    spread = measured['max_relative_c'] - measured['min_relative_c']
    return {
        'range_c': float(measured.max_relative_c.max()-measured.min_relative_c.min()),
        'median_range_c': float(median.max()-median.min()),
        'std_c': float(median.std(ddof=0)),
        'net_change_c': float(median.iloc[-1]-median.iloc[0]),
        'absolute_net_change_c': float(abs(median.iloc[-1]-median.iloc[0])),
        'max_absolute_rate_c_per_min': float(rate.abs().max()),
        'mean_absolute_rate_c_per_min': float(rate.abs().mean()),
        'mean_bin_spread_c': float(spread.mean()),
        'max_bin_spread_c': float(spread.max()),
        'observed_fraction': float(representation.observed.mean()),
    }


def load_dataset(annotations_path=ANNOTATIONS, cases_dir=CASES, data_dir=DATA):
    annotations = pd.read_csv(annotations_path, float_precision='round_trip').set_index('sample_id')
    if not annotations.index.is_unique:
        raise ValueError('Каждый sample_id должен встречаться в разметке один раз')
    # Неясно и недостоверные данные НЕ превращаем в норму.
    metadata = annotations.loc[annotations.label.isin(['событие','нет события'])].copy()
    metadata['target'] = metadata.label.map({'нет события':0, 'событие':1}).astype(int)
    device_by_file = {}
    for filename in metadata.source_file.unique():
        source_meta = pd.read_csv(data_dir/filename, usecols=['dts_id','channel_id'])
        if source_meta.dts_id.isna().any() or source_meta.dts_id.nunique() != 1:
            raise ValueError(f'Неоднозначный прибор в {filename}')
        device_by_file[filename] = str(source_meta.dts_id.iloc[0])
    # 2.csv, 3.csv, 4.csv одного прибора остаются вместе, включая разные каналы.
    metadata['group'] = metadata.source_file.map(device_by_file)
    features, sequences, representations = {}, [], []
    for sample_id in metadata.index:
        card = pd.read_csv(cases_dir/f'{sample_id}.csv', index_col=0, float_precision='round_trip')
        card.index = pd.to_datetime(card.index)
        representation, reference = represent_card(card)
        features[sample_id] = extract_features(representation)
        metadata.loc[sample_id,'reference_c'] = reference
        metadata.loc[sample_id,'raw_samples'] = len(card)
        scaled = representation.copy()
        # Постоянный перевод масштаба: единица входа = 10 °C, маска остаётся 0/1.
        scaled[CHANNELS[:3]] = scaled[CHANNELS[:3]] / 10.
        sequences.append(scaled[CHANNELS].T.to_numpy(dtype='float32').tolist())
        representations.append(representation.assign(sample_id=sample_id).reset_index())
    features = pd.DataFrame.from_dict(features, orient='index').loc[metadata.index]
    features.index.name = 'sample_id'
    inputs = torch.tensor(sequences, dtype=torch.float32)
    metadata['tensor_position'] = range(len(metadata))
    assert inputs.shape == (len(metadata),4,20)
    assert metadata.groupby('group').target.nunique().eq(2).all()
    return metadata, features, inputs, pd.concat(representations,ignore_index=True)


class TemperatureCNN(nn.Module):
    """Две свёртки изучают форму; максимумы и средние собирают итог окна."""
    def __init__(self, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv1d(4,width,3,padding=1), nn.ReLU(),
                                     nn.Conv1d(width,width,3,padding=1), nn.ReLU())
        self.output = nn.Linear(width*2,1)

    def forward(self, inputs):
        patterns = self.filters(inputs)
        average_pattern = patterns.mean(dim=2)
        strongest_pattern = patterns.amax(dim=2)
        summary = torch.cat([average_pattern,strongest_pattern],dim=1)
        return self.output(summary).squeeze(1)


def fit_model(family, config, ids, metadata, features, inputs):
    labels = metadata.loc[ids,'target']
    if labels.nunique() != 2:
        raise ValueError('В обучении нужны оба класса')
    history = []
    if family == 'range_rule':
        return None, history
    if family == 'cnn':
        torch.manual_seed(42)
        model = TemperatureCNN(config['width'])
        positions = metadata.loc[ids,'tensor_position'].to_list()
        x = inputs[positions]
        y = torch.tensor(labels.to_list(),dtype=torch.float32)
        # Вес положительного класса рассчитывается ТОЛЬКО по обучающим ответам.
        positive_weight = torch.tensor(float(labels.eq(0).sum()/labels.eq(1).sum()))
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        optimizer = torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001)
        model.train()
        for epoch in range(1,config['epochs']+1):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_function(logits,y)
            if not torch.isfinite(loss):
                raise ValueError('Ошибка обучения стала неконечной')
            loss.backward()
            optimizer.step()
            history.append({'epoch':epoch,'training_loss':float(loss.detach())})
        model.eval()
        return model, history
    if family == 'logistic':
        model = make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],class_weight='balanced',max_iter=2000,random_state=42))
    elif family in ['random_forest','extra_trees']:
        cls = RandomForestClassifier if family == 'random_forest' else ExtraTreesClassifier
        model = cls(n_estimators=150,class_weight='balanced',random_state=42,n_jobs=1,**config)
    elif family == 'boosting':
        model = HistGradientBoostingClassifier(learning_rate=.05,early_stopping=False,class_weight='balanced',random_state=42,**config)
    else:
        raise ValueError(f'Неизвестная модель {family}')
    model.fit(features.loc[ids],labels)
    return model, history


def predict_scores(family, model, ids, metadata, features, inputs):
    if family == 'range_rule':
        return features.loc[ids,'range_c'].rename('score')
    if family == 'cnn':
        positions = metadata.loc[ids,'tensor_position'].to_list()
        model.eval()
        with torch.no_grad():
            values = torch.sigmoid(model(inputs[positions])).tolist()
    else:
        values = model.predict_proba(features.loc[ids])[:,1].tolist()
    return pd.Series(values,index=ids,name='score')


def evaluate(labels, predictions):
    tn,fp,fn,tp = confusion_matrix(labels,predictions,labels=[0,1]).ravel()
    return {'n':len(labels),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
            'precision':precision_score(labels,predictions,zero_division=0),
            'recall':recall_score(labels,predictions,zero_division=0),
            'f1':f1_score(labels,predictions,zero_division=0),
            'f2':fbeta_score(labels,predictions,beta=2,zero_division=0)}


def select_settings(family, available_ids, metadata, features, inputs, outer_name, audit):
    """Настройки выбираются внутри обучения; внешняя проверочная группа недоступна."""
    groups = metadata.loc[available_ids,'group'].unique()
    trials = []
    for config_id,config in enumerate(CONFIGS[family]):
        score_parts = []
        for validation_group in groups:
            is_validation = metadata.loc[available_ids,'group'].eq(validation_group)
            training_ids = available_ids[~is_validation]
            validation_ids = available_ids[is_validation]
            assert set(metadata.loc[training_ids,'group']).isdisjoint(metadata.loc[validation_ids,'group'])
            model,_ = fit_model(family,config,training_ids,metadata,features,inputs)
            scores = predict_scores(family,model,validation_ids,metadata,features,inputs)
            score_parts.append(scores)
            audit.append({'outer':outer_name,'family':family,'config_id':config_id,
                          'training_ids':';'.join(training_ids),'validation_ids':';'.join(validation_ids),
                          'validation_group':validation_group})
        scores = pd.concat(score_parts).loc[available_ids]
        thresholds = [1.,1.5,2.,2.5,3.,4.,5.,6.,8.,10.] if family == 'range_rule' else [i/10 for i in range(1,10)]
        for threshold in thresholds:
            metrics = evaluate(metadata.loc[available_ids,'target'],scores.ge(threshold).astype(int))
            trials.append({'outer':outer_name,'family':family,'config_id':config_id,'threshold':threshold,**metrics})
    table = pd.DataFrame(trials)
    # F2 даёт полноте больший вес. При равенстве предпочитаем меньше пропусков,
    # меньше ложных тревог и более простой (ранний в списке) вариант модели.
    table = table.sort_values(['f2','recall','precision','config_id','threshold'],ascending=[False,False,False,True,False],kind='stable')
    best = table.iloc[0]
    return CONFIGS[family][int(best.config_id)],float(best.threshold),table


def save_model(family, model, config, threshold, path):
    if family == 'cnn':
        torch.save({'state_dict':model.state_dict(),'width':config['width'],
                    'epochs':config['epochs'],'threshold':threshold},path.with_suffix('.pt'))
    else:
        joblib.dump({'model':model,'config':config,'threshold':threshold},path.with_suffix('.joblib'))


def load_model(family, path):
    if family == 'cnn':
        saved = torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        model = TemperatureCNN(saved['width'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        return model,float(saved['threshold'])
    saved = joblib.load(path.with_suffix('.joblib'))
    return saved['model'],float(saved['threshold'])


def predict_card(card, model_dir, family):
    """Оценка завершённого фрагмента. Не выдаёт границы или причину события."""
    result = {'model':family,'status':'требуется проверка',
              'scope':'весь предоставленный фрагмент; не момент начала события',
              'cause':'неизвестна','normal_for_equipment':'пока неизвестно',
              'event_detected':None,'score':None,'score_is_calibrated_probability':False}
    if len(card):
        result['observation_start'] = str(card.index.min())
        result['observation_end'] = str(card.index.max())
    try:
        representation,reference = represent_card(card)
    except ValueError as error:
        result['data_quality'] = str(error)
        return result
    model,threshold = load_model(family,model_dir/'models'/f'final_{family}')
    features = pd.DataFrame([extract_features(representation)],index=['new_fragment'])
    metadata = pd.DataFrame({'tensor_position':[0]},index=['new_fragment'])
    scaled = representation.copy()
    scaled[CHANNELS[:3]] = scaled[CHANNELS[:3]] / 10.
    inputs = torch.tensor([scaled[CHANNELS].T.to_numpy(dtype='float32').tolist()],dtype=torch.float32)
    score = float(predict_scores(family,model,metadata.index,metadata,features,inputs).iloc[0])
    event_detected = score >= threshold
    result.update({'event_detected':event_detected,'score':score,'threshold':threshold,
                   'status':'требуется проверка' if event_detected else 'событие не обнаружено',
                   'data_quality':'проверены время, пропуски и нижняя физическая граница; не полная проверка достоверности',
                   'reference_c':reference,'range_c':float(features.range_c.iloc[0]),
                   'observed_bin_fraction':float(representation.observed.mean())})
    return result


def read_raw_fragment(path, requested_coordinate, end_text, whole_history=False):
    """Адаптер исходного CSV МГТС: выбираем 40 минут и одну координату."""
    source = pd.read_csv(path)
    if len(source[['dts_id','channel_id']].drop_duplicates()) != 1:
        raise ValueError('В файле смешаны приборы/каналы')
    times = pd.DatetimeIndex(pd.to_datetime(source['timestamp'],format='ISO8601'))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Некорректное время в исходном файле')
    end = pd.Timestamp(end_text) if end_text else times[-1]
    if (end.tz is None) != (times.tz is None):
        raise ValueError('Часовой пояс --end должен соответствовать наличию пояса в CSV')
    if whole_history:
        selected = source.loc[times <= end].copy()
    else:
        selected = source.loc[(times > end-pd.Timedelta(minutes=40)) & (times <= end)].copy()
    if selected.empty:
        raise ValueError('В указанном интервале нет измерений')
    first_points = pd.DataFrame(json.loads(selected.iloc[0]['points']),columns=['coordinate','temperature_c'])
    coordinate = min(first_points.coordinate,key=lambda value:abs(value-requested_coordinate))
    values = []
    for points_text in selected['points']:
        points = pd.DataFrame(json.loads(points_text),columns=['coordinate','temperature_c'])
        if not points.coordinate.is_unique:
            raise ValueError('Повтор координаты в профиле')
        profile = points.set_index('coordinate')['temperature_c']
        values.append(profile.get(coordinate,float('nan')))
    card = pd.DataFrame({'temperature_c':values},index=pd.DatetimeIndex(pd.to_datetime(selected.timestamp,format='ISO8601')))
    return card,float(coordinate)


def render_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(output/'metrics.csv').set_index('family')
    dataset = pd.read_csv(output/'dataset.csv')
    positive_count = int(dataset.target.eq(1).sum())
    negative_count = int(dataset.target.eq(0).sum())
    names = {'range_rule':'Порог по размаху','logistic':'Логистическая регрессия',
             'random_forest':'Случайный лес','extra_trees':'Extra Trees',
             'boosting':'Градиентный бустинг','cnn':'Свёрточная сеть (CNN)'}
    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    for axis,column,title,color in [(axes[0],'FN',f'Пропущено событий из {positive_count}','#bd621b'),
                                    (axes[1],'FP',f'Ложных срабатываний на {negative_count} фрагментах без события','#2366a1')]:
        values = metrics.loc[FAMILIES,column]
        axis.barh(range(len(FAMILIES)),values,color=color)
        axis.set_yticks(range(len(FAMILIES)),[names[f] for f in FAMILIES])
        axis.invert_yaxis()
        maximum = max(4, int(metrics.loc[FAMILIES,['FN','FP']].max().max()) + 1)
        axis.set_xlim(0,maximum)
        axis.set_xticks(range(maximum+1))
        axis.set_title(title,fontsize=11)
        axis.grid(axis='x',alpha=.2)
        axis.spines[['top','right']].set_visible(False)
        for i,value in enumerate(values): axis.text(value+.07,i,str(int(value)),va='center')
    fig.suptitle('Классификация реальных температурных фрагментов',fontsize=15,y=.98)
    fig.text(.03,.025,f'{len(dataset)} выбранных вручную фрагмента, {dataset.group.nunique()} прибора. Настройка без проверяемого прибора.\n'
             'Предварительное исследование; не оценка частоты тревог в эксплуатации.',fontsize=10)
    fig.subplots_adjust(left=.20,right=.97,top=.86,bottom=.20,wspace=.95)
    fig.savefig(output/'comparison.png',dpi=145,bbox_inches='tight')
    plt.close(fig)


def run(output, annotations_path=ANNOTATIONS, cases_dir=CASES, data_dir=DATA):
    started = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Выберите новую папку результата: прошлый эксперимент не перезаписываем')
    output.mkdir(parents=True,exist_ok=True)
    (output/'models').mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata,features,inputs,representation = load_dataset(annotations_path,cases_dir,data_dir)
    metadata.to_csv(output/'dataset.csv',encoding='utf-8-sig')
    features.to_csv(output/'features.csv')
    representation.to_csv(output/'representation.csv',index=False)
    # Сохраняем точные версии кода и ответов, на которых обучены эти модели.
    (output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'annotations_snapshot.csv').write_bytes(annotations_path.read_bytes())
    manifest = []
    for sample_id in metadata.index:
        card_path = (cases_dir/f'{sample_id}.csv').resolve()
        manifest.append({'sample_id':sample_id,'path':str(card_path),
                         'sha256':hashlib.sha256(card_path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(output/'input_manifest.csv',index=False)
    protocol = {'task':'event anywhere in a completed reviewed fragment',
                'evaluation':'exploratory nested leave-one-device-out; NOT blind final test',
                'grouping':'dts_id; files 2,3,4 kept together across channels',
                'window':'up to 40 min; 20 right-closed 2-min bins ending at last observation',
                'channels':CHANNELS,'scaling':'temperature differences /10 C; observed mask unchanged',
                'labels':'human reviewed; uncertain and physically invalid excluded from binary fit',
                'selection':'inner grouped F2; then recall, precision, simpler config, higher threshold',
                'configs':CONFIGS,'seed':42,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
                'annotations_path':str(annotations_path.resolve()),
                'cases_dir':str(cases_dir.resolve()),'data_dir':str(data_dir.resolve()),
                'cards':len(metadata),'positive_cards':int(metadata.target.sum()),
                'negative_cards':int(metadata.target.eq(0).sum()),'devices':metadata.group.nunique(),
                'annotation_sha256':hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'limitations':[f'{len(metadata)} selected fragments, {metadata.group.nunique()} devices; not representative operational prevalence',
                               'labels and study design were developed after viewing the source data',
                               'related physical events across devices cannot be ruled out without installation map',
                               'whole-fragment labels do not establish onset, offset, cause, or emergency',
                               'model scores are not calibrated confidence probabilities']}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    predictions,settings,trials,audit,losses = [],[],[],[],[]
    groups = sorted(metadata.group.unique())
    for fold, held_group in enumerate(groups,1):
        outer_name = f'fold_{fold}'
        test_ids = metadata.index[metadata.group.eq(held_group)]
        train_ids = metadata.index[~metadata.group.eq(held_group)]
        assert set(metadata.loc[train_ids,'group']).isdisjoint(metadata.loc[test_ids,'group'])
        print(f'{outer_name}: training {len(train_ids)}, held-out {len(test_ids)}, files {metadata.loc[test_ids,"source_file"].unique().tolist()}',flush=True)
        for family in FAMILIES:
            config,threshold,trial_table = select_settings(family,train_ids,metadata,features,inputs,outer_name,audit)
            trials.append(trial_table)
            model,history = fit_model(family,config,train_ids,metadata,features,inputs)
            scores = predict_scores(family,model,test_ids,metadata,features,inputs)
            settings.append({'outer':outer_name,'family':family,'test_group':held_group,
                             'train_ids':';'.join(train_ids),'test_ids':';'.join(test_ids),
                             'config':json.dumps(config),'threshold':threshold})
            model_path = output/'models'/f'{outer_name}_{family}'
            save_model(family,model,config,threshold,model_path)
            restored,restored_threshold = load_model(family,model_path)
            repeated = predict_scores(family,restored,test_ids,metadata,features,inputs)
            pd.testing.assert_series_equal(scores,repeated,check_exact=True)
            assert threshold == restored_threshold
            for sample_id,score in scores.items():
                predictions.append({'sample_id':sample_id,'family':family,'outer':outer_name,
                                    'target':int(metadata.loc[sample_id,'target']),
                                    'score':float(score),'threshold':threshold,'prediction':int(score>=threshold)})
            if family == 'cnn':
                losses.extend([dict(row,outer=outer_name) for row in history])
                print(f'  CNN: {config}; threshold {threshold}',flush=True)
    results = pd.DataFrame(predictions)
    for constant,name in [(0,'always_no_event'),(1,'always_event')]:
        extra = pd.DataFrame({'sample_id':metadata.index,'family':name,'outer':'reference',
                              'target':metadata.target.to_list(),'score':constant,'threshold':.5,'prediction':constant})
        results = pd.concat([results,extra],ignore_index=True)
    metrics = []
    for family,rows in results.groupby('family',sort=False):
        assert rows.sample_id.is_unique and len(rows)==len(metadata)
        metrics.append({'family':family,**evaluate(rows.target,rows.prediction)})
    metric_table = pd.DataFrame(metrics).sort_values(['f2','recall'],ascending=False,kind='stable')
    results.to_csv(output/'held_out_predictions.csv',index=False)
    metric_table.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(settings).to_csv(output/'outer_settings.csv',index=False)
    pd.concat(trials).to_csv(output/'inner_trials.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'split_audit.csv',index=False)
    pd.DataFrame(losses).to_csv(output/'cnn_training_history.csv',index=False)

    # Модели для применения учатся на всех разрешённых карточках ПОСЛЕ оценки.
    # Их ответы на этих же карточках не объявляем проверочными метриками.
    final_settings,final_audit = [],[]
    for family in FAMILIES:
        config,threshold,table = select_settings(family,metadata.index,metadata,features,inputs,'full_fit',final_audit)
        table.to_csv(output/f'full_selection_{family}.csv',index=False)
        model,history = fit_model(family,config,metadata.index,metadata,features,inputs)
        save_model(family,model,config,threshold,output/'models'/f'final_{family}')
        restored,_ = load_model(family,output/'models'/f'final_{family}')
        scores = predict_scores(family,model,metadata.index,metadata,features,inputs)
        replay = predict_scores(family,restored,metadata.index,metadata,features,inputs)
        pd.testing.assert_series_equal(scores,replay,check_exact=True)
        final_settings.append({'family':family,'config':json.dumps(config),'threshold':threshold,
                               'parameters':sum(p.numel() for p in model.parameters()) if family=='cnn' else None})
        if family == 'cnn':
            pd.DataFrame(history).to_csv(output/'final_cnn_training_history.csv',index=False)
            # Независимый повтор обучения выбранной CNN, а не только загрузка весов.
            rerun,_ = fit_model(family,config,metadata.index,metadata,features,inputs)
            pd.testing.assert_series_equal(scores,predict_scores(family,rerun,metadata.index,metadata,features,inputs),check_exact=True)
    pd.DataFrame(final_settings).to_csv(output/'final_settings.csv',index=False)
    pd.DataFrame(final_audit).to_csv(output/'final_split_audit.csv',index=False)
    mistakes = results.loc[results.target.ne(results.prediction)].copy()
    mistakes = mistakes.merge(metadata[['source_file','label','group']],left_on='sample_id',right_index=True,validate='many_to_one')
    mistakes.to_csv(output/'mistakes.csv',index=False,encoding='utf-8-sig')
    render_results(output)
    completion = {'status':'training_complete','elapsed_seconds':time.perf_counter()-started,
                  'exact_reload':'passed for all outer and final models',
                  'final_cnn_retraining':'exact predictions reproduced',
                  'independent_verification':'run verify_fiberwatch.py separately'}
    (output/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding='utf-8')
    print(metric_table.to_string(index=False))
    print('Saved all models. Exact reload passed for every fit; final CNN retraining reproducible.')
    print('Exploratory fragment classification only. No accident causes or event boundaries inferred.')




TRACE_COLUMNS = ['timestamp', 'temperature_c', 'score', 'threshold', 'window_event',
                 'phase', 'confirmed_phase', 'slope_c_per_min', 'trend_span_minutes',
                 'trend_window_minutes', 'trend_points', 'episode_id', 'armed',
                 'history_start', 'history_end']
UPDATE_COLUMNS = ['time', 'episode_id', 'action', 'description', 'status']
EPISODE_COLUMNS = ['episode_id', 'notification_start', 'notification_end',
                   'last_observed', 'initial_phase', 'last_phase', 'closure_reason', 'status']


def recent_trend(history, minutes=5., threshold=.2):
    """Наклон по реальным точкам; окно расширяется для редкой дискретизации.

    Для трёх измерений с шагом четыре минуты необходимо хотя бы восемь минут.
    Интерполяции нет. Расширение ограничено 20 минутами; разрывы >10 минут
    обрабатываются отдельно. Длительность и число использованных точек выдаём.
    """
    intervals = history.index.to_series().diff().dt.total_seconds().div(60).dropna().tail(10)
    cadence = intervals.median() if len(intervals) else None
    effective_minutes = minutes
    if pd.notna(cadence):
        effective_minutes = min(20., max(minutes, 2.1 * float(cadence)))
    end = history.index[-1]
    recent = history.loc[history.index >= end - pd.Timedelta(minutes=effective_minutes)]
    span = (recent.index[-1] - recent.index[0]).total_seconds() / 60
    result = {'slope_c_per_min': None, 'phase': 'недостаточно истории',
              'trend_span_minutes': span, 'trend_window_minutes': effective_minutes,
              'trend_points': len(recent)}
    if len(recent) < 3 or span < 3.:
        return result
    elapsed = pd.Series((recent.index - recent.index[0]).total_seconds() / 60,
                        index=recent.index)
    centered_time = elapsed - elapsed.mean()
    centered_temperature = recent.temperature_c - recent.temperature_c.mean()
    slope = float((centered_time * centered_temperature).sum() / centered_time.pow(2).sum())
    phase = 'без выраженного тренда'
    if slope > threshold:
        phase = 'рост'
    elif slope < -threshold:
        phase = 'снижение'
    result.update(slope_c_per_min=slope, phase=phase)
    return result


class NotificationState:
    """Состояние одного потока одной координаты, без просмотра будущего."""
    def __init__(self):
        self.active = None
        self.counter = 0
        self.armed = True
        self.negative_since = None
        self.confirmed_phase = None
        self.confirmed_since = None
        self.pending_phase = None
        self.pending_since = None
        self.invalid_run = False
        self.updates = []
        self.episodes = []

    def emit(self, stamp, action, description, episode_id=None):
        self.updates.append({'time': stamp, 'episode_id': episode_id, 'action': action,
                             'description': description, 'status': 'требуется проверка'})

    def close(self, stamp, reason):
        if self.active is not None:
            self.active['notification_end'] = stamp
            self.active['closure_reason'] = reason
            self.emit(stamp, 'закрытие группы', reason, self.active['episode_id'])
            self.episodes.append(self.active.copy())
            self.active = None

    def reset(self, stamp, reason):
        self.close(stamp, reason)
        self.armed = True
        self.negative_since = None
        self.confirmed_phase = None
        self.confirmed_since = None
        self.pending_phase = None
        self.pending_since = None

    def invalid(self, stamp):
        self.reset(stamp, 'недостоверное измерение; физическое окончание неизвестно')
        if not self.invalid_run:
            self.emit(stamp, 'качество данных', 'недостоверное измерение; началась серия')
        self.invalid_run = True

    def confirm_phase(self, stamp, phase):
        old_phase = self.confirmed_phase
        if self.confirmed_phase is None or phase == 'недостаточно истории':
            self.confirmed_phase = phase
            self.confirmed_since = stamp
            self.pending_phase = None
            self.pending_since = None
        elif phase == self.confirmed_phase:
            self.pending_phase = None
            self.pending_since = None
        elif phase != self.pending_phase:
            self.pending_phase = phase
            self.pending_since = stamp
        elif stamp - self.pending_since >= pd.Timedelta(minutes=3):
            self.confirmed_phase = phase
            self.confirmed_since = stamp
            self.pending_phase = None
            self.pending_since = None
        return old_phase != self.confirmed_phase

    def step(self, stamp, found, phase):
        if self.invalid_run:
            self.emit(stamp, 'восстановление данных',
                      'снова поступило физически допустимое значение; точность не подтверждена')
            self.invalid_run = False
        previous_phase = self.confirmed_phase
        changed = self.confirm_phase(stamp, phase)
        # После закрытия из-за устойчивости новое направление может открыть
        # новую группу, даже если CNN ещё видит прежнее событие в своём окне.
        fresh_direction = (changed and self.confirmed_phase in ['рост', 'снижение']
                           and previous_phase not in ['рост', 'снижение'])
        if self.active is None and fresh_direction:
            self.armed = True

        if found is False:
            if self.negative_since is None:
                self.negative_since = stamp
            if stamp - self.negative_since >= pd.Timedelta(minutes=5):
                self.close(stamp, 'нет срабатываний 5 минут; физическое окончание неизвестно')
                self.armed = True
        else:
            # Недостаточно данных не считается отрицательным ответом модели.
            self.negative_since = None

        opened_now = False
        if found is True and self.active is None and self.armed:
            self.counter += 1
            self.active = {'episode_id': self.counter, 'notification_start': stamp,
                           'notification_end': pd.NaT, 'last_observed': stamp,
                           'initial_phase': self.confirmed_phase, 'last_phase': self.confirmed_phase,
                           'closure_reason': 'ещё не закрыта', 'status': 'требуется проверка'}
            self.emit(stamp, 'первое уведомление',
                      'модель обнаружила изменение в прошедшем окне; причина неизвестна', self.counter)
            self.armed = False
            opened_now = True

        if self.active is not None:
            self.active['last_observed'] = stamp
            if changed and not opened_now:
                self.active['last_phase'] = self.confirmed_phase
                self.emit(stamp, 'обновление тенденции', self.confirmed_phase, self.active['episode_id'])
            # Не закрываем и не подавляем первый сигнал на этом же измерении.
            # Устойчивость подтверждается независимо от положительности CNN.
            sufficiently_old = stamp - self.active['notification_start'] >= pd.Timedelta(minutes=3)
            stable_confirmed = (self.confirmed_phase == 'без выраженного тренда'
                                and phase == 'без выраженного тренда')
            if not opened_now and sufficiently_old and stable_confirmed:
                self.close(stamp, 'нет выраженного тренда; нормально ли это для оборудования — пока неизвестно')
        return self.active['episode_id'] if self.active is not None else None


def replay(card, model, threshold, model_module=None, score_function=None, family='random_forest'):
    """Читает строки по порядку; score_function нужен только для проверок механики."""
    if not isinstance(card.index, pd.DatetimeIndex) or card.index.hasnans:
        raise ValueError('Нужно корректное время')
    if not card.index.is_unique or not card.index.is_monotonic_increasing or card.empty:
        raise ValueError('Нужны измерения в строго возрастающем порядке')
    if model_module is None and score_function is None:
        model_module = sys.modules[__name__]
    rows = []
    state = NotificationState()
    history = pd.DataFrame(columns=['temperature_c'], index=pd.DatetimeIndex([], tz=card.index.tz))
    previous_time = None
    if model is not None and hasattr(model, 'eval'):
        model.eval()
    for stamp, source_row in card.iterrows():
        if previous_time is not None and stamp - previous_time > pd.Timedelta(minutes=10):
            state.reset(stamp, 'разрыв данных; физическое окончание неизвестно')
            history = history.iloc[:0]
        previous_time = stamp
        temperature = float(source_row.temperature_c)
        valid = pd.notna(temperature) and abs(temperature) != float('inf') and temperature >= -273.15
        if not valid:
            state.invalid(stamp)
            history = history.iloc[:0]
            rows.append({'timestamp': stamp, 'temperature_c': temperature, 'threshold': threshold,
                         'phase': 'недостоверное измерение', 'armed': state.armed})
            continue
        history.loc[stamp, 'temperature_c'] = temperature
        history = history.loc[history.index > stamp - pd.Timedelta(minutes=40)].astype(float)
        assert history.index.max() == stamp
        trend = recent_trend(history)
        if score_function is not None:
            score = score_function(history)
        else:
            try:
                representation, _ = model_module.represent_card(history)
            except ValueError:
                score = None
            else:
                if family == 'cnn':
                    scaled = representation.copy()
                    channels = model_module.CHANNELS
                    scaled[channels[:3]] = scaled[channels[:3]] / 10.
                    tensor = torch.tensor([scaled[channels].T.to_numpy(dtype='float32').tolist()], dtype=torch.float32)
                    with torch.no_grad():
                        score = float(torch.sigmoid(model(tensor)).item())
                else:
                    features = pd.DataFrame([model_module.extract_features(representation)])
                    score = float(model.predict_proba(features)[0, 1])
        found = None if score is None else bool(score >= threshold)
        episode_id = state.step(stamp, found, trend['phase'])
        rows.append({'timestamp': stamp, 'temperature_c': temperature, 'score': score, 'threshold': threshold,
                     'window_event': found, **trend, 'confirmed_phase': state.confirmed_phase,
                     'episode_id': episode_id, 'armed': state.armed,
                     'history_start': history.index.min(), 'history_end': history.index.max()})
    trace = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    trace['episode_id'] = trace['episode_id'].astype('Int64')
    trace['window_event'] = trace['window_event'].astype('boolean')
    updates = pd.DataFrame(state.updates, columns=UPDATE_COLUMNS)
    updates['time'] = pd.Series(pd.DatetimeIndex(updates['time'], tz=card.index.tz))
    updates['episode_id'] = updates['episode_id'].astype('Int64')
    # Открытая группа остаётся открытой. Конец файла — не конец события.
    episodes = state.episodes.copy()
    if state.active is not None:
        episodes.append(state.active.copy())
    episodes = pd.DataFrame(episodes, columns=EPISODE_COLUMNS)
    return trace, updates, episodes




def temperature_evidence(card):
    """Причинные кандидаты по величине изменения, отдельно от ответа ML.

    3 °C — выбранная политика просмотра, не норматив аварии. Опорный уровень
    фиксируем при отклонении на 1 °C, чтобы рост не сдвигал свою же опору.
    Опора живёт не более 40 минут. Многочасовые изменения только сохраняем.
    Все длительности отсчитываются по timestamp, исходники не изменяются.
    """
    if (card.empty or not isinstance(card.index, pd.DatetimeIndex)
            or card.index.hasnans or not card.index.is_unique
            or not card.index.is_monotonic_increasing):
        raise ValueError('Нужны непустые измерения с уникальным возрастающим временем')
    history = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=card.index.tz))
    anchor = None
    anchor_time = None
    active = None
    previous = None
    counter = 0
    slow_direction = 0
    trace = []
    events = []

    def close_event(stamp, reason):
        nonlocal active
        if active is not None:
            active['tracking_end'] = stamp
            active['closure_reason'] = reason
            events.append(active)
            active = None

    for stamp, source in card.iterrows():
        value = float(source.temperature_c)
        valid = pd.notna(value) and abs(value) != float('inf') and value >= -273.15
        gap = previous is not None and stamp - previous > pd.Timedelta(minutes=10)
        previous = stamp
        if gap or not valid:
            close_event(stamp, 'разрыв или недостоверные данные; окончание события неизвестно')
            history = history.iloc[:0]
            anchor = anchor_time = None
            slow_direction = 0
        row = {'timestamp': stamp, 'temperature_c': value, 'valid_for_analysis': valid,
               'reference_c': None, 'delta_c': None, 'reached_3c': False,
               'threshold_event_id': None, 'new_threshold_event': False,
               'slow_reference_c': None, 'slow_delta_c': None,
               'slow_candidate': False, 'new_slow_candidate': False}
        if not valid:
            trace.append(row)
            continue
        history = history.loc[history.index >= stamp - pd.Timedelta(hours=3)]
        if anchor_time is not None and stamp - anchor_time >= pd.Timedelta(minutes=40):
            close_event(stamp, 'закончился срок опоры 40 минут; окончание события неизвестно')
            anchor = anchor_time = None
        past = history.loc[history.index >= stamp - pd.Timedelta(minutes=20)]
        enough_history = len(past) >= 3 and stamp - past.index[0] >= pd.Timedelta(minutes=6)
        reference = anchor
        if reference is None and enough_history:
            reference = float(past.median())
            if abs(value - reference) >= 1.:
                anchor = reference
                anchor_time = stamp
        if reference is not None:
            delta = value - reference
            reached = abs(delta) >= 3.
            row.update(reference_c=reference, delta_c=delta, reached_3c=reached)
            if active is not None:
                active['observations_after_alert'] += 1
                if abs(delta) > abs(active['peak_delta_c']):
                    active['peak_delta_c'] = delta
                # Перескок через опору тоже завершает прежнее направление.
                direction_reversed = delta * active['initial_delta_c'] < 0
                returned = abs(delta) <= .5
                if returned or direction_reversed:
                    event_id = active['event_id']
                    if returned and active['observations_after_alert'] == 1:
                        active['description'] = 'краткий выброс'
                    close_event(stamp, 'возврат к опоре' if returned else 'смена направления относительно опоры')
                    row['threshold_event_id'] = event_id
            if reached and active is None:
                counter += 1
                active = {'event_id': counter, 'alert_time': stamp, 'tracking_end': pd.NaT,
                          'reference_c': reference, 'initial_delta_c': delta, 'peak_delta_c': delta,
                          'description': 'нагрев' if delta > 0 else 'охлаждение',
                          'status': 'требуется проверка', 'method': 'amplitude_policy_3c',
                          'observations_after_alert': 0, 'closure_reason': 'ещё отслеживается'}
                row['new_threshold_event'] = True
            if active is not None:
                row['threshold_event_id'] = active['event_id']
            if anchor is not None and abs(delta) <= .5:
                anchor = anchor_time = None
        history.loc[stamp] = value
        # Сравниваем устойчивые уровни начала и конца доступной истории.
        # Минимум час наблюдений; одиночный выброс не меняет медиану 10 минут.
        if stamp - history.index[0] >= pd.Timedelta(hours=1):
            first = history.loc[history.index <= history.index[0] + pd.Timedelta(minutes=20)]
            recent = history.loc[history.index >= stamp - pd.Timedelta(minutes=10)]
            enough_recent = len(recent) >= 3 and stamp - recent.index[0] >= pd.Timedelta(minutes=6)
            if len(first) >= 3 and enough_recent:
                slow_reference = float(first.median())
                slow_delta = float(recent.median() - slow_reference)
                direction = (1 if slow_delta > 0 else -1) if abs(slow_delta) >= 3. else 0
                row.update(slow_reference_c=slow_reference, slow_delta_c=slow_delta,
                           slow_candidate=direction != 0,
                           new_slow_candidate=direction != 0 and direction != slow_direction)
                slow_direction = direction
        trace.append(row)
    if active is not None:
        events.append(active)
    result = pd.DataFrame(trace)
    for column in ['reference_c', 'delta_c', 'slow_reference_c', 'slow_delta_c']:
        result[column] = pd.to_numeric(result[column]).astype(float)
    result['threshold_event_id'] = result.threshold_event_id.astype('Int64')
    columns = ['event_id', 'alert_time', 'tracking_end', 'reference_c', 'initial_delta_c',
               'peak_delta_c', 'description', 'status', 'method', 'observations_after_alert', 'closure_reason']
    return result, pd.DataFrame(events, columns=columns)


def save_stream(card, model_dir, family, output, source_metadata, include_temperature_evidence=False):
    if family not in ['random_forest', 'cnn']:
        raise ValueError('Потоковый режим проверен для random_forest и cnn')
    if output.exists():
        raise ValueError('Папка результата уже существует; выберите новую')
    model, threshold = load_model(family, model_dir / 'models' / f'final_{family}')
    trace, notifications, episodes = replay(card, model, threshold, family=family)
    output.mkdir(parents=True)
    for table in [trace, notifications, episodes]:
        for key, value in source_metadata.items():
            table[key] = value
    trace.to_csv(output / 'measurements.csv', index=False, encoding='utf-8-sig')
    notifications.to_csv(output / 'notifications.csv', index=False, encoding='utf-8-sig')
    episodes.to_csv(output / 'notification_groups.csv', index=False, encoding='utf-8-sig')
    if include_temperature_evidence:
        evidence, amplitude_events = temperature_evidence(card)
        evidence['model_score'] = trace.score.to_list()
        evidence['model_window_event'] = trace.window_event.to_list()
        for table in [evidence, amplitude_events]:
            for key, value in source_metadata.items():
                table[key] = value
        evidence.to_csv(output / 'temperature_evidence.csv', index=False, encoding='utf-8-sig')
        amplitude_events.to_csv(output / 'amplitude_events.csv', index=False, encoding='utf-8-sig')
    protocol = {'mode':'experimental causal replay of CSV, one coordinate',
                'family':family, 'threshold':threshold, 'rows':len(trace),
                'model_fit':'final model trained on all reviewed cards; this run is not a quality test',
                'limitations':['No physical live feed; no accident or cause inference.',
                               'Repeated short events may fail to open a new notification group.',
                               'Group closure is not proof that the physical event has ended.',
                               'Model score is not calibrated probability; coordinates have unconfirmed units.'],
                **source_metadata}
    if include_temperature_evidence:
        protocol['temperature_evidence'] = {
            'mode': 'separate experimental rule evidence; NOT additional ML predictions',
            'amplitude_c': 3., 'freeze_at_c': 1., 'return_band_c': .5,
            'past_reference_minutes': 20, 'max_anchor_minutes': 40,
            'slow_history_minutes': 180, 'slow_minimum_history_minutes': 60,
            'slow_action': 'candidate only; may overlap a rapid event; cause and normality unknown',
            'limits': 'Heuristic references, no equipment limits; sample gaps >10min reset all history.',
        }
    (output / 'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output.resolve()),'measurements':len(trace),
                      'notifications':len(notifications),'notification_groups':len(episodes),
                      'mode':'экспериментальное воспроизведение CSV'},ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description='FiberWatch: классификация температурных фрагментов')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true', help='Сравнить и обучить модели по разметке')
    action.add_argument('--predict-card', type=Path, help='CSV с временем в первом поле и temperature_c')
    action.add_argument('--source', type=Path, help='Исходный CSV МГТС с points; нужна --coordinate')
    parser.add_argument('--output', type=Path, default=ROOT / 'training_run')
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--cases-dir', type=Path, default=CASES)
    parser.add_argument('--data-dir', type=Path, default=DATA, help='Каталог исходных CSV для проверки групп приборов при обучении')
    parser.add_argument('--coordinate', type=float)
    parser.add_argument('--end', help='До какого времени брать данные; пояс должен соответствовать источнику')
    parser.add_argument('--model-dir', type=Path, default=ROOT)
    parser.add_argument('--family', choices=FAMILIES, default='random_forest')
    parser.add_argument('--result', type=Path, help='Сохранить ответ о фрагменте в JSON')
    parser.add_argument('--replay-output', type=Path, help='Вместо оценки фрагмента воспроизвести поток; экспериментальный режим')
    parser.add_argument('--temperature-evidence', action='store_true',
                        help='При воспроизведении отдельно сохранить изменения >=3 °C и медленные кандидаты')
    args = parser.parse_args()
    if args.temperature_evidence and not args.replay_output:
        parser.error('--temperature-evidence требует --replay-output')
    if args.train:
        if args.replay_output or args.result or args.coordinate is not None or args.end:
            parser.error('--train не совмещается с параметрами применения')
        run(args.output, args.annotations, args.cases_dir, args.data_dir)
        return
    if args.replay_output and args.result:
        parser.error('--result и --replay-output задают разные форматы результата')
    torch.set_num_threads(1)
    source_metadata = {}
    if args.source:
        if args.coordinate is None:
            parser.error('--source требует --coordinate')
        card, coordinate = read_raw_fragment(args.source, args.coordinate, args.end,
                                             whole_history=args.replay_output is not None)
        source_metadata = {'source_file':str(args.source), 'coordinate_requested':args.coordinate,
                           'coordinate_selected':coordinate, 'coordinate_unit':'не подтверждена'}
    else:
        if args.coordinate is not None or args.end:
            parser.error('--coordinate и --end применяются только к --source')
        card = pd.read_csv(args.predict_card, index_col=0, float_precision='round_trip')
        card.index = pd.to_datetime(card.index, format='ISO8601')
        source_metadata = {'input_card':str(args.predict_card)}
    if args.replay_output:
        save_stream(card, args.model_dir, args.family, args.replay_output, source_metadata,
                    include_temperature_evidence=args.temperature_evidence)
    else:
        result = predict_card(card, args.model_dir, args.family)
        result.update(source_metadata)
        result_text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.result:
            if args.result.exists():
                raise ValueError('Файл ответа уже существует; выберите новый')
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(result_text, encoding='utf-8')
        print(result_text)


if __name__ == '__main__':
    main()
