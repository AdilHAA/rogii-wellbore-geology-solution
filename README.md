# ROGII — Wellbore Geology Prediction

Решение соревнования [ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) на Kaggle. Команда **N & A & O & A & A misis** заняла **22-е место из 6 125 команд** и получила золотую медаль.

Итоговый ensemble объединил пять подходов: dTVT-cumsum NN с PF/template features, 2D U-Net / CNN-SDF, physical MoE, Typewell Editor и PF/Q/HMM. В репозитории собраны два выбранных финальных pipeline и код компонентов команды.

| Submission | Public RMSE, ft | Private RMSE, ft |
|---|---:|---:|
| CV slot — `final_cv.py` | 5.079 | **6.543** |
| Mix slot — `final_mix.py` | 4.940 | 6.664 |

[Командный writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less) · [Сертификат Kaggle](https://www.kaggle.com/certification/competitions/adilkhabibullin/rogii-wellbore-geology-prediction)

## Задача

По траектории скважины, gamma-ray log (GR) и typewell нужно предсказать TVT — положение ствола относительно геологических пластов. До Prediction Start значения TVT известны, после — скрыты. Метрика — RMSE, ft.

GR patterns повторяются на разных уровнях TVT: несколько траекторий могут хорошо совпадать с одним typewell. Для выбора между ними мы использовали GR matching, геометрию скважины, spatial analogs и наблюдения других скважин с общим typewell.

## Пять компонентов

### Phong Nguyen Quoc — dTVT-cumsum NN + PF/template pool

Full-resolution dilated TCN предсказывает dTVT, затем траектория восстанавливается как `TVT_anchor + cumsum(dTVT)`. Loss сочетает RMSE по восстановленной траектории и SmoothL1 по приращениям. GPU particle filters дают дополнительные NN features: несколько вариантов TVT path и их spread.

Fake-anchor augmentation сдвигает Prediction Start назад, увеличивая число обучающих примеров далеко от anchor. Для скважин с общим master GR log строятся leave-one-well-out таблицы GR bias и sigma по TVT. Они используются в PF likelihood и как NN features. В pool также входят GBM, anchor NN, fake-anchor NN, form-aware NN/GRU и OU-bank NN.

### Adil Khabibullin — chunked 2D U-Net / CNN-SDF

GR скважины и typewell сопоставляются на 2D grid: по горизонтали — расстояние вдоль ствола, по вертикали — возможные уровни TVT. В 9-channel input также входят TVT history, relative TVT, четыре geometry channels и разность GR-сигналов.

2D U-Net с residual blocks предсказывает четыре signed distance fields (SDF) и selector. Forward-backward decoding восстанавливает TVT path с ограничением на изменение уровня между соседними участками. Пять fold-моделей объединяются до decoding.

Длинная скважина обрабатывается через chunked autoregressive rollout с reanchoring по предыдущему прогнозу и triangular overlap blending. Обучение включает synthetic pretraining, fine-tuning на реальных данных, history и anchor augmentations. В CV pipeline `exp151` входит в общий stack, а `exp217` используется в disagreement gate; в mix pipeline используется `exp165`.

### Ochir Dorzhiev — segment-residual physical MoE

Четыре beam-search DP decoders строят траектории в surface space `U = TVT + Z`. Среди них — directional self-reference по GR(TVT) известного prefix и decoder с cusp-shaped typewell response. Spatial k-NN analogs отбираются по heel XY и azimuth; LGBM ranker оценивает analogs.

Per-segment NN gates смешивают candidates. Tanh ограничивает величину поправок относительно проверенной prior simplex, чтобы gate мог уточнять смесь без произвольного изменения траектории. Эта модель входит в оба final pipelines как `sub12`.

### Andrey Serzhenko — Typewell Editor

Residual TCN корректирует GR самого typewell перед decoding. Сначала на известном prefix подбирается robust affine fit, затем модель вносит bounded per-bin edits и поправки gain/offset. Measurement-consistency loss требует, чтобы corrected typewell, sampled по истинному TVT, совпадал с lateral GR.

Для `sub31_avgTW` усредняются corrections пяти fold-моделей, затем берётся `0.5 * original GR + 0.5 * corrected GR`. По этому typewell Alexey static engine заново строит TVT path. В CV pipeline отдельно рассчитывается `az_tweditor_static` по corrected reference.

### Alexey Zakharov — static PF/Q/HMM

Kriging surface prior по train picks задаёт основу для PF с forward и quarter-backward legs. Далее строятся consistency composite Q, HMM decode и смесь нескольких profiles. Этот компонент не требует NN retraining.

GR tail rescaling использует invariant нормализации, связывающий lateral GR и typewell GR на истинном TVT. PF и HMM predictions входят в общий stack и дают features для fusion NN. Их разногласие используется для переключения CV pipeline на отдельный blend.

## CV pipeline

`final_cv.py` строит PF/ANCC features, семь beam configurations, multi-scale NCC, stochastic DTW и formation/spatial priors. GPU PF bank использует 23 profiles, 128 seeds, 500 particles и likelihood scales `(3, 5, 8, 12)`.

Stack содержит **47 prediction columns**. Shrink-NNLS смешивает найденные NNLS coefficients с uniform weights: `0.9 * NNLS + 0.1 / 47`. Далее применяется Ridge residual correction с `alpha=30`. При выборе конфигурации использовался `cv_refit`: весь stacker refit без оцениваемого fold. Разные seeds оставались отдельными columns.

Gamma calibration учитывает ensemble spread: 128 quantile bins, prior `0.85`, pseudo-count `10000`, clip `[-0.25, 1.75]`.

Если среднее по скважине `abs(az_pf - az_hmm)` достигает **8.5 ft**, disagreement gate заменяет прогноз на:

```text
gate_prediction = 0.5559 * az_tweditor_static + 0.4441 * adil_exp217
```

После этого применяются Savitzky–Golay smoothing с `window=17`, `polyorder=3` и duplicate-well hedge. Для совпадающих скважин hedge даёт `0.45 * train_TVT + 0.55 * prediction`. Этот pipeline получил итоговые **6.543** на private leaderboard.

## Mix pipeline

`final_mix.py` последовательно объединяет OU-bank NN `m1`, physical MoE `sub12`, CNN-SDF `exp165`, Alexey `sub22` и `sub31_avgTW`:

```text
base  = 0.20*m1 + 0.525*sub12 + 0.275*exp165 + 0.4
sub22 = 0.80*soft + 0.20*adjhauto
cv58  = 0.85*base + 0.15*sub22
final = 0.30*sub31_avgTW + 0.70*cv58
```

`soft` и `adjhauto` — два варианта Alexey PF/Q prediction. Сдвиг `+0.4` применяется к `base` до следующих blends. В `sub31_avgTW` static PF/HMM engine работает по averaged typewell. Этот pipeline дал **6.664** на private leaderboard.

## Мой вклад

Adil Khabibullin — ML engineer.

- Адаптировал и обучал CNN-SDF / 2D U-Net под данные соревнования. Сравнивал конфигурации модели и параметры forward-backward decoding.
- Подготовил input representation: GR скважины и typewell, TVT history, relative TVT и геометрию скважины на общей 2D grid.
- Работал над chunked autoregressive rollout: reanchoring по предыдущему прогнозу, overlap blending и усреднение SDF пяти fold-моделей внутри каждого шага.
- Развивал генератор synthetic data на основе реальных траекторий, typewell и измеренного шума. Проводил synthetic pretraining и fine-tuning на реальных скважинах.
- Добавлял history и anchor augmentations со смещениями, наклоном и сериями ошибок. Экспериментировал с UNROLL-2, чтобы учитывать накопление ошибки между соседними chunks.
- Проводил OOF validation и error analysis по скважинам, сравнивал эксперименты и готовил predictions своей модели для командного ensemble.

## Файлы

```text
submissions/final_cv.py       # Выбранный CV pipeline
submissions/final_mix.py      # Выбранный mix pipeline
components/adil/              # CNN-SDF, rendering, rollout и hedge
components/phong/             # PF, dTVT-cumsum NN и fusion
components/ochir/             # Physical MoE и analogs
components/alexey/            # Static PF/Q/HMM
components/typewell_editor/   # Typewell Editor
infer.py                      # Отдельный inference CNN-SDF exp217
src/                          # Компактная реализация CNN-SDF exp217
requirements.txt
```
