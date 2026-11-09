# ROGII — Wellbore Geology Prediction

Решение соревнования [ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) на Kaggle. Команда **N & A & O & A & A** заняла **22-е место из 6 125 команд** и получила золотую медаль.

Моя часть решения — модель 2D U-Net / CNN-SDF. Итоговый ensemble команды объединял пять разных подходов; здесь приведён код моей модели.

| Решение | Выборка | RMSE, ft |
|---|---|---:|
| Командный ensemble | Private leaderboard | **6.543** |
| 2D U-Net, `exp_217` | Public leaderboard | **5.985** |

[Командный writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less) · [Сертификат Kaggle](https://www.kaggle.com/certification/competitions/adilkhabibullin/rogii-wellbore-geology-prediction)

## Задача

По траектории скважины, gamma-ray log (GR) и typewell нужно предсказать TVT — положение ствола относительно геологических пластов. До Prediction Start значения TVT известны, после — скрыты. Метрика — RMSE, ft.

## Решение

GR скважины и typewell сопоставляются на 2D grid: по горизонтали — расстояние вдоль ствола, по вертикали — возможные уровни TVT. В 9-channel input также входят TVT history, relative TVT, четыре geometry channels и разность GR-сигналов.

2D U-Net с residual blocks предсказывает signed distance fields (SDF). Forward-backward decoding восстанавливает TVT path с ограничением на изменение уровня между соседними участками.

Для длинных скважин используется chunked autoregressive rollout. Reanchoring происходит на 75% предсказанного участка; overlap составляет 25%. Прогнозы на overlap усредняются с triangular weights. На каждом шаге SDF пяти fold-моделей усредняются до decoding, а общий прогноз задаёт anchor следующего chunk.

Обучение включало synthetic pretraining и fine-tuning на реальных данных. History и anchor augmentations имитировали ошибки предыдущих прогнозов, которые накапливаются во время rollout.

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
infer.py          # Сборка submission.csv
src/data.py       # Чтение данных скважин
src/render.py     # Input representation
src/model.py      # GeoSteerNet / 2D U-Net
src/rollout.py    # Forward-backward decoding и autoregressive rollout
requirements.txt
```
