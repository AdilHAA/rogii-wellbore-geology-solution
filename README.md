# ROGII — Wellbore Geology Prediction

Решение соревнования [ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) на Kaggle. Команда **N & A & O & A & A** заняла **22-е место из 6 125 команд** и получила золотую медаль.

Моя часть решения — модель 2D U-Net / CNN-SDF. Итоговый ансамбль команды объединял пять разных подходов; здесь приведён код моей модели.

| Решение | Выборка | RMSE, ft |
|---|---|---:|
| Командный ансамбль | Private leaderboard | **6.543** |
| 2D U-Net, `exp_217` | Public leaderboard | **5.985** |

[Командный writeup](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less) · [Сертификат Kaggle](https://www.kaggle.com/certification/competitions/adilkhabibullin/rogii-wellbore-geology-prediction)

## Задача

По траектории скважины, измерениям гамма-каротажа и опорному разрезу нужно восстановить TVT — положение ствола относительно геологических пластов. Для начального участка TVT известен, для оставшейся части его предсказывает модель. Метрика — RMSE в футах.

## Решение

Каротаж скважины и опорного разреза сопоставляются на двумерной сетке: по горизонтали — расстояние вдоль ствола, по вертикали — возможные уровни TVT. На вход также подаются геометрия скважины и история известных или уже предсказанных значений TVT.

Резидуальная 2D U-Net предсказывает поля знакового расстояния (SDF). Декодер forward-backward восстанавливает из них согласованную траекторию, ограничивая резкие скачки между соседними участками.

Длинная скважина обрабатывается окнами. Якорь следующего окна переносится на 75% предсказанного участка. Прогнозы на перекрытии 25% усредняются с треугольными весами. Пять fold-моделей усредняют поля на каждом шаге, после чего общий прогноз задаёт положение следующего окна.

При обучении использовались синтетические сигналы с последующим дообучением на реальных данных. История и положение якоря возмущались, чтобы модель училась продолжать прогноз после собственных ошибок.

## Мой вклад

Adil Khabibullin — адаптация и обучение 2D U-Net, обработка скважин по окнам с переякориванием, декодирование траектории, генерация синтетических примеров и анализ ошибок. Подготовка прогнозов для командного ансамбля.

## Запуск

```bash
python -m pip install -r requirements.txt
python infer.py \
  --data-dir /path/to/competition \
  --weights-dir /path/to/weights \
  --output submission.csv
```

В `--data-dir` должны находиться `sample_submission.csv` и каталог `test/` с файлами `*__horizontal_well.csv` и `*__typewell.csv`.

Веса пяти моделей — `exp_217_ancc_ft_full_fold0.pt` … `exp_217_ancc_ft_full_fold4.pt`. В репозиторий они не включены.

По умолчанию используется CUDA, если она доступна, иначе CPU. Устройство можно указать через `--device cpu` или `--device cuda`.

## Файлы

```text
infer.py          # Сборка submission.csv
src/data.py       # Чтение данных скважин
src/render.py     # Двумерное представление сигналов
src/model.py      # GeoSteerNet / 2D U-Net
src/rollout.py    # Декодирование и обработка по окнам
requirements.txt
```

## Источники

- [medali1992 — ROGII CNN MTP Train](https://www.kaggle.com/code/medali1992/rogii-cnn-mtp-train) — основа модели и подготовки сигналов.
- [hengck23 — CNN+SDF example](https://www.kaggle.com/code/hengck23/cnn-sdf-example) — представление через поле расстояний.
- [Vishal Kishore — SEGN](https://www.kaggle.com/writeups/kishorevishal/segn-an-autoregressive-cost-map-view-of-tvt) — идеи авторегрессионного сопоставления и обработки по окнам.

Адаптированные части GeoSteerNet распространяются под [Apache License 2.0](LICENSES/Apache-2.0.txt).
