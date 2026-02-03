# Data Processing Pipeline

Проект для обработки CSV-файлов с резюме с использованием паттерна "Цепочка ответственности"

## Работа

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Запустите скрипт
```bash
python main.py files/hh.csv
```
4. Запустите линейную регрессию
```bash
python linear_model.py --x-data files/x_data.npy --y-data files/y_data.npy 
```
## Результаты линейной регрессии: 
```
TEST SET METRICS:
----------------------------------------
R-squared (RUB): 0.5195
MAE (RUB): 0.0452
RMSE (RUB): 0.0661
MAPE: 136.32%
Within 20% error: 74.08%
Within 30% error: 83.87%
```