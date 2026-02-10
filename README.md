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
5. Запустите классификацию с POC. 
```
python run_classification.py files/hh.csv
Результаты находятся в файле classification_results/classification_poc_report.md
```