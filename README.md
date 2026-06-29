## Fraud detection model для P2P-переводов

Проект по созданию модели логистической регрессии для обнаружения мошеннических P2P-переводов.
Модель обучается на исторических данных банковских транзакций и журнала P2P-переводов,
строит признаки на основе поведения отправителя и получателя,
подбирает порог по ограничению на recall и сохраняет артефакты. 
Также реализован DAG этого пайплайна через airflow.
Проект был разработан с использованием gitlab сервера ВУЗа.

## Установка и зависимости

Требуется Python 3.10 - 3.13.

```
pip install -r requirements.txt
```

## Данные

Данные (final_p2p_log.csv и final_trans_log.csv) необходимо скачать по ссылке: https://disk.yandex.ru/d/Ci8nP9h4dP3aPg
и вставить в папку data
```
python main.py
```

# Проект ожидает два CSV-файла в папке data/:

- final_p2p_log.csv – журнал P2P-переводов с колонками:
  EventTime, UserID, RecipientID, Amount, Currency, IsFraud
- final_trans_log.csv – общий журнал банковских транзакций:
  UserID, EventTime, MerchantID, MerchantType, Amount, Country, TransactionSuccessful

# Временные границы периодов (warmup, train, val, test) заданы в src/config.py.

## Запуск

```
python main.py
```
или запуск DAGа через веб-клиент (по умолчанию http://localhost:8080/dags/fraud_detection_pipeline/tasks):
```
airflow standalone
```

# Скрипт выполнит:

1. EDA – сохранит графики и отчёт.
2. Генерацию признаков.
3. Обучение LogisticRegression на train, подбор порога на validation.
4. Финальную оценку на test и сохранение всех артефактов.

## Основные признаки

- Временные: час, день недели, день месяца, признак выходного.
- Исторические для отправителя/получателя: число прошлых переводов, накопленная сумма, средний чек, время с последнего перевода, количество уникальных контрагентов.
- Профили из общего журнала транзакций: количество транзакций, средняя сумма, число уникальных мерчантов, доля успешных.
- Отношение суммы текущего перевода к средним прошлым суммам отправителя и получателя.

## Модель

# sklearn.pipeline.Pipeline:

1. Заполнение пропусков (SimpleImputer, константа 0).
2. Стандартизация (StandardScaler).
3. Логистическая регрессия (LogisticRegression с class_weight='balanced').

Порог подбирается по максимальной precision при условии recall >= 0.70 или как "колено" precision-recall кривой.

## Результаты

# После запуска в project/artifacts/ появятся:

- metrics.json – ROC-AUC, Average Precision, Precision/Recall/F1 на пороге для validation и test.
- figures/ – распределение классов, log-amount по классам, fraud rate по часам и дням недели, топ мерчантов, Precision-Recall и ROC кривые.
- fraud_model.joblib – обученный пайплайн для деплоя.