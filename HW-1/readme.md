В данной работе:
<dl>
-провели EDA: удалили повторяющиеся строки, заменили пропуски на медианы, увидели как распределены значения признаков(построили корреляционные матрицы)
  <dl>
-подготовили данные к обучению, в частности выполнили стандартизацию и методом get_dummies обработали категориальные признаки
    <dl>
-обучили модели без и с регуляризациями Ridge и Lasso, сделали переобор гиперпараметров методом GridSearchCV, что впоследствии улушчило скор
      <dl>
-Реализовали сервис на FastAPI: методами post, который получает на вход один объект описанного класса, реализовали так же и для коллекции выбранных классов, которая впоследствии выдает таблицу с прогнозами
 <dl>
 auto_model.sav - модель для запуска на Fast_API, скриншоты predict_item / predict_items прикрепил в энитаск.
