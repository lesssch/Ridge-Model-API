# Ridge Model API
В данной работе были выполнены:
* Подготовка тренировочного и тестового датасетов:
  * Стандартизация
  * One-hot кодирование
* Обучена модель на подобранных гиперпараметрах
* Разработан микросервис FastAPI для получения предсказаний для одного объекта, либо списка объектов в файле csv

При тестировании моделей лучше всего показала себя модель Ridge, которая используется в микросервисе. Больший буст в качестве дала стандартизация вещественных признаков.

Не вышло "по-нормальному" обработать ситуации, когда в файле с объектами нет таких значений в столбце, чтобы при One-Hot кодировании, появился признак, на котором была обучена модель. Либо наоборот, в обученной модели не будет признаков, которые есьт в датасете.

**Скриншоты работы микросервиса:**
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/34c4388b-06b4-4578-91d8-03c7f61cf256)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/63a5bc66-d2d1-4dd2-9000-16b1b33df4a0)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/2ac0948c-170b-4a0c-88fb-595eb3b48cd3)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/a4f9e5bf-5769-4a68-8892-51f0a9130015)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/a31d7936-9d59-4c69-8cd1-370a6192d9af)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/5b154215-ec6f-4dda-8819-2b9ddbdf5c10)
![image](https://github.com/lesssch/Ridge-Model-API/assets/80597622/3b2b591c-b623-48dd-85af-01eb8b19c0c3)
