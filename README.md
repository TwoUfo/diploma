# 📘 Назва проєкту

*Мультизадачна модель машинного навчання для класифікації та прогнозування фізичних характеристик астероїдів за даними NASA JPL*

---

## 👤 Автор

- **ПІБ**: Журавський Віталій Олегович
- **Група**: ФЕП-41
- **Керівник**: Сінькевич Олег, доктор філосовських наук, доцент
- **Дата виконання**: 31.05.2026

---

## 📌 Загальна інформація

- **Тип проєкту**: Дослідницький / ML (дипломна робота)
- **Мова програмування**: Python 3.11+
- **Фреймворки / Бібліотеки**: PyTorch, scikit-learn, pandas, NumPy, Streamlit, LightGBM, Optuna, Plotly
- **Джерело даних**: [NASA JPL Small-Body Database (SBDB) Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)

---

## 🧠 Опис функціоналу

Одна модель (`AsteroidMTLModel`) зі спільним кодувальником (backbone) і чотирма «головами» розв'язує **чотири задачі одночасно** (multitask learning):

- **Класифікація** таксономічного класу астероїда (10 класів: MBA, TNO, NEO-групи тощо)
- **Регресія діаметра** (у `log1p`-просторі)
- **Регресія геометричного альбедо** (залишок над класовим пріором)
- **Період обертання** через Mixture Density Network (бімодальний розподіл)

Абсолютна зоряна величина **H — це вхідна ознака** (не ціль): каталог дає її майже для кожного об'єкта, а діаметр/альбедо моделюються за відомим H.

Демо-застосунок (Streamlit) має **три режими вводу**:
- **Preset** — реальний каталогізований астероїд (Ceres, Vesta, Eros, Apophis, Pluto…) з його справжнім вектором ознак;
- **Custom (k-NN)** — користувач задає лише орбіту (`e, a, i, ma, H`), решта ознак добирається методом k найближчих сусідів;
- **All features (manual)** — ручний ввід усіх 24 ознак (з чекбоксами «n/a» для невідомих).

---

## 🧱 Опис основних файлів

| Файл / Модуль | Призначення |
|---|---|
| `src/data/fetch_jpl_full.py` | Стягує повний каталог із JPL SBDB → `data/raw/dataset.csv` (~465 МБ) |
| `src/data/preprocessing.py` | `build_splits`: попередня обробка + поділ на навчальну/перевірну/тестову вибірки + scaler/encoder/пріор |
| `src/data/dataset.py` | `AsteroidDataset` — клас даних PyTorch (відділяє ознаки від цілей і масок) |
| `src/data/build_presets.py` | `compute_presets` + генерація `presets.joblib` (запасне джерело пресетів для демо) |
| `src/models/mtl_model.py` | `AsteroidMTLModel` (спільний backbone + 4 голови, MDN, залишкова голова альбедо) |
| `src/models/losses.py` | `MultiTaskLoss` — зважування задач за навченою невизначеністю (Kendall, навчані σ) |
| `src/training/trainer.py` | `Trainer` — цикл навчання, рання зупинка, збереження контрольних точок |
| `src/training/metrics.py` | масковані метрики класифікації та регресії |
| `app/streamlit_app.py` | демонстраційний вебзастосунок |
| `notebooks/01..05` | розвідувальний аналіз → попередня обробка → базові моделі → навчання → оцінка |
| `configs/default.yaml` | конфіг (шляхи артефактів, гіперпараметри) |

---

## ▶️ Швидкий старт (демо за 3 кроки)

Усе необхідне для запуску демо **вже є в репозиторії** — навчена модель (`models/mtl_model.pt`), `scaler`/`label_encoder`/`class_albedo_prior`, дані пресетів (`presets.joblib`) та навчальна вибірка (`data/processed/train.parquet`). **Сирий датасет завантажувати не потрібно.**

### 1. Встановлення інструментів

- **Python 3.11+**
- `git`, `pip`

### 2. Клонування та залежності

```bash
git clone https://github.com/TwoUfo/diploma.git
cd diploma

python -m venv .venv
source .venv/bin/activate          # Windows: .venv/Scripts/activate
pip install -r requirements.txt
```

### 3. Запустити демо-застосунок

```bash
streamlit run app/streamlit_app.py
```

Відкрийте у браузері адресу, яку виведе Streamlit (зазвичай `http://localhost:8501`).

> Усі шляхи в застосунку обчислюються відносно кореня проєкту (через `configs/default.yaml`), тож запускати можна з будь-якої теки. Усі три режими (готовий астероїд, k-NN, ручний) працюють одразу — без сирого `dataset.csv`.

---

## 🔄 (Необов'язково) Відтворити дані та перенавчити модель з нуля

Ці кроки **не потрібні** для запуску демо — лише якщо хочете оновити каталог або перенавчити модель.

### A. Стягнути свіжий сирий датасет із JPL

```bash
python -m src.data.fetch_jpl_full
```

Завантажує повний каталог малих тіл (13 орбітальних класів) у `data/raw/dataset.csv` (~465 МБ; цей файл у git **не** зберігається). Потребує інтернету; для класу MBA — до ~30 хв.

### B. Згенерувати оброблені вибірки

```bash
python -m src.data.preprocessing
```

Перезаписує `data/processed/{train,val,test}.parquet`, `scaler.joblib`, `label_encoder.joblib`, `class_albedo_prior.joblib`. Процес детермінований (`random_state=42`): на тому самому `dataset.csv` дає біт-у-біт ідентичні файли.

> **Перебудувати запасні пресети** після зміни сирих даних:
> ```bash
> python -m src.data.build_presets
> ```

### C. Перенавчити модель

```bash
jupyter notebook notebooks/
# 01_eda → 02_preprocessing → 03_baselines → 04_mtl_training → 05_evaluation
```

Ноутбук `04_mtl_training.ipynb` створює `models/mtl_model.pt`.

---

## 🐳 Запуск через Docker

Docker-образ **вшиває** оброблені дані та навчену модель усередину, тож контейнер працює одразу — нічого попередньо генерувати не треба (усі потрібні артефакти вже в репозиторії й копіюються в образ).

```bash
docker build -t asteroid-mtl .
docker run -p 8501:8501 asteroid-mtl
```

Потім відкрийте `http://localhost:8501`.

> `Dockerfile` копіює `data/processed/` та `models/` із репозиторію. Сам `dataset.csv` (465 МБ) в образ **не** потрапляє — демо в контейнері використовує `presets.joblib` як джерело пресетів.

---

## 🖱️ Інструкція для користувача (демо)

1. У бічній панелі оберіть **режим вводу**:
   - **Preset / Custom (k-NN)** — або реальний астероїд зі списку, або власна орбіта (`e, a, i, ma, H`), решта добирається k-NN.
   - **All features (manual)** — ручний ввід усіх 24 ознак; для невідомих позначте **n/a** (підставиться медіана навчального набору).
2. Натисніть **Predict**.
3. Результат:
   - передбачений **клас** + впевненість,
   - **альбедо**,
   - **діаметр** двома способами (фізична формула `D = 1329 / √p_v · 10^(−H/5)` та значення «голови» моделі),
   - **період обертання** (для k-NN — глобальна медіана + IQR),
   - графік ймовірностей класів, зведення входу, для пресета — справжні каталожні значення.

---

## 🏗️ Архітектура

```
24 ознаки → SharedBackbone [512→256→128→64] → shared_repr (64) ┬→ class_head    → 10 логітів
                                                               ├→ diameter_head → log1p(D)
                                                               ├→ albedo_head   → residual + класовий пріор
                                                               └→ rot_head(MDN) → суміш 5 гаусіан
```

- **Backbone**: стек `Linear → BatchNorm → ReLU → Dropout`.
- **Функція втрат**: 4 завдання зважуються через **навчані невизначеності** (Kendall et al.); масковані MSE для діаметра/альбедо, masked MDN-NLL для обертання, зважена крос-ентропія для класу.
- **Параметрів**: 196 411.

---

## 🧾 Використані джерела / технології

- NASA JPL Small-Body Database Query API — [ssd-api.jpl.nasa.gov](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)
- PyTorch, scikit-learn, pandas, NumPy
- Streamlit (демо), Plotly (візуалізація)
- A. Kendall, Y. Gal, R. Cipolla — *Multi-Task Learning Using Uncertainty to Weigh Losses* (2018)
- C. M. Bishop — *Mixture Density Networks* (1994)
