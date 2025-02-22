# Iris-Data-Analysis-Python

## 📌 Opis projektu

**Iris-Data-Analysis-Python** to projekt mający na celu analizę i wizualizację popularnego zbioru danych **Iris** z wykorzystaniem języka **Python**. Projekt zawiera skrypty do analizy statystycznej oraz tworzenia wykresów przedstawiających zależności między cechami kwiatów.

## 🛠 Wymagania

Aby uruchomić skrypty, potrzebujesz:

- **Python 3.8** lub nowszy
- **pip** – menedżer pakietów dla Pythona

## 🚀 Instalacja

1. **Klonowanie repozytorium:**

   ```bash
   git clone https://github.com/MatiLUzak/IrisAnalise.git
   cd IrisAnalise
   ```

2. **Utworzenie i aktywacja środowiska wirtualnego (opcjonalnie, ale zalecane):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # lub
   venv\Scripts\activate  # Windows
   ```

3. **Instalacja zależności:**

   Upewnij się, że masz zainstalowane następujące biblioteki:

   - **pandas**
   - **matplotlib**
   - **seaborn**

   Możesz je zainstalować za pomocą polecenia:

   ```bash
   pip install pandas matplotlib seaborn
   ```

## ▶️ Uruchomienie skryptów

1. **Analiza danych Iris:**

   Skrypt `iris_analysis.py` przeprowadza podstawową analizę statystyczną zbioru danych Iris.

   ```bash
   python iris_analysis.py
   ```

2. **Wizualizacja danych Iris:**

   Skrypt `plot_iris_data.py` tworzy wykresy przedstawiające zależności między cechami kwiatów.

   ```bash
   python plot_iris_data.py
   ```

3. **Wizualizacja poszczególnych cech:**

   Skrypt `plot_iris_data_separately.py` generuje wykresy dla każdej cechy osobno.

   ```bash
   python plot_iris_data_separately.py
   ```

## 📂 Struktura projektu

```
IrisAnalise/
├── iris_analysis.py
├── plot_iris_data.py
├── plot_iris_data_separately.py
├── README.md
└── .gitignore
```

- **`iris_analysis.py`** – skrypt do analizy statystycznej zbioru danych Iris.
- **`plot_iris_data.py`** – skrypt do tworzenia wykresów zależności między cechami.
- **`plot_iris_data_separately.py`** – skrypt do wizualizacji poszczególnych cech.

## ✍️ Autor

- **MatiLUzak** – [Profil GitHub](https://github.com/MatiLUzak)

## 📜 Licencja

Ten projekt jest licencjonowany na podstawie licencji MIT. Szczegóły znajdują się w pliku `LICENSE`.
