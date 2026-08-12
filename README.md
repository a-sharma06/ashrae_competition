# ASHRAE Energy Prediction — Neural Network Time Series Forecasting

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](LICENSE)

Neural network approaches to the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) Kaggle competition: predicting hourly building energy meter readings across 1,000+ buildings using historical readings and weather data.

## Approach

The core challenge is that the data contains many concurrent time series (one per building/meter combination, ~12 meter types), so Keras's `TimeseriesGenerator` had to be adapted to generate windows per building/meter group rather than across the whole dataset (168-hour / 7-day lookback windows). Several architectures were compared on this data:

- **Dense NN, one-hot categorical encoding** (`take-2 neural networks_onehot.ipynb`)
- **Dense NN, cyclical (sine) time encoding** instead of one-hot (`take-2 neural networks_onehot_sine.ipynb`)
- **Dense NN with entity embeddings** for categorical variables (`take-2 neural networks_embeddings.ipynb`)
- **CNN-1D** over the lookback window (`cnn-1d and generator.ipynb`)
- Exploratory work on Dask for out-of-memory/distributed feature engineering (`ASHRAE Competition.ipynb`)

## Results

These were short exploratory training runs (1–20 epochs, not fully tuned), so the numbers below should be read as a comparison across architectures rather than a final benchmark:

| Approach | Validation loss (RMSE-based) |
|---|---|
| Dense NN, one-hot encoding | 0.14 – 0.16 |
| Dense NN, sine time encoding | 0.15 |
| Dense NN, entity embeddings | 0.26 – 0.30 (best epoch: 0.2551) |
| CNN-1D | 0.45 (single epoch) |

The one-hot and sine-encoded dense networks outperformed the embeddings and CNN-1D variants in this set of runs; with more training and tuning the ranking could change.

## Tech Stack

Python, TensorFlow/Keras (LSTM, Dense, Conv1D, embeddings), Pandas, NumPy, Dask

## Project Structure

```
ashrae_competition/
├── ASHRAE Competition.ipynb                    # Dask-based feature engineering exploration
├── generator.ipynb                             # TimeseriesGenerator adapted for multi-series data
├── take-2 neural networks_onehot.ipynb         # Dense NN, one-hot encoding
├── take-2 neural networks_onehot_sine.ipynb    # Dense NN, cyclical time encoding
├── take-2 neural networks_embeddings.ipynb     # Dense NN, entity embeddings
├── cnn-1d and generator.ipynb                  # CNN-1D architecture
├── model.json / model.h5                       # Saved model artifacts
└── result.csv                                  # Sample predictions
```

Competition data is not included; see the [Kaggle competition page](https://www.kaggle.com/competitions/ashrae-energy-prediction) to download it.

## Authors

Akshay Sharma, Konica Mulani

## License

All Rights Reserved — see [LICENSE](LICENSE).
