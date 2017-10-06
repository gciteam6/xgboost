DeepAnalytics Competition #48 (da-compe-48)
==============================

Data analyses worked in GCI2017.

## データセットの生成 (2017/10/06執筆)
+ __入力__: コンペのサイトからダウンロードしたファイル類
    - amd_master.tsv
    - amd1.zip, amd2.zip, amd3.zip, amd4.zip, amd5.zip
    - sfc_master.tsv
    - sfc1.tsv, sfc2.tsv
    - train_kwh.tsv
+ __出力__: 各発電所ごとにデータをまとめたファイル
    - dataset_amd_sfc_kwh.ougishima.tsv
    - dataset_amd_sfc_kwh.ukishima.tsv
    - dataset_amd_sfc_kwh.yonekurayama.tsv
+ 注意
    - 天気予報の情報を含めていない
    - datetimeフォーマットで時刻情報を保持しているのでファイル容量が大きい
+ 手順
    1. Red Hat Enterprise Linux 7.2(RHEL7.2)のマシンを調達する
    1. このプロジェクトをRHEL7.2マシン上にクローンする
    1. クローンしたプロジェクトのルート階層に移動する
    1. [da-compe-48/download](https://deepanalytics.jp/compe/48/download)に掲載されているファイルを `data/raw` に移す(zipファイルは解凍しない)
    1．`provisioning.sh` を走らせる
    1. エラーが帰ってこなければ, `src/data/make_dataset.py` を叩く
    1. エラーが帰ってこなければ, `data/processed/` に上記の.tsvファイルが生成している
+ コマンドとしてやること
```bash
$ git clone https://github.com/gciteam6/xgboost
$ # 生データの輸送に関する何かしらのコマンド(自分はローカルマシンからrsyncしました)
$ cd xgboost
$ sh provisioning.sh
$ python src/data/make_dataset.py data/raw -a 1.2 -s 1.8
```

## 次やること
+ 特徴量の作成
+ XGBoostによるkwh予測モデル作成


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
