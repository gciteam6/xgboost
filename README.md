DeepAnalytics Competition #48 (da-compe-48)
==============================

Data analyses worked in GCI2017.

データセットの生成
------------
+ コマンドとしてやること
```bash
$ git clone https://github.com/gciteam6/xgboost && cd xgboost
$ # data/raw/に下記の入力ファイルを輸送する何かしらのコマンド(rsyncとか)
$ sh provisioning.sh
$ python src/data/make_dataset.py data/raw -a 1.2 -s 1.8
```
+ __入力__: コンペのサイトからダウンロードしたファイル
    - zipファイル類
        * amd1.zip, amd2.zip, amd3.zip, amd4.zip, amd5.zip
        * sfc1.zip, sfc2.zip
        * forecast.zip
    - 観測地点所在ファイル類
        * amd_master.tsv
        * sfc_master.tsv
    - 発電量ファイル
        * train_kwh.tsv
+ __出力__: データ規模で分割された各発電所ごとのデータ結合物
    - data/processed/に生成される
    - defaultではtrain0 ~ train7, test0 ~ test2の11分割でデータが保存される
    - 保存データの例(pandas.DataFrameの分解物, ↓3つで1セット)
        * dataset.amd_sfc_forecast_kwh.train0.columns.ougishima.blp
        * dataset.amd_sfc_forecast_kwh.train0.index.ougishima.blp
        * dataset.amd_sfc_forecast_kwh.train0.values.ougishima.blp
+ 中間生成物
    - zipファイル解凍物
        * amd1/, amd2/, amd3/, amd4/, amd5/, sfc1/, sfc2/, forecast/
    - 結合用中間ファイル類(例↓)
        * amd_data.values.ougishima.data#2, amd_flags.values.ougishima.data#2
        * sfc_data.values.ougishima.data#3, sfc_flags.values.ougishima.data#3
        * forecast_data.values.ougishima.data#4
        * sola_data.values.ougishima.data#5
+ 注意
    - 天気予報の情報を含めていない → 含めるようにした(2017/10/08~)
    - 時刻領域を定めた予報情報を1つのカラムにまとめた
        * e.g. `pc_00-06`, `pc_06-12`, `pc_12-18`, `pc_18-24` → `pc`
    - datetimeフォーマットで時刻情報を保持しているのでファイル容量が大きい → bloscpackを使って保存するようにした(2017/10/16~)
    - location(ougishima, ukishima, yonekurayama)ごとに処理を行う
        * そのため冗長なところがある(forecast_data取得の部分とか)
    - 出力・中間ファイルが人間に優しくない
        * tsv形式などで保存しない
        * bloscpackとpandas.DataFrameで内容を理解できる
+ コマンドがやっていること
    - 各発電所の位置が対角線の交点にくるように, 格子状のメッシュを作成
    - 格子内に含まれるアメダス観測点, 地上気象観測点のデータのみ収集する
    - `-a`, `-m`: 格子のグリッドの大きさ(単位は10進法の緯経度で与える)
+ 編集ログ
    - 2017/10/06執筆
    - 2017/10/08修正: 主に天気予報データ包含に関する箇所訂正
    - 2017/10/16修正: 主にbloscpackに関わる箇所訂正


特徴量の作成
------------
+ コマンドとしてやること
```bash
$ # 当プロジェクトのroot階層に移動するコマンド(cdとか)
$ python src/features/build_features.py
```
+ __入力__: src/data/make_dataset.pyの出力
+ __出力__: トレーニング/テストデータセット
    - data/processed/に生成する
    - 保存データの例(pandas.DataFrameの分解物, ↓3つで1セット)
        * data/processed/dataset.train_X_y.columns.ougishima.blp
        * data/processed/dataset.train_X_y.index.ougishima.blp
        * data/processed/dataset.train_X_y.values.ougishima.blp
+ 中間生成物
    - data/interim/に生成する
    - 欠損処理, 特徴量生成過程の中間ファイル(例↓)
        * data/interim/dataset.data.train0.values.ougishima.features#1
        * data/interim/dataset.data.values.ougishima.features#2
+ 注意
    - メモリが死にやすい
        * bloscpackの操作でよく死ぬ
        * pandasですべての処理を行うと多分死なない
+ コマンドがやっていること
    - ダミー変数を加える(features#1)
    - 時刻をずらしたのち欠損値を処理する(features#2)
        * 時刻をずらすのは, amedasとsurfaceのデータは前日のものしか利用できないことに起因する
    - 10分毎のデータを30分毎のデータにリサンプリングする
+ 編集ログ
    - 2017/10/16執筆


XGBoostによるkwh予測モデル作成
------------
+ コマンドとしてやること
```bash
$ python src/models/separate_validation_index.py -n 5  # cross-validationでトレーニングデータを何分割するか指定する
$ python src/models/train_model.py -v -f 0  # cross-validationのデータ0-fold番目を予測するためのモデルを作成する(=全てのトレーニングサンプルを使わない)
$ python src/models/predict_model.py -v -f 0  # cross-validationのデータ0-fold番目を予測する
$ python src/models/train_model.py -t  # トレーニングサンプルすべてを使ってモデルを作成する

# `-l`で発電所を指定できる. その場合, `ukishima`・`ougishima`・`yonekurayama`のいずれかを続ける
$ python src/models/train_model.py -t -l ougishima
$ python src/models/predict_model.py -t -l ougishima
```
+ __入力__: src/features/build_features.pyの出力
+ __出力__:
    - pickleでserializeされたXGBoostモデル(例↓)
        * models/xgb/fit_model.n_estimators_1000.max_depth_3.learning_rate_0.05.test.ougishima.pkl
        * models/xgb/fit_model.n_estimators_1000.max_depth_3.learning_rate_0.05.crossval0.ougishima.pkl
    - 予測値を保存したTSVファイル(例↓)
        * models/xgb/predict.n_estimators_1000.max_depth_3.learning_rate_0.05.test.ougishima.tsv
        * models/xgb/predict.n_estimators_1000.max_depth_3.learning_rate_0.05.crossval0.ougishima.tsv
+ 中間生成物
    - Cross-validation時のfolding indexをまとめたファイル(例↓)
        * data/processed/dataset.train_X_y.crossval0.ougishima.blp
        * data/processed/dataset.train_X_y.crossval1.ougishima.blp
+ 注意
    - トレーニングデータの最初が欠損している
        * データセットの時刻をずらす操作を行っているため
    - コマンドラインで愚直にすべて回すのは面倒
        * shell scriptの記述例も載せている(src/models/itertive_xgb_modeling.sh.sample)
+ コマンドがやっていること
    - XGBoostでモデルを構築して予測する
    - 後々を考えて, cross-validationの予測結果も残す
+ 編集ログ
    - 2017/10/17執筆
    - 2017/11/15修正: 保存形式変更(.blp → .tsv), ファイル名のつけ方変更


Stacking/blendingによる予測モデル作成
------------
+ コマンドとしてやること
```bash
$ python src/models/train_predict_stacking.py -v  # fold-outでトレーニングデータのサンプルに対し予測値を算出する
$ python src/models/train_predict_stacking.py -t  # トレーニングデータ中すべてのサンプルを使ってblendingモデルを構築し, テストデータの予測値を出す

# `-l`で発電所を指定できる. その場合, `ukishima`・`ougishima`・`yonekurayama`のいずれかを続ける
$ python src/models/train_predict_stacking.py -v -l ougishima
```
+ __入力__: models/xgb/内の予測結果
+ __出力__:
    - pickleでserializeされたblemndingモデル(例↓)
        * models/blending/fit_model.layer1.xgb.n_estimators_1000.max_depth_3.learning_rate_0.05.test.ougishima.pkl
    - 予測値を保存したTSVファイル(例↓)
        * models/blending/predict.layer1.xgb.n_estimators_1000.max_depth_3.learning_rate_0.05.crossval.ougishima.tsv
        * models/blending/predict.layer1.xgb.n_estimators_1000.max_depth_3.learning_rate_0.05.test.ougishima.tsv
+ 注意
    - Blendingモデルについて
        * XGBoost以外のモデルを使いたいならば, src/models/train_predict_stacking.pyを編集する
        * blending部分は関数化したので, コマンドライン以外からでも使用できる(使用例は載せていない)
+ コマンドがやっていること
    - XGBoostの予測結果からblendingする
+ 編集ログ
    - 2017/11/15執筆


次やること
------------
+ XGBoostの結果をまとめる


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
