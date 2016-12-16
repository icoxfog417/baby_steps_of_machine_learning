# Baby Steps of Machine Learning

機械学習モデルを実装するための、基礎的なステップを学ぶためのリポジトリです。

## 0.Setup

### 0-1. Environmental Construction

環境構築については、以下の手順を参考にしてください。

[Pythonで機械学習アプリケーションの開発環境を構築する](http://qiita.com/icoxfog417/items/950b8af9100b64c0d8f9)

### 0-2. Download Source Code

以下のGitHubリポジトリを**forkして**、そこからダウンロード(もしくはclone)してください。

[icoxfog417/baby_steps_of_machine_learning](https://github.com/icoxfog417/baby_steps_of_machine_learning)  
Starも押していただけると励みになりますm(_ _)m

forkとは、オリジナルのコードをコピーして手元に持ってくることです。これで、自分なりの編集などを行うことができます。

### 0-3. アプリケーションの動作確認

アプリケーションを動作させるための仮想環境を有効化します。  
※以下は、0-1の環境構築の手順通り用意してきた場合(Minicondaで仮想環境ml_envを作成)を想定しています。変えている場合は、適宜読み替えてください。

Windows(コマンドプロンプトから実行)

```
activate ml_env
```

Mac/Linux(ターミナルから実行 (#で始まる行はコメントなので、実行する際は無視してください))

```
# pyenvとのバッティングを防ぐため、activateは仮想環境のパスを確認し直接実行する
conda info -e
# conda environments:
#
ml_env                   /usr/local/pyenv/versions/miniconda-X.X.X/envs/ml_env
root                  *  /usr/local/pyenv/versions/miniconda-X.X.X

source /usr/local/pyenv/versions/minicondaX.X.X/envs/ml_env/bin/activate ml_env
```

※miniconda-X.X.Xは、インストールしたminicondaのバージョンによって変わります。

仮想環境が有効化出来たら、Jupyter Notebookを起動してみます。これは、インタラクティブにPythonの実行ができるアプリケーションです。  
このリポジトリをダウンロードしたフォルダの直下で、以下のコマンドを実行します。

```
jupyter notebook
```

Webブラウザのページが立ち上がったら準備完了です。Jupyter Notebookの使い方については順次紹介していきますが、以下の資料もご参考ください。

[はじめるJupyter Notebook](http://qiita.com/icoxfog417/items/175f69d06f4e590face9)

## 1. Make Machine Learning Model Basic

では、Jupyter Notebookを使って機械学習モデルを作成するプロセスを体感していきます。

Jupyter Notebookを立ち上げて、`worksheet.ipynb`を開いてください。  
こちらには機械学習の各ステップが順番に書かれているので、そちらに従い上から順に実行をしていきます。
途中にある`write your code here`については、この後に解説をしていくためいったん無視して頂いて構いません。解説を読みながら、最後のモデルの保存まで行ってみてください。

## 2. Data Preprocessing

機械学習において、集めたデータがそのまま使える、ということは稀です。  
精度の高いモデルを作るため、学習のスピードを上げるため、データをモデルにとって適切な形に加工する処理が欠かせません(生の食材を、調理するようなイメージです)。  
多くの場合、この前処理が最終的な精度に大きなインパクトを与えます。今回は前処理の一つの手法である、正規化を実行してみます。

![normalization.PNG](./pictures/normalization.PNG)

正規化とは、各特徴について平均を0、標準偏差を1にそろえる処理です。これで学習速度の向上を図ることができます。  
実際、身長や体重といった様々な特徴がある場合、その単位はバラバラでとる値の範囲も異なります。このままでは扱いづらいので、値の範囲をそろえるというのが正規化の役割です。  

以下のコードを、`Data Preprocessing`の下に実装しましょう。正規化を行う`normalization`関数を実装し、それにより`digits.data`の正規化を行っています。

```py3
def normalization(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    stds[stds < 1.0e-6] = np.max(x) - np.min(x)
    means[stds < 1.0e-6] = np.min(x)
    return means, stds

means, stds = normalization(digits.data)
print(means.shape)
print(stds.shape)

normalized_data = (digits.data - means) / stds  # normalization
```

## 3. Split the Training Data and Test Data

機械学習においては、学習に使ったデータと評価用のデータは分ける必要があります。

ここでは、以下2つを実施します。

* データを、学習用と評価用に分ける
* 学習データに対する精度と、評価データに対する精度をそれぞれ算出する

学習データの分割を行には、[`cross_validation.train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html)を使用します。
こちらを利用し、Training the Modelの前に、以下の処理を入れます。

```py3
def split_dataset(dataset, test_size=0.3):
    from sklearn import cross_validation
    from collections import namedtuple

    DataSet = namedtuple("DataSet", ["data", "target"])
    train_d, test_d, train_t, test_t = cross_validation.train_test_split(dataset.data, dataset.target, test_size=test_size, random_state=0)

    left = DataSet(train_d, train_t)
    right = DataSet(test_d, test_t)
    
    return left, right

# use 30% of data to test the model
training_set, test_set = split_dataset(digits, 0.3)
print("dataset is splited to train/test = {0} -> {1}, {2}".format(
        len(digits.data), len(training_set.data), len(test_set.data))
     )

```

上記で`training_set`と`test_set`にデータを分割したので、Training the Modelを以下のように修正します。

```
classifier.fit(training_set.data, training_set.target)
```

これで学習は完了しました。データを分割したおかげで、評価用のデータが30%分のこっています。これを使って学習していないデータに対する精度を計測することができます。

Evaluate the Modelの精度計算部分を、以下のように修正します。

```py3
print(calculate_accuracy(classifier, training_set))
print(calculate_accuracy(classifier, test_set))
```

## 4. Evaluate the Model

＜Handson #3 解説＞

* 過学習になっていないか確認するため、学習データ・評価データそれぞれについての精度がどう変化しているかを測定します。
* データ不均衡による「なんちゃって精度が高いモデル」を防ぐため、適合率・再現率を確認します。

### 4.1 Check Accuracy

以下のスクリプトで、学習/評価データに対する精度の推移を確認します。
この、横軸に学習データ数、縦軸に精度を取ったグラフを学習曲線(learning curver)と呼び、scikit-learnでは[`sklearn.learning_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.learning_curve.learning_curve.html)を利用することで簡単に描画することができます。

```py3
def plot_learning_curve(model_func, dataset):
    from sklearn.learning_curve import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = [i / 10 for i in range(1, 11)]
    train_sizes, train_scores, valid_scores = learning_curve(model_func(), dataset.data, dataset.target, train_sizes=sizes, cv=5)
    
    take_means = lambda s: np.mean(s, axis=1)
    plt.plot(sizes, take_means(train_scores), label="training")
    plt.plot(sizes, take_means(valid_scores), label="test")
    plt.ylim(0, 1.1)
    plt.title("learning curve")
    plt.legend(loc="lower right")
    plt.show()

plot_learning_curve(make_model, digits)
```

追加し終わったら、実行してみて下さい。以下のように図がプロットされるはずです。

![image](https://qiita-image-store.s3.amazonaws.com/0/25990/441770dd-db03-dd82-99d5-98b7ef3cde31.png)

### 4.1 Check Precision and Recall

scikit-learnでは[classification_report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)関数を使うことで、簡単に確認できます。
各ラベル(#0～#9)内で、具体的に予測したもののうちどれだけが合っていたのかなどの分析は[confusion_matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)で行うことができます。

```py3
def show_confusion_matrix(model, dataset):
    from sklearn.metrics import classification_report
    
    predicted = model.predict(dataset.data)
    target_names = ["#{0}".format(i) for i in range(0, 10)]

    print(classification_report(dataset.target, predicted, target_names=target_names))

show_confusion_matrix(classifier, digits)
```

![image](https://qiita-image-store.s3.amazonaws.com/0/25990/ba6f0580-4093-7d11-0270-bbd0e95a698b.png)

