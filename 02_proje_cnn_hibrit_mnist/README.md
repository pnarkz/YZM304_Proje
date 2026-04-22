# YZM304 Derin Ogrenme I. Proje Modulu II. Proje Odevi

Bu proje, Ankara Universitesi Yapay Zeka ve Veri Muhendisligi Bolumu `YZM304 Derin Ogrenme` dersi icin hazirlanmistir. Calismada `MNIST` veri kumesi uzerinde evrisimli sinir aglari ile ozellik cikarma ve siniflandirma yapilmaktadir. Proje, odev gereksinimlerine uygun olarak dort ana deney sunar:

1. `Model 1`: Temel CNN.
2. `Model 2`: Ayni katman hiperparametrelerini koruyup Batch Normalization ve Dropout ile iyilestirilmis CNN.
3. `Model 3`: Literaturde yaygin bir mimari olan `ResNet-18`.
4. `Model 4`: `Model 2` tarafindan cikarilan ozellikler uzerinde egitilen hibrit `CNN + SVM`.

`Model 4`, ayni veri kumesi uzerinde egitilen tam CNN olan `Model 2` ile karsilastirilmaktadir. Boylece odevde istenen hibrit yaklasim ile klasik tam CNN yaklasimi ayni veri uzerinde degerlendirilmektedir.

Calismada temel model sonuclarina ek olarak yanlis siniflandirma galerileri, ozellik uzayi izdusumleri ve model karmasikligi karsilastirmasi da uretilmistir.

## Introduction

El yazisi rakam tanima, derin ogrenme literaturunde temel goruntu siniflandirma problemlerinden biridir. `MNIST` veri kumesi, farkli CNN tasarimlarinin hem ogrenme davranisini hem de siniflandirma basarisini karsilastirmak icin uygun bir benchmark saglar.

Bu projede temel amac, farkli CNN tasarimlarinin ayni veri kumesindeki davranisini incelemek ve tam CNN tabanli ozellik cikarma ile klasik makine ogrenmesi birlestirilerek hibrit bir siniflandirma yaklasimi kurmaktir. Bu amacla once temel katmanlari acik bicimde tanimlanmis bir CNN modeli kurulmus, daha sonra ayni cekirdek hiperparametreleri BatchNorm ve Dropout ile iyilestirilmis, son olarak da daha derin bir referans mimari olarak ResNet-18 kullanilmistir.

## Method

### Veri Kumesi ve On Isleme

- Veri kumesi: `MNIST`
- Egitim ornek sayisi: `60,000`
- Test ornek sayisi: `10,000`
- Goruntu boyutu: `28x28`, gri-seviye
- On isleme adimlari:
- CNN modelleri icin goruntuler etrafa `2` piksel `padding` uygulanarak `32x32` boyutuna getirilmistir.
- Piksel degerleri tensore cevrilmistir.
- `Normalize((0.1307,), (0.3081,))` ile normalize edilmistir.

### Model 1: Temel CNN

Bu model, odevde istenen sekilde CNN sinifi olarak acik bicimde yazilmistir. Mimaride temel CNN bilesenleri kullanilmistir:

- `Conv2d(1, 6, 5)` + `ReLU` + `MaxPool`
- Iki paralel `Conv2d(6, 16, 5)` blogunun toplandigi bir ara blok
- `Conv2d(16, 120, 5)`
- `Linear(120, 84)`
- `Linear(84, 10)`

Bu yapi, evrisim, aktivasyon, havuzlama, duzlestirme ve tam baglantili siniflandirma katmanlarini acikca icerdigi icin odevin ilk model gereksinimini karsilamaktadir.

### Model 2: Iyilestirilmis CNN

Bu modelde ilk modeldeki `Conv2d` kanal sayilari, kernel boyutlari ve tam baglantili katman boyutlari korunmustur. Iyilestirme amaciyla:

- Evrisim bloklarina `BatchNorm2d`
- Evrisim ve tam baglantili katman cikislarina `Dropout`

eklenmistir. Bu tercih, egitimin daha stabil olmasi ve asiri ogrenmenin azaltilmasi beklentisiyle yapilmistir.

### Model 3: ResNet-18

Ucuncu model olarak `torchvision.models.resnet18(weights=None)` kullanilmistir. Girdi kanal sayisi `1` olacak sekilde ilk evrisim katmani guncellenmis, son tam baglantili katman ise `10` sinif icin yeniden tanimlanmistir. Bu model, daha derin artik baglantili bir mimarinin ayni problemdeki performansini gozlemlemek icin eklenmistir.

### Model 4: Hibrit CNN + SVM

Hibrit model asagidaki adimlarla kurulmustur:

1. `Model 2` egitilir.
2. Son siniflandirici katmandan onceki `84` boyutlu temsil vektoru ozellik olarak kullanilir.
3. Egitim ve test ozellikleri `train_features.npy`, `train_labels.npy`, `test_features.npy`, `test_labels.npy` dosyalarina kaydedilir.
4. Bu ozellikler uzerinde `StandardScaler + SVC(RBF)` boru hatti egitilir.
5. Hibrit modelin test basarimi, ayni veri kumesi uzerindeki tam CNN olan `Model 2` ile karsilastirilir.

### Kayip Fonksiyonu ve Hiperparametreler

- Loss function: `nn.CrossEntropyLoss()`
- Optimizer: `Adam`
- Varsayilan ogrenme orani: `1e-3`
- Varsayilan batch size: `64`
- Varsayilan epoch sayisi: `5`

Bu degerler, MNIST gibi nispeten kolay bir veri kumesinde gozlenebilir ogrenme davranisi uretecek ve ayni zamanda makul egitim suresi sunacak bir baslangic noktasi sagladigi icin secilmistir. Nihai deneyler CUDA destekli GPU ortaminda (`NVIDIA GeForce RTX 3060 Laptop GPU`, `CUDA 12.6`) gerceklestirilmistir.

## Results

Kod calistirildiginda asagidaki ciktilar uretilir:

- `outputs/summary.json`: tum modellerin test accuracy degerleri
- `outputs/metrics/*.json`: model bazli metrikler ve classification report
- `outputs/plots/*_history.png`: loss ve accuracy egitim grafikleri
- `outputs/plots/*_confusion_matrix.png`: karmasiklik matrisi gorselleri
- `outputs/plots/*_misclassified_grid.png`: en guvenli yanlis tahmin galerileri
- `outputs/plots/*_feature_projection.png`: CNN ozelliklerinin PCA ile 2B izdusumu
- `outputs/plots/accuracy_vs_complexity.png`: dogruluk-parametre karmasikligi karsilastirmasi
- `outputs/features/*.npy`: hibrit model icin kaydedilen ozellik ve etiket dosyalari
- `outputs/reports/results_table.md`: sonuc tablosu

Asagidaki sonuclar `epochs=5`, `batch_size=64`, `feature_batch_size=256` ve `learning_rate=0.001` ayarlariyla elde edilmistir:

| Model | Test Accuracy | Not |
| --- | ---: | --- |
| Model 1 - Temel CNN | 0.9879 | Temel model |
| Model 2 - Iyilestirilmis CNN | 0.9901 | BatchNorm + Dropout |
| Model 3 - ResNet-18 | 0.9837 | Daha derin referans mimari |
| Model 4 - Hibrit CNN + SVM | 0.9921 | Model 2 ozellikleri + SVM |

Epoch bazli test accuracy ozeti:

- `Model 1`: `0.9786 -> 0.9862 -> 0.9834 -> 0.9898 -> 0.9879`
- `Model 2`: `0.9820 -> 0.9892 -> 0.9901 -> 0.9899 -> 0.9901`
- `Model 3`: `0.9823 -> 0.9759 -> 0.9900 -> 0.9855 -> 0.9837`

Hibrit model icin olusturulan ozellik dosyalarinin boyutlari:

- `train_features shape: (60000, 84)`
- `train_labels shape: (60000,)`
- `test_features shape: (10000, 84)`
- `test_labels shape: (10000,)`

Tam CNN ile hibrit model karsilastirmasi:

- Referans tam CNN: `Model 2 - Iyilestirilmis CNN`
- Tam CNN accuracy: `0.9901`
- Hibrit model accuracy: `0.9921`
- Accuracy farki: `+0.0020`

Bu sonuclar, hibrit yaklasimin ayni veri uzerinde tam CNN modelini cok kucuk bir farkla gectigini gostermektedir.

## Discussion

Deney sonuclarina gore en yuksek test basarimi `Model 4 - Hibrit CNN + SVM` tarafindan `0.9921` accuracy ile elde edilmistir. Onu `Model 2 - Iyilestirilmis CNN` modeli `0.9901` accuracy ile takip etmektedir. Bu durum, `BatchNorm` ve `Dropout` ile guclendirilen ikinci modelin temel modele gore daha iyi genellestigini gostermektedir. Nitekim `Model 1` ile `Model 2` arasindaki fark `0.0022` olup, iyilestirilmis yapinin yine daha yuksek son test basarimi verdigi gorulmektedir.

`Model 2`'nin epochlar boyunca daha stabil bir test performansi sergilemesi, eklenen duzenlilestirme katmanlarinin egitimi dengeledigini dusundurmektedir. Buna karsin `Model 1` ve `Model 3`, bazi epochlarda daha yuksek seviyelere cikmalarina ragmen son epochta gerileme gostermistir. Bu da tek basina daha buyuk veya daha derin bir model secmenin daha iyi sonuc garanti etmedigini ortaya koymaktadir.

`ResNet-18` modelinin `0.9837` accuracy ile iyilestirilmis ikinci modelin gerisinde kalmasi, MNIST'in dusuk boyutlu ve tek kanalli yapisi nedeniyle cok derin bir mimariye her zaman ihtiyac duyulmadigini gostermektedir. Daha yuksek temsil gucune sahip modeller, problem basit oldugunda veya hiperparametreler veri setine tam uyarlanmadiginda daha hafif modellere gore avantaj saglamayabilir.

Hibrit modelin `84` boyutlu CNN ozellikleri uzerinde `SVM` ile egitildiginde tam CNN modelini `+0.0020` farkla gecmesi, ogrenilen ozelliklerin klasik makine ogrenmesi yontemleri icin de ayirici oldugunu gostermektedir. Fark kucuk olsa da hibrit yaklasimin uygulanabilir oldugu gorulmektedir.

Karmasiklik matrisleri, yanlis siniflandirma galerileri ve PCA tabanli ozellik izdusumleri sayesinde modellerin hangi durumlarda hata yaptigi ve siniflarin ozellik uzayinda nasil ayrildigi da incelenebilmektedir. Bu gorseller, sayisal sonuclarin yorumlanmasini kolaylastirmistir.

## References

1. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 1998.
2. Deng, L. The MNIST Database of Handwritten Digit Images for Machine Learning Research. *IEEE Signal Processing Magazine*, 2012.
3. He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
4. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
5. Torchvision Model Documentation: https://pytorch.org/vision/stable/models.html

## Calistirma

Asagidaki komut varsayilan ayarlarla tum deneyleri calistirir:

```bash
python main.py
```

Daha kisa bir deneme icin:

```bash
python main.py --epochs 2 --batch-size 64 --max-train-samples 2000 --max-test-samples 500
```

MNIST indirmesi yarida kesilirse `data/MNIST/raw` altinda eksik `.gz` dosyalari kalabilir. Bu durumda ilgili klasoru temizleyip komutu tekrar calistirin.

## Proje Yapisi

```text
.
|-- YZM304Lab5_MNIST_LeNet (1).ipynb
|-- main.py
|-- README.md
`-- src
    |-- data.py
    |-- hybrid.py
    |-- models.py
    `-- training.py
```
