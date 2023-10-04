# Emotion Research
## Usage
1. Place the files in the following structure:
```
├── data
│   ├── anger
│   ├── contempt
│   ├── disgust
│   ├── fear
│   ├── happy
│   ├── neutral
│   ├── sad
│   └── surprise
│   └── labels.csv
├── dataset.py
├── eval.py
├── LICENSE
├── README.md
├── split.py
└── train.py
```
2. Run `python split.py` to split the data into train, validation and testing sets.
3. Download weights from https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq and saved in `pretrained/`
4. Run `python train.py [ARGS]` to train the model.
5. Run `python eval.py` to evaluate the model.  
The metrics returned are accuracy, a confusion matrix and accuracy weighted per class.