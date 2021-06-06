import os

wikiart = os.path.abspath(os.path.join(__file__[:-len('path.py')], "./wikiart"))
kaggle = os.path.abspath(os.path.join(__file__[:-len('path.py')], "./preprocessed_kaggle"))

paths = {
    'wikiart': wikiart,
    'kaggle': kaggle
}