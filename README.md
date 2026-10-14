Detecting Stance in Media on Global Warming for Indic Languages

CSET 346 – Natural Language Processing · Bennett University

Three-class stance detection (Favour / Against / None) for Hindi and Bengali YouTube comments toward the claim "Climate change and global warming is a serious concern."

Team member	Enrollment no.
Akshat Dhaka	S24CSEU2096
Pratham Mishra	S24CSEU2062
Arpit Sharma	S24CSEU2100
Tanish Vashishtha	S24CSEU0589
Ramansh Singh	S24CSEU0312

Instructor: Dr. Supriya Chanda, SCSET.

Task
Label	Meaning
Favour	Agrees that climate change / global warming is a serious concern
Against	Disagrees with the claim
None	Neutral, unrelated, or no clear stance

Primary metric: macro F1. Also reported: accuracy, precision, recall, confusion matrix. The final test set is hidden; the training and test files are shared by email.

Repository layout
configs/config.yaml     all paths, column names, label aliases, hyper-parameters
data/raw/               put train.csv / test.csv here (git-ignored)
src/stance/             preprocessing, data loading, models, metrics
scripts/                runnable entry points (see Quick start)
tests/                  unit + smoke tests (pytest)
outputs/                metrics, saved models, validation predictions (git-ignored)
predictions/            submission files (ID, Predicted Label)
reports/                technical report and figures
Setup
bash
git clone <this-repo-url> && cd stance-detection-indic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                                # sanity check

A GPU is recommended for the transformer runs (Google Colab / Kaggle works; use pip install -r requirements.txt then run the same commands).

Quick start
Copy the files received by email to data/raw/train.csv and data/raw/test.csv.
Open configs/config.yaml and set data.id_col, data.text_col, data.label_col to the real column names (and extend labels.aliases if the label values differ).
Run the pipeline:
bash
python scripts/explore_data.py                  # dataset audit -> outputs/data_report.json
python scripts/train_baseline.py --model tfidf  # TF-IDF + linear classifier baseline
python scripts/train_baseline.py --model fasttext   # optional
python scripts/train_transformer.py --model-name google/muril-base-cased
python scripts/train_transformer.py --model-name xlm-roberta-base
python scripts/error_analysis.py --preds outputs/val_predictions_tfidf.csv
python scripts/predict.py --model ensemble --model-dir outputs/transformer/muril-base-cased/best

The submission file is written to predictions/predictions_<model>.csv with columns ID, Predicted Label.

Method
Preprocessing (src/stance/preprocess.py): Unicode NFC, URL / @mention removal, hashtag cleanup, repeated-character collapsing, emoji handling (kept by default; stance cues are preserved), optional Indic normalisation. Script detection (Devanagari / Bengali / Latin) gives a language tag for per-language results.
Baseline: word + character n-gram TF-IDF with logistic regression / linear SVM, class-balanced.
Representations: FastText (subword) and fine-tuned multilingual transformers (mBERT, XLM-R, MuRIL, IndicBERT).
Training: fixed stratified train/validation split, class-weighted cross-entropy, model selection on validation macro F1.
Ensemble: probability averaging of TF-IDF and a transformer (predict.py --model ensemble).
Evaluation: macro F1 (primary), accuracy, precision, recall, confusion matrix, per-language breakdown.
Error analysis: error_analysis.py samples misclassified comments into a CSV for manual categorisation.

Reproducibility: one seed in the config, preprocessing is fitted on training data only, the test set is never used for tuning.

Results

Filled in after the experiments are run (validation split).

Model	Macro F1	Accuracy	Hindi F1	Bengali F1
TF-IDF + LogReg	–	–	–	–
FastText	–	–	–	–
mBERT	–	–	–	–
XLM-R	–	–	–	–
MuRIL	–	–	–	–
IndicBERT	–	–	–	–
Ensemble	–	–	–	–
Submission checklist
 Source code (this repository) with README
 Technical report, max 8 pages (Introduction, Dataset, Methodology, Experiments, Results, Error Analysis, Conclusion, References) → reports/
 Prediction file (ID, Predicted Label) → predictions/
 Presentation slides
Libraries, pre-trained models and tools used

Libraries: scikit-learn, pandas, NumPy, SciPy, PyTorch, Hugging Face Transformers / Datasets, FastText, emoji, indic-nlp-library (optional), matplotlib, PyYAML, pytest.

Pre-trained models (downloaded from Hugging Face when used): bert-base-multilingual-cased, xlm-roberta-base, google/muril-base-cased, ai4bharat/IndicBERTv2-MLM-only.

AI assistance: the initial project scaffold (repository structure, pipeline code and tests) was generated with Claude (Anthropic) and then reviewed by the team. Update this section with any LLM used for the "LLM-based models" part of the assignment, as required by the submission guidelines.

References
Mohammad et al. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets.
Devlin et al. (2019). BERT. NAACL-HLT.
Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.
Kakwani et al. (2020). IndicNLPSuite. Findings of EMNLP.
Khanuja et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv.
License

MIT, see LICENSE
