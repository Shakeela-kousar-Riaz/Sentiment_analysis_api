# Sentiment Analysis REST API

A Python REST API that analyzes English text and classifies its sentiment as **positive**, **negative**, or **neutral**. The API returns a sentiment label and a confidence score between 0 and 1.

## Features

* Pre-trained Hugging Face sentiment model
* Single-text sentiment analysis
* Batch analysis of up to 10 texts
* Input validation for 1–500 words per text
* Confidence scores between 0 and 1
* Automatic Swagger API documentation
* Health-check endpoint
* Automated API tests
* Model evaluation with accuracy, precision, recall, F1-score, and latency

## Tech Stack

* Python
* FastAPI
* Hugging Face Transformers
* PyTorch
* Pydantic
* Pytest
* Hugging Face Datasets
* scikit-learn

## Model

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

The application uses a pre-trained RoBERTa-based model for three-class sentiment classification.

The input is tokenized and passed to the model. The API returns the highest-scoring class and its associated model confidence score.

Long texts are processed in smaller chunks to respect model token limits. Chunk scores are combined using word-count weighting.

**Limitations:** The model was developed for Twitter-style text. Accuracy may differ on formal reviews, customer support messages, or other domains. Confidence scores are not necessarily calibrated probabilities.

## Project Structure

```text
sentiment-analysis-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── sentiment.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── scripts/
│   └── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_PUBLIC_GITHUB_REPOSITORY_URL
cd sentiment-analysis-api
```

Replace the example URL with your actual repository URL.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive documentation:

`http://127.0.0.1:8000/docs`

## API Documentation

### 1. Health Check

**GET** `/api/health`

Example response:

```json
{
  "status": "ok",
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "model_loaded": false
}
```

`model_loaded` becomes true after the model has been initialized successfully.

### 2. Analyze One Text

**POST** `/api/analyze`

Request:

```json
{
  "text": "I really love this product. It works perfectly."
}
```

Example response:

```json
{
  "text": "I really love this product. It works perfectly.",
  "sentiment": "positive",
  "confidence": 0.97
}
```

The numerical confidence is illustrative; the actual result depends on model inference.

### 3. Batch Analysis

**POST** `/api/analyze/batch`

Request:

```json
{
  "texts": [
    "I love this product.",
    "This is a terrible experience.",
    "The package arrived today."
  ]
}
```

Example response structure:

```json
{
  "results": [
    {
      "text": "I love this product.",
      "sentiment": "positive",
      "confidence": 0.97
    },
    {
      "text": "This is a terrible experience.",
      "sentiment": "negative",
      "confidence": 0.94
    },
    {
      "text": "The package arrived today.",
      "sentiment": "neutral",
      "confidence": 0.85
    }
  ]
}
```

The example scores and predictions are illustrative, not measured output.

## Validation and Error Handling

| Condition                | HTTP status |
| ------------------------ | ----------: |
| Valid request            |         200 |
| Empty text               |         422 |
| More than 500 words      |         422 |
| Empty batch              |         422 |
| More than 10 batch texts |         422 |
| Model inference failure  |         503 |

The API validates request data and returns a clear error message when inference fails.

## Automated Tests

Run:

```bash
pytest -v
```

The tests cover the health endpoint, single-text analysis, batch analysis, empty inputs, word limits, and batch-size limits. Model inference is mocked in these API tests.

## Model Evaluation

The evaluation script uses the three-class TweetEval sentiment test split.

Run an initial evaluation:

```bash
python -m scripts.evaluate --limit 100
```

Evaluate 1,000 examples:

```bash
python -m scripts.evaluate --limit 1000
```

Evaluate the entire test split:

```bash
python -m scripts.evaluate --limit 0
```

Record the measured accuracy, per-class precision and recall, macro F1-score, sample count, and latency in the final project report.

### Results

Fill this table using the output from your own evaluation run.

| Metric          | Measured result |
| --------------- | --------------- |
| Test examples   | To be measured  |
| Accuracy        | To be measured  |
| Macro precision | To be measured  |
| Macro recall    | To be measured  |
| Macro F1-score  | To be measured  |
| Average latency | To be measured  |

Do not substitute example values or published model-card metrics for your own test results.

## Deployment

The application can be deployed to a Python-compatible hosting platform such as Render or Railway.

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Configure the host's Python environment to install the dependencies from `requirements.txt`. Ensure sufficient memory is available for PyTorch and the downloaded model.

After deployment, test `/api/health`, `/api/analyze`, and `/api/analyze/batch` using the public API URL.

## Future Improvements

* Add request timeouts and rate limiting.
* Add performance benchmarks after model warm-up.
* Support GPU inference where available.
* Add confidence calibration and broader-domain evaluation.
* Add continuous integration for automated tests.

## License and Attribution

Review the selected model's model card and license before redistribution or commercial use. Credit the model authors and the dataset authors in the final report.
