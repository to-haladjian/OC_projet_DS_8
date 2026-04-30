# Credit Scoring API

API de prédiction de défaut de remboursement de crédit, déployée sur HuggingFace Spaces.

## Architecture

```
app/
├── main.py              # FastAPI + Gradio mount
├── artefacts/           # model.onnx + preprocessor.joblib
├── core/                # Config, structured JSON logging
├── routers/             # /health, /predict
├── schemas/             # Pydantic input/output validation
├── services/            # Preprocessing, ONNX inference, DB logging
└── gradio_ui.py         # Interface Gradio
```

**Stack** : FastAPI + Gradio | ONNX Runtime | PostgreSQL (Supabase) | Evidently AI | Docker | GitHub Actions

## Setup local

```bash
# Installer les dépendances
pip install -r requirements.txt
pip install -r requirements-tests.txt

# Configurer les variables d'environnement
cp .env.example .env
# Editer .env avec vos credentials Supabase (optionnel)

# Lancer l'API
uvicorn app.main:app --reload --port 7860
```

- Swagger : http://localhost:7860/docs
- Gradio : http://localhost:7860/gradio
- Health : http://localhost:7860/health

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t credit-scoring .
docker run -p 7860:7860 credit-scoring
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`) : tests automatiques sur push `dev`/`main`
- **Deploy** (`.github/workflows/deploy.yml`) : deploy vers HuggingFace Spaces sur push `main`

Secrets GitHub requis : `HF_TOKEN`

## Monitoring

- **Logging structuré JSON** : chaque prédiction loggée (input, output, temps d'exécution)
- **Supabase PostgreSQL** : stockage des logs de prédiction (si `DATABASE_URL` configuré)
- **Drift detection** : Evidently AI (`monitoring/drift_detection.py`)
- **Notebooks d'analyse** : `notebooks/03_drift_analysis.ipynb`

## RGPD

Les données stockées sont des features numériques/catégorielles anonymisées du dataset Home Credit. Aucune donnée personnelle identifiable n'est collectée.
