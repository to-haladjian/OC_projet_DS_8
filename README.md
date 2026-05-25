---
title: Credit Payback Scoring
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Credit Scoring API

API de prédiction de défaut de remboursement de crédit, déployée sur HuggingFace Spaces.

## Architecture

```
app/                     # API : FastAPI + Gradio
├── main.py              # Entrée FastAPI, montage Gradio
├── artefacts/           # model.onnx, preprocessor.joblib, feature_list.csv
├── core/                # Config, logging JSON structuré
├── routers/             # /health, /predict
├── schemas/             # Validation Pydantic (entrées/sorties)
├── services/            # Preprocessing, inférence ONNX, persistance DB
└── gradio_ui.py         # UI Gradio (28 champs)

database/                # Modèles SQLAlchemy + session (Supabase / PostgreSQL)
monitoring/              # drift_detection.py + dashboard.py (Streamlit) + reference_data.csv
optimization/            # cProfile + benchmark sklearn vs ONNX
notebooks/               # 03_drift_analysis.ipynb (analyse drift narrée)
docs/                    # OpenAPI, rapport d'optimisation, guide Supabase, drift_report.html
tests/                   # pytest (37 tests, 93% coverage)
.github/workflows/       # ci.yml (tests) + deploy.yml (HF Spaces)
```

**Stack** : FastAPI + Gradio | ONNX Runtime | PostgreSQL (Supabase) | Evidently AI | Streamlit | Docker | GitHub Actions

## Setup local

```bash
# Dépendances
pip install -r requirements.txt
pip install -r requirements-tests.txt

# Variables d'environnement (optionnel : DATABASE_URL pour la persistance)
cp .env.example .env

# Lancer l'API
uvicorn app.main:app --reload --port 7860
```

- Gradio UI : <http://localhost:7860/>
- Swagger : <http://localhost:7860/api/docs>
- ReDoc : <http://localhost:7860/api/redoc>
- Health : <http://localhost:7860/health>

## Tests

```bash
pytest -v
```

37 tests, 93% de couverture. Couverture HTML générée dans `docs/coverage/`.

## Docker

```bash
docker build -t credit-scoring .
docker run -p 7860:7860 credit-scoring
```

## CI/CD

- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) — pytest sur chaque push/PR vers `main` ou `dev`.
- **Deploy** ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) — sur push `main`, rejoue les tests puis upload le dépôt vers le HuggingFace Space `torphinator/credit_payback_scoring`.

Secret GitHub requis : `HF_TOKEN`.

## Monitoring

### Logs et stockage
- **JSON structuré** : chaque prédiction logue input, probabilité, décision, latence (`app/core/logging.py`).
- **Persistance Supabase PostgreSQL** (quand `DATABASE_URL` est défini) : prédictions dans `prediction_logs` (`app/services/db_service.py`) et logs applicatifs dans `app_logs` (`SupabaseLogHandler`), pour survivre aux redémarrages éphémères des HF Spaces. Voir [`docs/supabase_setup.md`](docs/supabase_setup.md) pour le free tier + DDL + screenshots à capturer.
- **Gestion des erreurs** : validation Pydantic (422), écritures DB/logs fail-safe (jamais bloquantes), et handler global renvoyant un 500 JSON propre sans fuite de stack trace (`app/main.py`).

### Dashboard Streamlit

KPIs (volume, taux d'approbation, latence p50/p95), distributions, série temporelle, et rapport de drift Evidently à la demande.

```bash
pip install -r requirements-monitoring.txt
streamlit run monitoring/dashboard.py
```

Le dossier `monitoring/` est exclu de l'image Docker HF Spaces (`deploy.yml` `ignore_patterns`), donc Streamlit n'alourdit pas le déploiement.

### Drift detection
- Notebook narré : [`notebooks/03_drift_analysis.ipynb`](notebooks/03_drift_analysis.ipynb).
- Rapport Evidently rendu : [`docs/drift_report.html`](docs/drift_report.html).
- Module réutilisable : `monitoring/drift_detection.py` (référence : `monitoring/reference_data.csv`).

## Optimisation

Profil cProfile + benchmark sklearn LightGBM (baseline Part 1) vs ONNX Runtime (production).

```bash
python optimization/profile_inference.py   # écrit optimization/profile.prof
python optimization/benchmark.py            # écrit optimization/benchmark_results.json
```

Conclusions et chiffres : [`docs/optimization_report.md`](docs/optimization_report.md).

## RGPD

Les données stockées sont des features numériques/catégorielles anonymisées du dataset Home Credit. Aucune donnée personnelle identifiable n'est collectée.

## Livrables — mission OpenClassrooms

| Livrable | Fichier(s) |
| --- | --- |
| Historique Git | branche `dev` (10+ commits), branche `main` (release) |
| API fonctionnelle | [`app/main.py`](app/main.py), [`app/routers/prediction.py`](app/routers/prediction.py), [`app/gradio_ui.py`](app/gradio_ui.py) |
| Tests unitaires | [`tests/`](tests/), 37 tests / 93% coverage |
| Dockerfile | [`Dockerfile`](Dockerfile) |
| Pipeline CI/CD | [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) |
| Stockage données de production | [`app/services/db_service.py`](app/services/db_service.py), [`database/models/prediction_log.py`](database/models/prediction_log.py), guide [`docs/supabase_setup.md`](docs/supabase_setup.md), screenshots `docs/screenshots/` |
| Analyse du Data Drift | [`notebooks/03_drift_analysis.ipynb`](notebooks/03_drift_analysis.ipynb), [`docs/drift_report.html`](docs/drift_report.html), [`monitoring/drift_detection.py`](monitoring/drift_detection.py) |
| Dashboard / rapport de monitoring | [`monitoring/dashboard.py`](monitoring/dashboard.py) (Streamlit) |
| Rapport d'optimisation | [`docs/optimization_report.md`](docs/optimization_report.md), [`optimization/benchmark.py`](optimization/benchmark.py), [`optimization/profile_inference.py`](optimization/profile_inference.py) |
| Documentation | ce README + [`docs/`](docs/) |
