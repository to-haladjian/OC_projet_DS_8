# Supabase Setup — Production Prediction Logging

This guide walks through configuring [Supabase](https://supabase.com/) (free tier, no credit card required) as the production data store for the credit-scoring API. Once configured, every `/predict` call writes a row to `prediction_logs`; the Streamlit dashboard, drift notebook, and benchmark scripts can then read real production traffic.

The free tier is sufficient for this project (500 MB database, 50 000 monthly active users, 1 GB egress). Note that **the project auto-pauses after 7 days of inactivity** — unpause it from the dashboard before the soutenance.

---

## 1. Create a Supabase account and project

1. Open https://supabase.com/dashboard and sign in with GitHub (or email).
2. Click **New project**, then fill in:
   - **Name**: `credit-scoring-mlops` (or any short name)
   - **Database password**: generate a strong one and store it in your password manager — Supabase will not show it again in plain text
   - **Region**: the one closest to your users / Hugging Face Space (e.g. `West EU (Ireland)`)
   - **Pricing plan**: Free
3. Wait ~2 minutes for the project to provision.

> **Screenshot #1 — Project dashboard**: capture the project home page (showing the project name, region, and "Active" status). Save as `docs/screenshots/01_supabase_project.png`.

---

## 2. Grab the connection string

1. In the project dashboard, click the **Connect** button at the top of the page. A modal opens with the available connection methods.
2. In the modal, switch between the modes using the selector at the top:
   - **Session pooler** (port 5432) — use this for long-lived processes (local dev, scripts, the dashboard).
   - **Transaction pooler** (port 6543) — use this for serverless/short-lived workers (Hugging Face Spaces).
   - **Direct connection** (port 5432) — fine for one-off SQL clients; not recommended for the API because it doesn't survive Supabase's IPv6 networking on most hosts.
3. Copy the **URI** value displayed for the selected mode and replace `[YOUR-PASSWORD]` with the password from step 1.

> **Screenshot #2 — Connect modal**: capture the Connect modal with the Session pooler URI visible (password redacted in the screenshot). Save as `docs/screenshots/02_connection_string.png`.

---

## 3. Wire the URL into the API

The API reads `DATABASE_URL` from the environment. The DB module silently disables logging when the variable is missing, so this is a soft dependency.

**Local development** — copy `.env.example` to `.env` and paste the **session pooler** URI:

```bash
cp .env.example .env
# edit .env and set:
# DATABASE_URL=postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

**Hugging Face Space (production)** — in the Space's **Settings → Variables and secrets**, add a **Secret** named `DATABASE_URL` with the **transaction pooler** URI (port 6543).

**GitHub Actions (CI)** — only needed if you want CI to write to the same DB. The current `ci.yml` does not require it; database-touching tests are marked `@pytest.mark.db` and skipped when `DATABASE_URL` is unset.

---

## 4. Create the `prediction_logs` table

Open **SQL Editor** in the Supabase dashboard, paste the DDL below, and run it. The schema mirrors `database/models/prediction_log.py`.

```sql
create table if not exists public.prediction_logs (
  id                   bigserial primary key,
  timestamp            timestamptz not null default now(),
  input_features       jsonb       not null,
  default_probability  double precision not null,
  credit_approved      boolean     not null,
  execution_time_ms    double precision not null,
  model_version        text        not null
);

create index if not exists prediction_logs_timestamp_idx
  on public.prediction_logs (timestamp desc);
```

The `timestamp` index keeps the dashboard's "most recent N" query fast as the table grows.

Verify with:

```sql
select count(*) from public.prediction_logs;
```

You should see `0`.

> **Screenshot #3 — Table editor**: open **Table Editor → prediction_logs**, capture the columns view (showing the 7 columns and their types). Save as `docs/screenshots/03_table_schema.png`.

---

## 5. Seed the table with real predictions

The fastest way to populate rows for the screenshots is to fire the running API. Start the API locally with the `DATABASE_URL` set:

```bash
set -a; source .env; set +a   # robust: handles @, ?, # in the password
uvicorn app.main:app --port 7860
```

On startup the API should log `Database logging enabled (DATABASE_URL is set)`. If you see `disabled`, run `echo "$DATABASE_URL"` — if empty, the `.env` line probably has stray quotes or a `=` inside the password that needs URL-encoding (see the table in step 3 of the main setup).

Then in another terminal, send ~20 predictions with varied inputs:

```bash
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:7860/predict \
    -H "Content-Type: application/json" \
    -d "{
      \"application_id\": $((100000 + i)),
      \"code_gender\": \"$([ $((i % 2)) -eq 0 ] && echo M || echo F)\",
      \"flag_own_car\": \"N\",
      \"name_contract_type\": \"Cash loans\",
      \"name_family_status\": \"Married\",
      \"name_education_type\": \"Higher education\",
      \"organization_type\": \"Self-employed\",
      \"amt_income_total\": $((100000 + i * 5000)),
      \"amt_credit\": 500000,
      \"amt_annuity\": 25000,
      \"amt_goods_price\": 450000,
      \"days_birth\": $((-12000 - i * 30)),
      \"days_employed\": -3000,
      \"days_registration\": -5000,
      \"days_id_publish\": -4000,
      \"days_last_phone_change\": -1000,
      \"ext_source_2\": 0.5,
      \"ext_source_3\": 0.4,
      \"region_population_relative\": 0.02,
      \"region_rating_client_w_city\": 2,
      \"obs_30_cnt_social_circle\": 1,
      \"def_30_cnt_social_circle\": 0,
      \"amt_req_credit_bureau_qrt\": 0,
      \"reg_city_not_live_city\": 0,
      \"floorsmax_avg\": 0.2,
      \"totalarea_mode\": 0.1,
      \"years_beginexpluatation_medi\": 0.97,
      \"flag_document_3\": 1,
      \"cnt_fam_members\": 2
    }" > /dev/null
done
```

Re-run the verify query — `select count(*) from public.prediction_logs;` should now return `20`.

> **Screenshot #4 — Sample rows**: in the Table Editor, sort by `timestamp desc` and capture the populated table with several rows visible. Save as `docs/screenshots/04_sample_rows.png`.

---

## 6. Verify the dashboard reads the data

With `DATABASE_URL` still exported:

```bash
pip install -r requirements-monitoring.txt
streamlit run monitoring/dashboard.py
```

The KPIs should show 20 predictions and the histograms should populate. The "Generate drift report" button will work once the table has at least a few dozen rows.

---

## Troubleshooting

- **`relation "prediction_logs" does not exist`** — the DDL targeted the wrong schema. Re-run it in **SQL Editor** with the schema selector set to `public`.
- **`SSL connection has been closed`** — your Supabase project went to sleep; open the dashboard and click **Resume**, then retry.
- **API starts but no rows appear** — confirm `DATABASE_URL` is exported in the **same shell** that ran `uvicorn`. Check the API logs for `Database logging enabled` on startup.
- **`pgbouncer error: prepared statement does not exist`** — you're hitting the transaction pooler (port 6543) with a long-lived SQLAlchemy connection. Switch to the session pooler (port 5432) for local work, keep the transaction pooler only for the Hugging Face Space.
