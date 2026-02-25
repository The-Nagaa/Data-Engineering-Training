# Springer Capital - Referral Program Data Pipeline

Take-home test submission for Data Engineer Intern position.

## What this does

Processes 7 CSV files from the referral program, profiles the data, joins everything into one clean report, and flags potentially fraudulent reward claims based on the business rules given.

Output: `output/final_report.csv` — 46 rows with `is_business_logic_valid` and `invalid_reason` columns.

## Project structure

```
├── pipeline.py                  # main script - run this
├── Dockerfile
├── requirements.txt
├── tests.py
├── data/                        # source CSV files go here
│   ├── lead_log.csv
│   ├── paid_transactions.csv
│   ├── referral_rewards.csv
│   ├── user_logs.csv
│   ├── user_referral_logs.csv
│   ├── user_referral_statuses.csv
│   └── user_referrals.csv
└── output/
    ├── final_report.csv          # generated after running pipeline
    ├── data_dictionary.xlsx      # column definitions for business users
    └── data_profiling/           # one profile CSV per source table
```

## How to run (local)

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
python pipeline.py
```

The script will print progress as it runs. Report is saved to `output/final_report.csv`.

**To run tests:**
```bash
pytest tests.py -v
```

## How to run with Docker

**Build:**
```bash
docker build -t springer-pipeline .
```

**Run and export the output:**
```bash
# Mac/Linux
docker run --rm -v "$(pwd)/output:/app/output" springer-pipeline

# Windows PowerShell
docker run --rm -v "${PWD}/output:/app/output" springer-pipeline
```

The report will appear in your local `output/` folder after the container finishes.

## Pipeline steps

1. **Load** — reads all 7 CSV files, treats "null" strings as NaN
2. **Profile** — generates column stats (nulls, distinct values, min/max) for each table
3. **Clean** — parses timestamps to UTC, fixes boolean columns, extracts reward days from text
4. **Process** — deduplicates user_logs and referral_logs, joins all tables, converts timestamps to local timezone (Asia/Jakarta / Asia/Makassar based on location)
5. **Fraud detection** — applies 5 business logic rules, flags invalid cases
6. **Output** — saves final 46-row CSV

## Business logic (fraud detection)

The `is_business_logic_valid` column is `True` when:

**Valid case 1** — all of these are true:
- reward value > 0
- referral status = Berhasil
- has a linked transaction
- transaction status = PAID and type = NEW
- transaction happened AFTER the referral, in the same month
- referrer's membership is active and account is not deleted
- reward was actually granted

**Valid case 2** — status is Menunggu or Tidak Berhasil with no reward assigned

**Flagged as invalid** when any of these conditions are detected:
- `INVALID C1` — reward given but status isn't Berhasil
- `INVALID C2` — reward given but no transaction linked
- `INVALID C3` — PAID transaction exists after referral but no reward was assigned
- `INVALID C4` — status is Berhasil but reward is null/0
- `INVALID C5` — transaction date is before the referral date

## Notes on the data

- Names and phone numbers in the source files are privacy-hashed (MD5) — this is expected, they're not actual names
- Some referrals don't have a referrer (anonymous Draft Transactions) — these get the transaction's timezone for timestamp conversion
- Club names stay uppercase throughout (business requirement)
- Nulls in transaction/reward columns are expected for pending or failed referrals

## Credentials / cloud storage

No credentials are stored in the code. If you need to upload results to cloud storage, use environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Then add your upload logic to the `save_report()` function.
