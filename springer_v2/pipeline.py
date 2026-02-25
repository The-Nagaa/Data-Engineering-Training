# Springer Capital - Data Engineer Intern Take Home Test
# Author: [Your Name]
# 
# This script processes the referral program data, does profiling,
# joins everything together and checks business logic for fraud detection.
# Output: final_report.csv (46 rows)

import os
import pandas as pd
from zoneinfo import ZoneInfo
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROFILING_DIR = os.path.join(OUTPUT_DIR, "data_profiling")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILING_DIR, exist_ok=True)

NULL_VALS = ["null", "NULL", "none", "None", ""]


# ---------------------------------------------------------------
# DATA PROFILING
# ---------------------------------------------------------------

def profile_dataframe(df, table_name):
    """Profile a dataframe - get null count, distinct count, min, max per column"""
    rows = []
    for col in df.columns:
        col_data = df[col]
        non_null = col_data.dropna()
        rows.append({
            "column_name":    col,
            "data_type":      str(col_data.dtype),
            "total_rows":     len(col_data),
            "null_count":     int(col_data.isna().sum()),
            "null_pct":       round(col_data.isna().mean() * 100, 1),
            "distinct_count": int(col_data.nunique(dropna=False)),
            "min_value":      str(non_null.min()) if len(non_null) > 0 else None,
            "max_value":      str(non_null.max()) if len(non_null) > 0 else None,
        })

    profile_df = pd.DataFrame(rows)
    save_path = os.path.join(PROFILING_DIR, f"profile_{table_name}.csv")
    profile_df.to_csv(save_path, index=False)
    logger.info(f"Profiled {table_name} -> {save_path}")


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

def load_data():
    logger.info("Loading CSV files...")

    lead_logs              = pd.read_csv(os.path.join(DATA_DIR, "lead_log.csv"), na_values=NULL_VALS)
    user_referrals         = pd.read_csv(os.path.join(DATA_DIR, "user_referrals.csv"), na_values=NULL_VALS)
    user_referral_logs     = pd.read_csv(os.path.join(DATA_DIR, "user_referral_logs.csv"), na_values=NULL_VALS)
    user_logs              = pd.read_csv(os.path.join(DATA_DIR, "user_logs.csv"), na_values=NULL_VALS)
    user_referral_statuses = pd.read_csv(os.path.join(DATA_DIR, "user_referral_statuses.csv"), na_values=NULL_VALS)
    referral_rewards       = pd.read_csv(os.path.join(DATA_DIR, "referral_rewards.csv"), na_values=NULL_VALS)
    paid_transactions      = pd.read_csv(os.path.join(DATA_DIR, "paid_transactions.csv"), na_values=NULL_VALS)

    logger.info(f"  lead_logs: {len(lead_logs)} rows")
    logger.info(f"  user_referrals: {len(user_referrals)} rows")
    logger.info(f"  user_referral_logs: {len(user_referral_logs)} rows")
    logger.info(f"  user_logs: {len(user_logs)} rows")
    logger.info(f"  user_referral_statuses: {len(user_referral_statuses)} rows")
    logger.info(f"  referral_rewards: {len(referral_rewards)} rows")
    logger.info(f"  paid_transactions: {len(paid_transactions)} rows")

    return (lead_logs, user_referrals, user_referral_logs,
            user_logs, user_referral_statuses, referral_rewards, paid_transactions)


# ---------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------

def parse_timestamps(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def to_bool(series):
    return series.astype(str).str.lower().map({
        "true": True, "false": False,
        "1": True,    "0": False
    })


def clean_data(lead_logs, user_referrals, user_referral_logs,
               user_logs, user_referral_statuses, referral_rewards, paid_transactions):
    logger.info("Cleaning data...")

    # parse timestamps
    user_referrals         = parse_timestamps(user_referrals,         ["referral_at", "updated_at"])
    user_referral_logs     = parse_timestamps(user_referral_logs,     ["created_at"])
    user_logs              = parse_timestamps(user_logs,              ["membership_expired_date"])
    paid_transactions      = parse_timestamps(paid_transactions,      ["transaction_at"])
    lead_logs              = parse_timestamps(lead_logs,              ["created_at"])
    user_referral_statuses = parse_timestamps(user_referral_statuses, ["created_at"])
    referral_rewards       = parse_timestamps(referral_rewards,       ["created_at"])

    # fix booleans
    user_referral_logs["is_reward_granted"] = to_bool(user_referral_logs["is_reward_granted"])
    user_logs["is_deleted"]                 = to_bool(user_logs["is_deleted"])

    # reward_value is stored as "10 days", "20 days" etc - extract number
    referral_rewards["num_reward_days"] = (
        referral_rewards["reward_value"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .astype("Int64")
    )

    # initcap for string columns that are actual text (not hashed IDs, not club names)
    # note: names/phones in this dataset are hashed for privacy - don't touch them
    # club names stay uppercase per business requirement
    for df, col in [
        (user_referrals,         "referral_source"),
        (user_referral_statuses, "description"),
        (paid_transactions,      "transaction_type"),
        (paid_transactions,      "transaction_status"),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)

    return (lead_logs, user_referrals, user_referral_logs,
            user_logs, user_referral_statuses, referral_rewards, paid_transactions)


# ---------------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------------

def utc_to_local(ts, tz_str, fallback_tz="Asia/Jakarta"):
    """Convert UTC timestamp to local time. Returns naive timestamp."""
    if pd.isna(ts):
        return pd.NaT
    tz = fallback_tz if (pd.isna(tz_str) or not tz_str) else str(tz_str)
    try:
        return ts.astimezone(ZoneInfo(tz)).replace(tzinfo=None)
    except Exception:
        return ts.replace(tzinfo=None)


def drop_tz(ts):
    if pd.notna(ts) and hasattr(ts, "tzinfo") and ts.tzinfo:
        return ts.replace(tzinfo=None)
    return ts


def get_source_category(referral_source, lead_category):
    """
    Derive referral_source_category based on business logic:
    - User Sign Up      -> Online
    - Draft Transaction -> Offline
    - Lead              -> use lead_logs.source_category
    """
    if referral_source == "User Sign Up":
        return "Online"
    elif referral_source == "Draft Transaction":
        return "Offline"
    elif referral_source == "Lead":
        return lead_category if pd.notna(lead_category) else "Online"
    return None


def process_and_join(lead_logs, user_referrals, user_referral_logs,
                     user_logs, user_referral_statuses, referral_rewards, paid_transactions):
    logger.info("Processing and joining tables...")

    # deduplicate user_logs - keep latest entry per user
    user_logs_clean = (
        user_logs
        .sort_values("membership_expired_date", ascending=False, na_position="last")
        .drop_duplicates(subset="user_id", keep="first")
    )

    # deduplicate user_referral_logs - keep latest per referral
    referral_logs_clean = (
        user_referral_logs
        .sort_values("created_at", ascending=False)
        .drop_duplicates(subset="user_referral_id", keep="first")
        .rename(columns={
            "user_referral_id": "referral_id",
            "created_at":       "reward_granted_at"
        })
    )

    # lead source category lookup
    lead_lookup = (
        lead_logs[["lead_id", "source_category"]]
        .drop_duplicates("lead_id")
        .rename(columns={"lead_id": "referee_id", "source_category": "lead_source_category"})
    )

    # start joining - base is user_referrals
    df = user_referrals.copy()

    # join lead source
    df = df.merge(lead_lookup, on="referee_id", how="left")
    df["referral_source_category"] = df.apply(
        lambda r: get_source_category(r["referral_source"], r.get("lead_source_category")), axis=1
    )

    # join referral status
    status_lookup = user_referral_statuses[["id", "description"]].rename(
        columns={"id": "user_referral_status_id", "description": "referral_status"}
    )
    df = df.merge(status_lookup, on="user_referral_status_id", how="left")

    # join reward info
    reward_lookup = referral_rewards[["id", "num_reward_days"]].rename(
        columns={"id": "referral_reward_id"}
    )
    df = df.merge(reward_lookup, on="referral_reward_id", how="left")

    # join transaction details
    df = df.merge(
        paid_transactions.rename(columns={"transaction_at": "transaction_at_utc"}),
        on="transaction_id", how="left"
    )

    # join referrer info from user_logs
    referrer_info = user_logs_clean[[
        "user_id", "name", "phone_number", "homeclub",
        "timezone_homeclub", "membership_expired_date", "is_deleted"
    ]].rename(columns={
        "user_id":                 "referrer_id",
        "name":                    "referrer_name",
        "phone_number":            "referrer_phone_number",
        "homeclub":                "referrer_homeclub",
        "timezone_homeclub":       "referrer_tz",
        "membership_expired_date": "referrer_membership_exp",
        "is_deleted":              "referrer_is_deleted",
    })
    df = df.merge(referrer_info, on="referrer_id", how="left")

    # join reward granted log
    df = df.merge(
        referral_logs_clean[["referral_id", "reward_granted_at", "is_reward_granted"]],
        on="referral_id", how="left"
    )

    # timezone conversions
    # referral_at: use referrer timezone, fallback to transaction timezone if no referrer
    def convert_referral_at(row):
        tz = row.get("referrer_tz")
        if pd.isna(tz):
            tz = row.get("timezone_transaction")
        return utc_to_local(row["referral_at"], tz)

    df["referral_at"]    = df.apply(convert_referral_at, axis=1)
    df["transaction_at"] = df.apply(
        lambda r: utc_to_local(r["transaction_at_utc"], r.get("timezone_transaction")), axis=1
    )

    # strip tz from other timestamps
    df["updated_at"]              = df["updated_at"].apply(drop_tz)
    df["reward_granted_at"]       = df["reward_granted_at"].apply(drop_tz)
    df["referrer_membership_exp"] = df["referrer_membership_exp"].apply(drop_tz)

    # add sequential ID
    df = df.reset_index(drop=True)
    df.insert(0, "referral_details_id", df.index + 101)

    logger.info(f"Join complete: {len(df)} rows")
    return df


# ---------------------------------------------------------------
# BUSINESS LOGIC - FRAUD DETECTION
# ---------------------------------------------------------------

def check_business_logic(row):
    """
    Check if a referral reward is valid based on business rules.
    Returns (is_valid: bool, reason: str or None)
    """
    reward     = row.get("num_reward_days")
    status     = str(row.get("referral_status", ""))
    txn_id     = row.get("transaction_id")
    txn_status = str(row.get("transaction_status", "")).upper().strip()
    txn_type   = str(row.get("transaction_type",   "")).upper().strip()
    txn_at     = row.get("transaction_at")
    ref_at     = row.get("referral_at")
    exp_date   = row.get("referrer_membership_exp")
    is_deleted = row.get("referrer_is_deleted")
    rewarded   = row.get("is_reward_granted")

    has_reward = pd.notna(reward) and int(reward) > 0
    has_txn    = pd.notna(txn_id)

    # --- VALID CONDITION 1: successful referral with all checks passed ---
    if (
        has_reward
        and status == "Berhasil"
        and has_txn
        and txn_status == "PAID"
        and txn_type == "NEW"
        and pd.notna(txn_at) and pd.notna(ref_at)
        and txn_at > ref_at
        and txn_at.month == ref_at.month
        and txn_at.year  == ref_at.year
        and (pd.isna(exp_date) or drop_tz(exp_date) >= ref_at)
        and is_deleted is False
        and rewarded is True
    ):
        return True, None

    # --- VALID CONDITION 2: pending or failed referrals with no reward ---
    if status in ("Menunggu", "Tidak Berhasil") and not has_reward:
        return True, None

    # --- INVALID CONDITIONS ---

    # C1: reward was given but status is not Berhasil
    if has_reward and status != "Berhasil":
        return False, "INVALID C1: Reward granted but referral status is not Berhasil"

    # C2: reward was given but there's no transaction linked
    if has_reward and not has_txn:
        return False, "INVALID C2: Reward granted but no transaction ID found"

    # C3: there's a valid PAID transaction after the referral but no reward given
    if (
        not has_reward and has_txn and txn_status == "PAID"
        and pd.notna(txn_at) and pd.notna(ref_at) and txn_at > ref_at
    ):
        return False, "INVALID C3: PAID transaction exists after referral but reward was not assigned"

    # C4: status is Berhasil but no reward value
    if status == "Berhasil" and not has_reward:
        return False, "INVALID C4: Status is Berhasil but reward value is null or 0"

    # C5: transaction happened before referral was created
    if has_txn and pd.notna(txn_at) and pd.notna(ref_at) and txn_at < ref_at:
        return False, "INVALID C5: Transaction occurred BEFORE referral creation date"

    return True, None


def run_fraud_detection(df):
    logger.info("Running fraud detection...")

    results = df.apply(check_business_logic, axis=1)
    df["is_business_logic_valid"] = results.apply(lambda x: x[0])
    df["invalid_reason"]          = results.apply(lambda x: x[1])

    valid_count   = df["is_business_logic_valid"].sum()
    invalid_count = (~df["is_business_logic_valid"]).sum()

    logger.info(f"Results: {valid_count} valid, {invalid_count} flagged")
    if invalid_count > 0:
        logger.warning("Flagged cases:")
        for reason, count in df["invalid_reason"].value_counts().items():
            logger.warning(f"  {count}x -> {reason}")

    return df


# ---------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------

REPORT_COLUMNS = [
    "referral_details_id",
    "referral_id",
    "referral_source",
    "referral_source_category",
    "referral_at",
    "referrer_id",
    "referrer_name",
    "referrer_phone_number",
    "referrer_homeclub",
    "referee_id",
    "referee_name",
    "referee_phone",
    "referral_status",
    "num_reward_days",
    "transaction_id",
    "transaction_status",
    "transaction_at",
    "transaction_location",
    "transaction_type",
    "updated_at",
    "reward_granted_at",
    "is_business_logic_valid",
    "invalid_reason",
]

def save_report(df):
    # select only the columns we need
    final = df[[c for c in REPORT_COLUMNS if c in df.columns]].copy()

    # format datetime columns to string
    dt_cols = ["referral_at", "transaction_at", "updated_at", "reward_granted_at"]
    for col in dt_cols:
        if col in final.columns:
            final[col] = final[col].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
            )

    out = os.path.join(OUTPUT_DIR, "final_report.csv")
    final.to_csv(out, index=False)
    logger.info(f"Report saved: {out} ({len(final)} rows)")
    return out


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    logger.info("=== Springer Capital Referral Pipeline ===")

    # step 1: load
    tables = load_data()
    (lead_logs, user_referrals, user_referral_logs,
     user_logs, user_referral_statuses, referral_rewards, paid_transactions) = tables

    # step 2: profile all source tables
    logger.info("Profiling tables...")
    profile_dataframe(lead_logs,              "lead_logs")
    profile_dataframe(user_referrals,         "user_referrals")
    profile_dataframe(user_referral_logs,     "user_referral_logs")
    profile_dataframe(user_logs,              "user_logs")
    profile_dataframe(user_referral_statuses, "user_referral_statuses")
    profile_dataframe(referral_rewards,       "referral_rewards")
    profile_dataframe(paid_transactions,      "paid_transactions")

    # step 3: clean
    tables = clean_data(*tables)

    # step 4: process and join
    df = process_and_join(*tables)

    # step 5: fraud detection
    df = run_fraud_detection(df)

    # step 6: save output
    save_report(df)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
