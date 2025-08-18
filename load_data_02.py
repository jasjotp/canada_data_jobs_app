import os 
import psycopg2
import json 
from psycopg2.extras import execute_values, Json

dsn = os.getenv("PG_DSN")
print(f"DSN: {dsn}")

if not dsn:
    raise ValueError("Set PG_DSN, example postgresql://postgres:NewStrongPassword123@localhost:5432/jobs")

INPUT_PATH = "linkedin_jobs.json"

with open(INPUT_PATH, "r", encoding = "utf-8") as f:
    data = json.load(f)

# sanity check: try to connect to PG SQL
try:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select 1")
            print("Connected to PG")
except Exception as e:
    print("Failed: ", e)

# function to get each column that we want to insert into the database
def to_row(record):
    base_salary = record.get("base_salary") or {}
    return (
        record.get("job_posting_id"),
        record.get("url"),
        record.get("job_title"),
        record.get("company_name"),
        record.get("company_id"),
        record.get("job_location"),
        record.get("job_summary"),
        record.get("job_seniority_level"),
        record.get("job_function"),
        record.get("job_employment_type"),
        record.get("job_industries"),
        record.get("company_url"),
        record.get("job_posted_time"),
        record.get("job_num_applicants"),
        record.get("apply_link"),
        record.get("country_code"),
        record.get("title_id"),
        record.get("job_posted_date"),
        Json(record.get("job_poster")),
        record.get("job_description_formatted"),
        base_salary.get("min_amount"),
        base_salary.get("max_amount"),
        base_salary.get("currency"),
        base_salary.get("payment_period"),
        record.get("timestamp")
    )

rows = [to_row(record) for record in data]

sql = """
insert into linkedin_jobs (
  job_posting_id, url, job_title, company_name, company_id, job_location,
  job_summary, job_seniority_level, job_function, job_employment_type,
  job_industries, company_url, job_posted_time, job_num_applicants,
  apply_link, country_code, title_id, job_posted_date, job_poster,
  job_description_formatted, base_salary_min_amount, base_salary_max_amount,
  base_salary_currency, base_salary_payment_period, snapshot_ts
) values %s
on conflict (job_posting_id) do update set
  url = excluded.url,
  job_title = excluded.job_title,
  company_name = excluded.company_name,
  company_id = excluded.company_id,
  job_location = excluded.job_location,
  job_summary = excluded.job_summary,
  job_seniority_level = excluded.job_seniority_level,
  job_function = excluded.job_function,
  job_employment_type = excluded.job_employment_type,
  job_industries = excluded.job_industries,
  company_url = excluded.company_url,
  job_posted_time = excluded.job_posted_time,
  job_num_applicants = excluded.job_num_applicants,
  apply_link = excluded.apply_link,
  country_code = excluded.country_code,
  title_id = excluded.title_id,
  job_posted_date = excluded.job_posted_date,
  job_poster = excluded.job_poster,
  job_description_formatted = excluded.job_description_formatted,
  base_salary_min_amount = excluded.base_salary_min_amount,
  base_salary_max_amount = excluded.base_salary_max_amount,
  base_salary_currency = excluded.base_salary_currency,
  base_salary_payment_period = excluded.base_salary_payment_period,
  snapshot_ts = excluded.snapshot_ts
"""

# try to commit the JSON data to the postgressql database 
try: 
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size = 500)
        conn.commit()
    print(f"Upserted {len(rows)} rows into jobs database")
except Exception as e:
    print(f"Error inserting rows into jobs database: {e}")