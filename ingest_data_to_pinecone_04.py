import os 
import json 
import math 
import time 
import psycopg2
import tiktoken 
from bs4 import BeautifulSoup
from dotenv import load_dotenv 
from pinecone import Pinecone 
from openai import OpenAI 

load_dotenv()

# set the environment variables
PG_DSN = os.getenv("PG_DSN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize the Pinecone instance and index, and the OpenAI instnance 
pinecone = Pinecone(
    api_key = PINECONE_API_KEY
)

index = pinecone.Index(PINECONE_INDEX)
openai = OpenAI(
    api_key = OPENAI_API_KEY
)

enc = tiktoken.get_encoding("cl100k_base")

# helper function to strip the MTML content and extract all text contennt from the HTML
def strip_html(html):
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator = "\n").strip()

# function to build a formatted document before encoding the text
def build_doc(row):
    # row is a tuple from SQL, so the fields in each row should be in sync with the query below
    (
        job_posting_id, url, job_title, company_name, company_id, job_location,
        job_summary, job_seniority_level, job_function, job_employment_type,
        job_industries, company_url, job_posted_time, job_num_applicants,
        apply_link, country_code, title_id, job_posted_date, job_poster_json,
        job_description_formatted, base_min, base_max, base_currency,
        base_period, snapshot_ts
    ) = row 

    # get the text out of the job description
    job_desc = strip_html(job_description_formatted)
    poster = ""

    # extract the name and title from the jos poster field (which contains url, name, and title in JSON format)
    try:
        if job_poster_json:
            jp = json.loads(job_poster_json) if isinstance(job_poster_json, str) else job_poster_json
            name = jp.get("name")
            title = jp.get("title")
            poster = f"Posted by {name}, {title}" if name or title else ""
    except Exception:
        poster = ""

    # format the base pay fields as text
    base_pay = ""
    if base_min or base_max or base_currency or base_period:
        base_pay = f"Salary range. {base_min} to {base_max} {base_currency} per {base_period}"
    
    # only keep the content that helps match questions, as including things like ID can add noise and do not add any semantic value
    text = "\n".join([ # onky keep columns that are valuable for for semantic search
        f"Title. {job_title or ''}",
        f"Company. {company_name or ''}",
        f"Location. {job_location or ''}",
        f"Seniority. {job_seniority_level or ''}",
        f"Function. {job_function or ''}",
        f"Type. {job_employment_type or ''}",
        f"Industry. {job_industries or ''}",
        f"Posted. {job_posted_date or job_posted_time or ''}",
        base_pay, 
        poster, 
        "Summary.",
        job_summary,
        "Description.",
        job_desc, 
        f"Apply. {apply_link or url or ''}"
    ])
    text = "\n".join([line for line in text.splitlines() if line.strip()])
    return text

# function to chunk the text into vectors of 800 tokens each: max tokens are set to 800 for now 
def chunk_text(text, max_tokens = 800, overlap = 80):
    tokens = enc.encode(text)
    out = []
    step = max_tokens - overlap 
    for i in range(0, len(tokens), step): # sample random chunks of text out of our data
        chunk_ids = tokens[i:i + max_tokens] # get the chunk for each step
        out.append(enc.decode(chunk_ids))
    return out 

# function to embed the text data 
def embed_texts(texts):
    response = openai.embeddings.create(
        model = "text-embedding-3-small",
        input = texts
    )
    return [d.embedding for d in response.data]

def fetch_rows(batch=1000):
    sql = """
    select
      job_posting_id, url, job_title, company_name, company_id, job_location,
      job_summary, job_seniority_level, job_function, job_employment_type,
      job_industries, company_url, job_posted_time, job_num_applicants,
      apply_link, country_code, title_id, job_posted_date, job_poster,
      job_description_formatted, base_salary_min_amount, base_salary_max_amount,
      base_salary_currency, base_salary_payment_period, snapshot_ts
    from linkedin_jobs
    order by job_posted_date desc nulls last, job_posted_time desc nulls last
    """

    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor(name = "jobs_stream") as cur:
            cur.itersize = batch 
            cur.execute(sql)
            for row in cur:
                yield row 

# upsert the vector embeddings for each row into our vector DB
def upsert_jobs():
    buffer = []
    count = 0 
    for row in fetch_rows():
        job_id = row[0]
        doc = build_doc(row) # text to embed (only chose text that has semantic meaning)
        chunks = chunk_text(doc)

        for idx, chunk in enumerate(chunks):
            vector_id = f"{job_id}_{idx}"
            metadata = {
                "job_posting_id": job_id,
                "chunk_id": idx,
                "title": row[2],
                "company": row[3],
                "location": row[5],
                "url": row[1],
                "apply_link": row[14] or row[1],
                "posted_date": str(row[17] or row[12] or ""),
                "text": chunk
            }
            buffer.append((vector_id, metadata))
        
        # embed the text and upsert 64 vectors at a time into pinecone for performance
        while len(buffer) >= 64:
            to_embed = buffer[:64]
            buffer = buffer[64:]

            texts = [m[1]["text"] for m in to_embed]
            embeddings = embed_texts(texts)
            vectors = []
            
            for pair, embedding in zip(to_embed, embeddings):
                vid, metadata = pair
                vectors.append({
                    "id": vid, 
                    "values": embedding,
                    "metadata": metadata
                })

            index.upsert(vectors = vectors)
            count += len(vectors)
            print(f"Upserted {count} vectors")
    
    # flush any remainder 
    if buffer:
        texts = [m[1]['text'] for m in buffer]
        embeddings = embed_texts(texts)
        vectors = []
        
        for pair, embedding in zip(buffer, embeddings):
            vid, metadata = pair
            vectors.append({
                "id": vid, 
                "values": embedding,
                "metadata": metadata
            })

        index.upsert(vectors = vectors)
        count += len(vectors)
        print(f"Upserted {count} vectors")

if __name__ == "__main__":
    upsert_jobs()
    print("Done upserting")