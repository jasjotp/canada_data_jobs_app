import os 
import csv 
import io
import numpy as np 
import streamlit as st 
import tiktoken
from dotenv import load_dotenv 
from pinecone import Pinecone 
from openai import OpenAI 

load_dotenv()
_enc = tiktoken.get_encoding("cl100k_base")

# setup the environment variables and streamlit page 
st.set_page_config(page_title = "Data/AI Jobs Chatbot (Canada)", page_icon = "🧠", layout = "wide")

# set the environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not PINECONE_INDEX or not OPENAI_API_KEY:
    st.error("Missing environment vars. Set PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY in .env")
    st.stop()

# initialize the Pinecone instance and index, and the OpenAI instnance 
pinecone = Pinecone(
    api_key = PINECONE_API_KEY
)

index = pinecone.Index(PINECONE_INDEX)

openai = OpenAI(
    api_key = OPENAI_API_KEY
)

# set the title and caption for the user 
st.title("Ask the Canada Data and AI Jobs Database")
st.caption("Type a question. Pick filters. Get the most relevant response to your question from a database of jobs from all across Canada. The answer will only use the retrieved context.")

# helpeer functions 
# function that embeds the question 
@st.cache_data(show_spinner = False, ttl = 3600)
def embed_text(q: str):
    r = openai.embeddings.create(
        model = "text-embedding-3-small",
        input = q
    )
    return r.data[0].embedding

# function to embed text data 
@st.cache_data(show_spinner = False, ttl = 3600)
def embed_texts(texts):
    r = openai.embeddings.create(
        model = "text-embedding-3-small",
        input = texts
    )

    # texts is a list so go through each chunk and turn it into an embedding 
    return [chunk.embedding for chunk in r.data]

# measures the similary of the 2 embeddings passed in as a and b (closer to 1 means they are a very similar direction, 0 means unrelated and negative means opposite direction)
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0 
    return float(np.dot(a, b) / (an * bn))


def mmr_rerank(query_vec, cand_texts, k = 8, lambda_param = 0.75):
    """MMR (Maximal Marginal Relevance )rerank function with dense embeddings to get relevant and diverse responses.
   Picks the top k items that are relevant to the query and not all the same. 
   Inputs: 
        query_vec: query_embedding 
        cand_texts: the candidate texts 
        k: how many results to return 
        lambda_param: weight for whehter to favour reelvance or diversity in our response. Closer to 1 favours relevance, closer to 0.5 adds more diversity. 
    """
    # if the texts we answer from are empty, return an empty list 
    if not cand_texts:
        return []
    # embed the candidate texts that we answer from 
    cand_embeddings = embed_texts(cand_texts)
    cand_arr = np.array(cand_embeddings, dtype = np.float32)
    query = np.array(query_vec, dtype = np.float32)

    # find the similarity of the query and embedding through using the cosine distance helper function
    relevance = np.array([cosine(query, embedding) for embedding in cand_arr])
    selected = []
    remaining = list(range(len(cand_texts)))

    while len(selected) < min(k, len(remaining)):
        if not selected: # if there are vectors remaining, find the highest similarity embedding and append it to our selectd list of responses
            idx = int(np.argmax(relevance[remaining]))
            selected.append(remaining.pop(idx))
            continue 

        best_score = -1.0 # start with a very low best score so it is overwritten in the first iteration of the loop
        best_remaining_pos = None # keeps track of which remaining candidate is best this round
        
        for pos, cand_id in enumerate(remaining):
            sim_to_most_similar = max(cosine(cand_arr[cand_id], cand_arr[s]) for s in selected)

            # combine relevance to the query and the penalty for being too similar to what you already picked
            score = lambda_param * relevance[cand_id] - (1 - lambda_param) * sim_to_most_similar
            if score > best_score: 
                best_score = score 
                best_remaining_pos = pos
        selected.append(remaining.pop(best_remaining_pos))

    return selected 

# function to condense long inputs like resumes 
@st.cache_data(show_spinner=False, ttl=900)
def condense_query(raw: str) -> str:
    sys = (
        "You create short focused search queries for finding matching job postings."
        "Use role titles, seniority, key skills, locations, etc."
    )

    prompt = (
        "From this text, produce a concise job search query:\n\n"
        f"{raw}\n\n"
        "Return only the query."
    )

    chat = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
        temperature = 0.2,
    )

    query = chat.choices[0].message.content.strip()
    return query 

# function to apply filters to get correct outputs for role and location 
def apply_client_filters(matches, role, location):
    out = matches
    if role:
        role = role.lower().strip()
        out = [match for match in out if role in (match.metadata or {}).get("title", "").lower()]
    if location:
        location = location.lower().strip()
        out = [match for match in out if location in (match.metadata or {}).get("location", "").lower()]
    return out 

# dense search to return the top matches for the user query 
def dense_search(query, fetch_k):
    vec = embed_text(query)
    res = index.query(
        vector = vec,
        top_k = fetch_k, 
        include_metadata = True
    )

    return res.matches or []

# function to build the context of the response 
def build_context(matches, max_tokens = 6000):
    """
    Build context for the response. Keep one chunk per job sp we do not have repeats
    """
    lines = []
    headers = []
    used = 0 
    seen = set()
    counter = 1 

    for match in matches: 
        match_metadata = match.metadata or {}
        job_id = match_metadata.get("job_posting_id")
        
        # if the job id is in seen, skip that match as we do not want repeats 
        if job_id and job_id in seen: 
            continue 
        seen.add(job_id) # else, the job id is not in seen, do add the job to our set as we have not seen that job before

        title = match_metadata.get("title", "")
        company = match_metadata.get("company", "")
        location = match_metadata.get("location", "")
        url = match_metadata.get("apply_link") or match_metadata.get("url")
        text = match_metadata.get("text", "")
        
        header = f"[{counter}] {title} at {company}, {location}. Apply. {url}\n"
        block = header + text + "\n\n"
        
        tokens = _enc.encode(block)
        need = len(tokens)

        # if the current block of tokens are greater than the max token limit, truncate the block of tokens
        if used + need > max_tokens:
            remain = max_tokens - used 
            if remain > 0:
                # fit a truncated slice of this block 
                lines.append(_enc.decode(tokens[:remain]))
                headers.append(header.strip())
            break

        lines.append(block)
        headers.append(header.strip())
        used += need # update the token budget
        counter += 1 

    return "".join(lines), headers

# function to answer the users query using OpenAI API 
def answer(query, context):
    system = (
        "You answer questions about jobs using only the provided context and sources. "
        "Be clear and concise. Use short bullets for lists."
        "When you return jobs, answer with at least the job title, job location, company name, summary of the position, experience level, apply link, and salary"
        "At the end include ALL of the numbered sources you used."
    )
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"

    chat = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{
                "role": "system", 
                "content": system
            }, 
            {
                "role": "user",
                "content": prompt
        }],
        temperature = 0.2 
    )
    return chat.choices[0].message.content

# writes rows that are matches to a csv file
def to_csv(matches):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "company", "location", "apply_link", "url", "posted_date", "score"])
    for m in matches:
        md = m.metadata or {}
        writer.writerow([
            md.get("title", ""),
            md.get("company", ""),
            md.get("location", ""),
            md.get("apply_link") or "",
            md.get("url", ""),
            md.get("posted_date", ""),
            f"{m.score:.3f}" if hasattr(m, "score") else "",
        ])
    return buf.getvalue().encode("utf-8")

# sidebar for the streamlit dashboard 
with st.sidebar:
    st.header("Filters")
    role = st.selectbox("Role", ["", "Data Engineer", "Data Scientist", "Machine Learning Engineer"])
    location = st.text_input("Location contains")
    top_k = st.slider("Results to show", 3, 20, 8, step = 1)

    st.divider()
    st.header("Search options")
    overfetch_k = st.slider("Overfetch size\n(how many results to pull when returning answer)\n", 20, 100, 30, step = 5)
    
    # place the use MMR checkbox with a help tooktip in the sidebar
    use_mmr = st.checkbox(
        "Rerank with MMR (Improve variety)", 
        value = True, 
        help = "Reduces near duplicates by choosing results that are relevant and diverse. Best for broad queries."
        )

# examples for users to pkug in to the searchbar form 
examples = [
    "entry level data engineer roles in Vancouver with salary",
    "machine learning engineer roles with PyTorch",
    "data scientist roles in Toronto",
    "entry level and associate level data roles"
]

# single source for the input 
st.session_state.setdefault("q", "")

# apply chip click from previous run before the widget is created
pending = st.session_state.pop("q_pending", None)
if pending is not None:
    st.session_state["q"] = pending

# set height to auto size for the text box
base = 160
per_line = 22
max_h = 520
curr = st.session_state.get("q", "")
textbox_height = min(base + per_line * max(6, curr.count("\n") + 1), max_h)

with st.form("search"):
    q_col1, q_col2 = st.columns([4, 1])
    with q_col1:
        user_q = st.text_area(
            "Ask about roles, skills, salary, location, or application links",
            key = "q",
            placeholder = "Example. entry level data engineer roles in Vancouver with salary",
            height = textbox_height
        )
    with q_col2:
        submitted = st.form_submit_button("Search", use_container_width = True)

st.caption("Tips. Include a role and a location for stronger results. Add tools or skills like PySpark or Databricks if needed.")
st.warning("Heads up. This version uses dense search only. Short queries work best. Long paste in inputs like full resumes may return weaker results. Multi vector search is coming soon.")

# display clickable example queries for the examples we have above
example_cols = st.columns(len(examples))

for i, example in enumerate(examples):
    if example_cols[i].button(example, key = f"example_{i}"):
        st.session_state["q_pending"] = example
        st.session_state["auto_submit"] = True
        st.rerun()

# run the example if clicked 
auto = st.session_state.pop("auto_submit", False)
query = st.session_state.get("q", "").strip()

# run the dense search and find the matches, before choosing the final set of results to return 
if (submitted or auto) and query.strip():
    with st.spinner("Searching..."):
        try:
            fetch_k = max(top_k, overfetch_k)
            matches = dense_search(query, fetch_k = fetch_k)
            matches = apply_client_filters(matches, role, location)
        except Exception as e:
            st.error(f"Search failed. {e}")
            matches = []

    if not matches: 
        st.info("No results found")
    else: 
        # choose the final set of results 
        overfetched = matches[:max(top_k, overfetch_k)]
        if use_mmr:
            texts = [(m.metadata or {}).get("text", "") for m in overfetched]
            pairs = [(i, text) for i, text in enumerate(texts) if text.strip()]

            if pairs: 
                idxs, only_texts = zip(*pairs)
                selected = mmr_rerank(embed_text(query), list(only_texts), k = top_k, lambda_param = 0.75)
                final = [overfetched[idxs[i]] for i in selected]
            else:
                final = overfetched[:top_k]
        else:
            final = overfetched[:top_k]

        # build the answer 
        context, headers = build_context(final)
        response = answer(query, context)
        
        # answer card 
        st.subheader("Answer")
        st.write(response)

        # sources grid 
        st.subheader("Sources & Similar Results")
        cols = st.columns(2)

        for i, match in enumerate(final, 1):
            md = match.metadata or {}

            with cols[(i - 1) % 2]:
                st.markdown(f"**{i}. {md.get('title','')}**")
                st.caption(f"{md.get('company','')} • {md.get('location','')}")
                link = md.get("apply_link") or md.get("url", "")
                if link:
                    st.markdown(f"[Apply link]({link})")
                st.caption(f"score {match.score:.3f}")

                with st.expander("Preview"):
                    st.write(md.get("text", "")[:2000])

        # export to csv function
        csv_bytes = to_csv(final)
        st.download_button("Download sources as CSV", data=csv_bytes, file_name="job_sources.csv", mime="text/csv")
        st.caption("Dense retrieval with overfetch and optional MMR rerank. Client filters for role and location.")