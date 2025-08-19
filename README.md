# TrueNorthJobs

A job search platform that leverages AI-powered semantic search to help data and AI professionals find relevant opportunities across Canada. Built with modern vector databases and large language models for intelligent job matching.

Click the Landing Page below to try it.

[![True North Jobs AI Landing Page](landing_page.png)](https://www.truenorthjobsai.ca/)

Or open the site directly: https://www.truenorthjobsai.ca/

## Demo Video

![Demo](truenorthjobs_demo.gif)
---

## Overview

This application combines web scraping, vector embeddings, and conversational AI to create an intelligent job search experience. It automatically extracts job postings from LinkedIn, processes them through embeddings, and provides natural language search capabilities with contextual answers.

### Key Features

- **Semantic Search**: Natural language queries that understand intent, not just keywords
- **AI-Powered Responses**: Contextual answers with job recommendations and insights
- **Advanced Filtering**: Role-based and location-based filtering for precise results
- **Intelligent Ranking**: MMR (Maximal Marginal Relevance) reranking for diverse, relevant results
- **Export Functionality**: Download search results as CSV for further analysis
- **Fresh Data**: Daily refresh scripts to pull the latest postings

---

## Usage

1. **Open the site [https://www.truenorthjobsai.ca/](https://www.truenorthjobsai.ca/)**

2. **Enter your query in the text box for example “data scientist Toronto with PySpark” or “entry level machine learning engineer Vancouver**

3. **Pick filters such as Role or Location then click Search**

4. **Wait for results, review the answer with sources and use the Apply link or download sources as CSV**

--- 

## Architecture

### Data Pipeline
1. **Web Scraping** (`extract_jobs_data_01.py`): Automated LinkedIn job extraction via BrightData API
2. **Data Storage** (`load_data_02.py`): PostgreSQL database for storing structured job data
3. **Vector Indexing** (`create_index_03.py`): Pinecone vector database setup
4. **Embedding Pipeline** (`ingest_data_to_pinecone_04.py`): Text processing and creating vector embeddings in Pinecone
5. **Search Interface** (`app_05.py`): Streamlit-powered web application

---

### Technology Stack

**Backend Infrastructure**
- PostgreSQL for relational data storage
- Pinecone for vector similarity search
- OpenAI GPT-4o-mini for response generation
- OpenAI text-embedding-3-small for semantic embeddings

**Data Processing**
- BeautifulSoup for HTML parsing
- tiktoken for token management
- NumPy for mathematical operations
- psycopg2 for PostgreSQL connectivity

**Frontend**
- Streamlit for interactive web interface
- Real-time search with caching optimization
- Responsive design with filtering capabilities

---

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database
- API keys for OpenAI, Pinecone, and BrightData

### Environment Setup

```bash
git clone [repository-url]
cd canada-data-jobs-app
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Database Configuration
PG_DSN=postgresql://username:password@localhost:5432/jobs

# API Keys
BRIGHTDATA_API_KEY=your_brightdata_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
OPENAI_API_KEY=your_openai_key
```

### Database Setup

```sql
CREATE TABLE linkedin_jobs (
    job_posting_id VARCHAR PRIMARY KEY,
    url TEXT,
    job_title TEXT,
    company_name TEXT,
    company_id TEXT,
    job_location TEXT,
    job_summary TEXT,
    job_seniority_level TEXT,
    job_function TEXT,
    job_employment_type TEXT,
    job_industries TEXT,
    company_url TEXT,
    job_posted_time TEXT,
    job_num_applicants INTEGER,
    apply_link TEXT,
    country_code TEXT,
    title_id TEXT,
    job_posted_date DATE,
    job_poster JSONB,
    job_description_formatted TEXT,
    base_salary_min_amount DECIMAL,
    base_salary_max_amount DECIMAL,
    base_salary_currency TEXT,
    base_salary_payment_period TEXT,
    snapshot_ts TIMESTAMP
);
```

---

### Data Pipeline Execution

1. **Extract latest job data**:
   ```bash
   python extract_jobs_data_01.py
   ```

2. **Load data into PostgreSQL**:
   ```bash
   python load_data_02.py
   ```

3. **Create Pinecone index**:
   ```bash
   python create_index_03.py
   ```

4. **Generate embeddings and populate vector database**:
   ```bash
   python ingest_data_to_pinecone_04.py
   ```

### Launch Application

```bash
streamlit run app_05.py
```

### Search Examples

- "entry level data engineer roles in Vancouver with salary"
- "machine learning engineer roles with PyTorch experience"
- "senior data scientist positions in Toronto"
- "remote analytics engineer opportunities"

---

## Technical Features

### Intelligent Text Processing
- HTML content sanitization and text extraction
- Token-aware text chunking with overlap for context preservation
- Metadata extraction for enhanced search relevance

### Advanced Search Algorithms
- Cosine similarity for vector matching
- MMR reranking for result diversity
- Client-side filtering for real-time refinement

### Performance Optimizations
- Batch processing for vector operations
- Streamlit caching for repeated queries
- Efficient database streaming for large datasets

---

## API Integration

### BrightData Integration
Automated job extraction targeting:
- Data Engineer positions
- Data Scientist roles
- Machine Learning Engineer opportunities
- Analytics Engineer positions
- AI Engineer roles

### OpenAI Integration
- GPT-4o-mini for conversational responses
- text-embedding-3-small for semantic search
- Temperature-controlled generation for consistency

---

## Future Enhancements

- Multi-vector search capabilities to allow for pasting of resumes in text box
- Airflow DAG for automated daily ingestion of latest jobs in the last 24 hours
- Real-time job alerts and notifications

---

## Contributing

This project demonstrates proficiency in:
- Full-stack application development
- Vector database implementation
- Machine learning pipeline design
- API integration and data processing
- Modern web application deployment

---

## Contact

**Jasjot Parmar**  
Aspiring Data Scientist | Python & Machine Learning Enthusiast | Data Science Graduate Student

🔗 [GitHub](https://github.com/jasjotp) • [LinkedIn](https://www.linkedin.com/in/jasjotparmar)

---

*Built with modern AI technologies to revolutionize the job search experience for data professionals in Canada.*
