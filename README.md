# SAR Narrative Generator with Audit Trail

## Problem Statement

Banks are required to file Suspicious Activity Reports (SARs) with FinCEN whenever they detect potential money laundering, fraud, or financial crime. Each SAR narrative takes compliance officers **5-6 hours to draft manually**, and large institutions produce thousands annually. With regulatory expectations rising, poorly written SARs lead to remediation demands or enforcement actions. Compliance teams are understaffed, creating operational bottlenecks and growing backlogs.

**Our solution** is an AI system that generates complete, regulatory-compliant SAR narratives in under 60 seconds with a full audit trail explaining every decision -- reducing manual effort by 98% while maintaining transparency and defensibility.

---

## System Architecture

```
                              +---------------------------+
                              |    Streamlit Dashboard    |
                              |  (Alert Dashboard, SAR   |
                              |  Generator, Audit Trail) |
                              +------------+--------------+
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
          +---------v--------+   +---------v--------+   +---------v--------+
          | Alert Monitoring |   | SAR Generation   |   | Audit Trail      |
          | & Risk Scoring   |   | Engine           |   | Viewer           |
          +--------+---------+   +--------+---------+   +---------+--------+
                   |                      |                       |
         +---------v--------+   +---------v--------+    +---------v-------+
         | XGBoost + SHAP   |   | Mistral 7B LLM   |    | JSON Audit Logs |
         | Risk Classifier  |   | (via Ollama)      |    | (Per-SAR)       |
         +--------+---------+   +--------+---------+    +-----------------+
                   |                      |
                   |             +---------v--------+
                   |             | ChromaDB RAG     |
                   |             | (nomic-embed-text)|
                   |             | 10 SAR Templates |
                   |             +------------------+
                   |
         +---------v-----------------+
         | SQLite Database           |
         | (customers, transactions, |
         |  alerts)                  |
         +---------------------------+
```

### Data Flow

1. **Transaction data** enters the system (1,000 customers, 125,000+ transactions)
2. **XGBoost classifier** scores each customer's risk based on 30 engineered features
3. **SHAP explainer** identifies the top risk drivers for each alert
4. When an analyst selects an alert, **ChromaDB** retrieves the 2 most relevant SAR narrative templates via semantic similarity
5. **Mistral 7B** generates a complete SAR narrative using the retrieved templates as style guidance and the actual customer/transaction data as content
6. **Compliance validator** checks the output against FinCEN 5W+H requirements
7. The entire process is logged in a **structured audit trail** (JSON)

---

## System Components

### 1. Synthetic Data Generator (`generate_aml_data.py`)

Generates realistic AML transaction data covering 10 money laundering typologies:

| Typology | Pattern | Example Indicators |
|----------|---------|-------------------|
| Structuring | Cash deposits just under $10,000 | 15-25 deposits, $7K-$9.9K range, multiple branches |
| Rapid Movement | Deposit then immediate wire out | Large wire in, 95-99% wired out within 24-72 hours |
| Layering | Complex transfer chains | 5-8 hop chains across 5 countries |
| Trade-Based | Over-invoiced international trade | Invoice values 2-5x fair market value |
| Cash-Intensive Business | Excessive cash for business size | 3-5x industry benchmark deposits |
| Shell Company | High volume, no real operations | $100K-$500K wires, offshore jurisdictions |
| Funnel Account | Many sources to one to distribution | 6 sources, 2 beneficiaries, cycled |
| Third-Party Payments | Unrelated party transactions | 8 third parties, no business relationship |
| Round-Tripping | Funds leave and return | Outgoing wire, returns as "investment income" |
| Smurfing | Coordinated multi-person deposits | 3-5 mules, same-day deposits at different branches |

**Output:** 1,000 customers (100 suspicious), 125,074 transactions, 100 alerts stored in SQLite.

### 2. Alert Classifier (`classify_alerts.py`)

- **Model:** XGBoost gradient-boosted classifier
- **Features:** 30 engineered features across 6 categories:
  - Customer profile (income, account age, business type)
  - Transaction volume (total, average, max, std deviation)
  - Transaction type ratios (deposit/withdrawal ratio)
  - Method features (% cash, % wire, total cash amount)
  - International features (# high-risk country transactions, % international)
  - Behavioral indicators (structuring count, rapid sequence count, volume-to-income ratio)
- **Explainability:** SHAP TreeExplainer generates per-alert feature importance, showing exactly which features drove the risk score
- **Performance:** AUC-ROC of 1.0 on synthetic data

### 3. RAG Pipeline (`setup_rag.py`)

- **Vector Store:** ChromaDB with 10 SAR narrative templates
- **Embedding Model:** `nomic-embed-text` (274MB, runs locally via Ollama)
- **Retrieval:** Semantic similarity search returns top-2 matching templates for any alert typology
- **Purpose:** Provides structural and stylistic guidance to the LLM without hardcoding templates into prompts

### 4. SAR Generation Engine (`generate_sar.py`)

- **LLM:** Mistral 7B (Q4_K_M quantization, 5.1GB VRAM, 32K context window) running locally via Ollama
- **Temperature:** 0.1 (low for factual consistency)
- **Prompt Engineering:** Structured prompt containing:
  - Retrieved SAR template context (from RAG)
  - Formatted customer profile
  - Transaction summary with statistics and top-20 transactions
  - Alert details with SHAP risk drivers
  - Explicit 5W+H structure requirements
- **Compliance Validation:** Automated check for all 8 required sections (Introduction, Who, What, When, Where, Why Suspicious, How, Conclusion), word count, specific dollar amounts, and specific dates
- **Audit Trail:** JSON document recording data lineage, model parameters, retrieved templates, SHAP features, generation timestamp, and human review status

### 5. Streamlit Dashboard (`app.py`)

Three interactive pages:

- **Alert Dashboard:** Real-time metrics (total alerts, severity breakdown, pending review count), risk score distribution histogram, typology breakdown bar chart, volume by typology, and top-20 high-risk alerts table
- **Generate SAR:** Alert selector with preview, one-click generation, live compliance validation, editable narrative text area, SHAP risk driver visualization, downloadable outputs (.txt and .json)
- **Audit Trail Viewer:** Complete data lineage, model decision transparency, template retrieval records, SHAP feature breakdown, compliance validation results, and raw JSON audit trail

---

## Methodology

### Data Pipeline

The system follows a sequential pipeline architecture:

```
Raw Data -> Feature Engineering -> Risk Classification -> Template Retrieval -> Narrative Generation -> Compliance Validation -> Audit Trail
```

Each stage is independent and can be re-run without affecting others. The SQLite database serves as the shared state between stages.

### RAG (Retrieval-Augmented Generation) Approach

Rather than fine-tuning the LLM or hardcoding templates, we use RAG to dynamically retrieve relevant SAR templates based on the detected typology. This approach:

- Allows adding new templates without retraining
- Ensures the LLM follows established regulatory writing style
- Prevents hallucination by grounding generation in real data and proven templates
- Makes template selection transparent in the audit trail

### Explainability with SHAP

Every risk score is accompanied by SHAP (SHapley Additive exPlanations) values that quantify each feature's contribution to the decision. This directly addresses the regulatory requirement that AI decisions must be explainable -- the audit trail shows not just *that* something is suspicious, but *why* the model flagged it.

### Scalability

- **Horizontal Scaling:** The system uses SQLite for the POC but the architecture supports PostgreSQL or any SQL database. Multiple instances can read from the same data source
- **Stateless Generation:** Each SAR generation is independent -- no shared state between requests. Multiple analysts can generate SARs concurrently
- **Model Serving:** Ollama natively supports concurrent requests, so the LLM can serve multiple generation requests without queuing
- **Vector Store:** ChromaDB supports persistent storage and can be replaced with Weaviate or Milvus for production-scale deployments

### Performance

| Metric | Value |
|--------|-------|
| SAR generation time | 30-60 seconds |
| Data loading | < 2 seconds |
| Risk scoring (per alert) | < 100ms |
| Template retrieval | < 500ms |
| Compliance validation | < 10ms |
| Manual SAR writing time | 5-6 hours |
| **Time savings** | **~98%** |

### Security

- **Zero Data Exfiltration:** The entire system runs locally. Mistral 7B runs on-device via Ollama -- no API calls to external LLM providers, no data leaves the environment
- **On-Premises Compatible:** Designed for air-gapped deployment. No internet required after initial model download
- **Data Isolation:** Customer, transaction, and alert data are stored in a local SQLite database. No cross-domain data leakage
- **LLM Output Constraints:** Temperature set to 0.1 to minimize hallucination. Prompt explicitly instructs the model to use only provided data. Compliance validator rejects outputs that don't meet structural requirements
- **Unbiased Generation:** System prompt instructs the LLM to remain factual and unbiased, limiting output scope to on-topic SAR content only
- **Role-Based Access:** The Streamlit dashboard architecture supports adding authentication middleware (e.g., Streamlit Authenticator) for role-based access control in production

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Mistral 7B (Q4_K_M via Ollama) | SAR narrative generation |
| Embedding Model | nomic-embed-text (via Ollama) | Template vectorization for RAG |
| Vector Database | ChromaDB | Semantic template retrieval |
| ML Classifier | XGBoost | Risk scoring |
| Explainability | SHAP (TreeExplainer) | Feature importance for audit trail |
| Orchestration | LangChain | LLM and RAG pipeline management |
| Database | SQLite | Customer, transaction, alert storage |
| Frontend | Streamlit | Interactive dashboard |
| Visualization | Plotly | Charts and data visualization |
| Data Generation | Faker, NumPy, Pandas | Synthetic AML data |

### Why These Choices

- **Mistral 7B over larger models:** Fits in 8GB VRAM with headroom, 32K context handles full prompts, produces structured regulatory text well at low temperature
- **ChromaDB over Weaviate/Milvus:** Lightweight, embedded, zero-config. Perfect for POC. Easily swappable for production
- **XGBoost over neural networks:** Interpretable, fast training, excellent with tabular data, native SHAP integration
- **SQLite over PostgreSQL:** Zero setup for POC. The code uses standard SQL queries and Pandas `read_sql` -- switching to PostgreSQL requires only changing the connection string
- **Local LLM over API-based:** Meets banking security requirements. No data leaves the environment. No API costs or rate limits

---

## Project Structure

```
PS5/
+-- app.py                        # Streamlit dashboard (Phase 6)
+-- generate_aml_data.py          # Synthetic data generation (Phase 2)
+-- classify_alerts.py            # XGBoost + SHAP classifier (Phase 3)
+-- setup_rag.py                  # ChromaDB RAG pipeline (Phase 4)
+-- generate_sar.py               # SAR generation engine (Phase 5)
+-- collect_sar_templates.py      # SAR template collection (Phase 1)
+-- EDA.py                        # Exploratory data analysis
+-- vramcheck.py                  # GPU/VRAM verification
+--
+-- aml_data.db                   # SQLite database
+-- alert_features_scored.csv     # Features with risk scores
+-- risk_classifier.pkl           # Trained XGBoost model
+-- shap_explainer.pkl            # SHAP explainer
+-- feature_columns.pkl           # Feature column list
+--
+-- data/
|   +-- sar_templates.json        # 10 SAR narrative templates
|   +-- generated/                # Generated CSVs
|   +-- ibm_aml/                  # IBM AMLSim dataset
|   +-- paysim/                   # PaySim dataset
|   +-- synthetic_aml/            # SAML-D dataset
+--
+-- chroma_db/                    # ChromaDB vector store
+-- outputs/                      # Generated SAR files
|   +-- sar_alert_*.json          # SAR with audit trail
|   +-- sar_narrative_*.txt       # Plain text narratives
+-- models/                       # Saved ML models
```

---

## Future Scope

### Short-Term Enhancements
- **Multi-language support:** Generate SARs in different regulatory jurisdictions (UK SARs, EU STRs) by adding region-specific templates to ChromaDB
- **Batch generation:** Generate SARs for all pending alerts in one click with progress tracking
- **Analyst feedback loop:** Allow analysts to rate generated narratives, feeding quality scores back to improve prompt engineering
- **PDF export:** Generate formatted PDF SARs matching FinCEN form layout

### Medium-Term Improvements
- **Fine-tuned model:** Fine-tune Mistral on a corpus of real anonymized SARs using LoRA/QLoRA for domain-specific vocabulary and tone
- **Active learning:** Use analyst edits to continuously improve generation quality
- **Real data connectors:** Integrate with actual transaction monitoring systems (Actimize, Mantas, Fircosoft) via API adapters
- **PostgreSQL migration:** Replace SQLite with PostgreSQL for multi-user concurrent access
- **Authentication:** Add role-based access control with Streamlit Authenticator or OAuth integration

### Long-Term Vision
- **Multi-model ensemble:** Use specialized models for different typologies (e.g., a trade-finance-specific model for TBML SARs)
- **Regulatory update pipeline:** Automatically ingest new FinCEN guidance documents into ChromaDB to keep templates current
- **Cross-institution patterns:** Federated learning across institutions to detect multi-bank laundering networks without sharing raw data
- **Cloud deployment:** Containerized deployment on AWS/Azure/GCP with auto-scaling, using Amazon Bedrock or Azure OpenAI for customers preferring managed LLM services
- **Real-time streaming:** Process transaction streams in real-time using Apache Kafka for instant alert generation

---

## How to Run

### Prerequisites
- Python 3.10+
- Ollama installed ([ollama.com](https://ollama.com))
- GPU with 8GB+ VRAM (recommended) or CPU (slower)

### Setup
```bash
# Install dependencies
pip install pandas numpy faker chromadb langchain langchain-community langchain-ollama xgboost shap streamlit plotly joblib scikit-learn

# Pull models
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### Run Pipeline
```bash
# Phase 1: Collect SAR templates
python collect_sar_templates.py

# Phase 2: Generate synthetic data (~2 minutes)
python generate_aml_data.py

# Phase 3: Train classifier (~1 minute)
python classify_alerts.py

# Phase 4: Setup RAG pipeline (~30 seconds)
python setup_rag.py

# Phase 5: Test SAR generation (~60 seconds)
python generate_sar.py

# Phase 6: Launch dashboard
streamlit run app.py
```

Dashboard will be available at **http://localhost:8501**

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Customers simulated | 1,000 |
| Suspicious customers | 100 (10%) |
| Transactions generated | 125,074 |
| Money laundering typologies | 10 |
| SAR templates in RAG | 10 |
| XGBoost features | 30 |
| AUC-ROC score | 1.00 |
| SAR sections validated | 8 (5W+H) |
| Average narrative length | 350-500 words |
| Generation time per SAR | 30-60 seconds |
| Manual writing time | 5-6 hours |
| Time savings | ~98% |
| LLM VRAM usage | 5.1 GB |
| Data exfiltration risk | Zero (fully local) |
