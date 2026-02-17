# Hosting Guide - SAR Narrative Generator POC

This document covers how to make your POC accessible via a shareable link for the hackathon submission.

---

## Option 1: Streamlit Community Cloud (Recommended - Free, Easiest)

**The catch:** Streamlit Cloud does not support Ollama/GPU. You would need to swap the LLM call to an API-based model (e.g., Groq free tier running Mistral). This is the most practical option for a shareable link.

### Steps

1. **Push your project to a public GitHub repo**
   ```bash
   cd C:\Users\abhij\Barclay\PS5
   git init
   git add app.py generate_sar.py classify_alerts.py generate_aml_data.py setup_rag.py collect_sar_templates.py
   git add aml_data.db alert_features_scored.csv risk_classifier.pkl shap_explainer.pkl feature_columns.pkl
   git add data/sar_templates.json
   git add requirements.txt README.md
   git commit -m "SAR Narrative Generator POC"
   git remote add origin https://github.com/YOUR_USERNAME/sar-generator.git
   git push -u origin main
   ```

2. **Create a `requirements.txt`**
   ```
   streamlit
   pandas
   numpy
   plotly
   joblib
   scikit-learn
   shap
   xgboost
   chromadb
   langchain
   langchain-community
   langchain-ollama
   groq
   ```

3. **Modify `app.py` for cloud deployment**
   - Replace Ollama LLM calls with Groq API (free tier, runs Mistral)
   - Sign up at https://console.groq.com and get a free API key
   - Use `langchain-groq` instead of `langchain-ollama` for the LLM
   - ChromaDB and embeddings can still run on CPU (slower but works)

4. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app" and select your repo
   - Set `app.py` as the main file
   - Add your Groq API key in the "Secrets" section:
     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     ```
   - Click "Deploy"

5. **You get a URL like:** `https://your-app.streamlit.app`

### Pros
- Free hosting
- Shareable URL
- Auto-deploys on git push
- No server management

### Cons
- Requires swapping Ollama for a cloud LLM API
- Free tier has rate limits
- ChromaDB embeddings will be slower on CPU

---

## Option 2: Ngrok Tunnel (Simplest - Keep Everything Local)

Run the app on your machine and expose it to the internet via ngrok. **No code changes needed.** This is the fastest option if you just need a link during demo day.

### Steps

1. **Install ngrok**
   ```bash
   # Download from https://ngrok.com/download
   # Or via winget:
   winget install ngrok
   ```

2. **Sign up for free at https://ngrok.com** and get your auth token

3. **Configure ngrok**
   ```bash
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```

4. **Start your Streamlit app**
   ```bash
   cd C:\Users\abhij\Barclay\PS5
   streamlit run app.py
   ```

5. **In another terminal, start ngrok**
   ```bash
   ngrok http 8501
   ```

6. **You get a URL like:** `https://abc123.ngrok-free.app`
   - Share this URL with anyone
   - Works as long as your computer is running and online

### Pros
- Zero code changes -- your local Ollama + GPU setup works as-is
- Full performance (GPU-accelerated Mistral)
- Live demo with real LLM generation
- Setup takes 5 minutes

### Cons
- Your computer must stay on and connected to the internet
- URL changes every time you restart ngrok (unless you pay for a fixed subdomain)
- Free tier shows an ngrok interstitial page on first visit (visitor clicks through)
- If your internet drops, the link goes down

---

## Option 3: Cloud VM with GPU (Most Professional, Not Free)

Deploy on a cloud VM with a GPU so everything runs exactly as on your local machine.

### Providers with GPU VMs

| Provider | VM Type | Cost | GPU |
|----------|---------|------|-----|
| Google Cloud | g2-standard-4 | ~$0.70/hr | L4 (24GB) |
| AWS | g4dn.xlarge | ~$0.53/hr | T4 (16GB) |
| Lambda Cloud | gpu_1x_a10 | ~$0.75/hr | A10 (24GB) |
| Vast.ai | Various | ~$0.20/hr | Various |
| RunPod | Various | ~$0.20/hr | Various |

### Steps (using any Linux GPU VM)

1. **Provision a GPU VM** with Ubuntu 22.04

2. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull mistral:7b
   ollama pull nomic-embed-text
   ```

3. **Install Python dependencies**
   ```bash
   pip install pandas numpy faker chromadb langchain langchain-community langchain-ollama xgboost shap streamlit plotly joblib scikit-learn
   ```

4. **Copy your project files** (scp, git clone, etc.)
   ```bash
   scp -r C:\Users\abhij\Barclay\PS5 user@VM_IP:~/sar-generator/
   ```

5. **Run the pipeline**
   ```bash
   cd ~/sar-generator
   python generate_aml_data.py
   python classify_alerts.py
   python setup_rag.py
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

6. **Open the VM's firewall** for port 8501

7. **Access at:** `http://VM_PUBLIC_IP:8501`

### Pros
- Full performance with GPU
- Stable, always-on URL
- Identical to local setup
- Professional for demo

### Cons
- Costs money (~$5-15 for a day of hosting)
- Requires cloud account setup
- Must manage the VM

---

## Recommended Strategy for the Hackathon

**For the submission link:** Use **Option 2 (ngrok)** since it requires zero code changes, gives you full GPU performance for live demos, and takes 5 minutes to set up.

**During the demo:** Run the app locally on your machine with ngrok tunneling. The judges will see real-time Mistral generation on your GPU, which is more impressive than a cloud-hosted version.

**If they need a persistent link:** Use **Option 1 (Streamlit Cloud)** with the Groq API swap. The free Groq tier gives you enough calls for evaluation purposes. The trade-off is you lose the "zero data leaves the environment" security story, but you can note this in the submission and explain that the production deployment runs on-premises.

---

## Quick ngrok Setup (5 minutes)

```bash
# 1. Download ngrok from https://ngrok.com/download and extract it

# 2. Sign up at ngrok.com, copy your auth token

# 3. Set up ngrok
ngrok config add-authtoken YOUR_TOKEN

# 4. Start Streamlit (Terminal 1)
cd C:\Users\abhij\Barclay\PS5
streamlit run app.py

# 5. Start ngrok tunnel (Terminal 2)
ngrok http 8501

# 6. Share the https://xxxxx.ngrok-free.app URL
```

That is it. Your SAR Generator is now accessible to anyone with the link.
