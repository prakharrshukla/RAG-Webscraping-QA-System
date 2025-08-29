# 🤖 RAG Webscraping Q&A System

A complete **Retrieval-Augmented Generation (RAG)** system that scrapes websites and answers questions using local LLM inference. Built with **Streamlit**, **Ollama**, **ChromaDB**, and **LangChain**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 **What This Does**

1. **Scrape any website** - Advanced bot detection evasion 
2. **Process & chunk text** - Intelligent content extraction
3. **Generate embeddings** - Semantic search ready
4. **Store in vector DB** - ChromaDB for fast retrieval
5. **Answer questions** - Local Ollama LLM inference
6. **Beautiful UI** - Professional Streamlit interface

## 🚀 **Quick Start**

### **Option 1: Streamlit App (Recommended)**
```bash
git clone https://github.com/yourusername/RAG-Webscraping-QA-System.git
cd RAG-Webscraping-QA-System
pip install -r requirements.txt
streamlit run streamlit_rag_app.py --server.port 8875
```
Access at: `http://localhost:8875`

### **Option 2: Jupyter Notebook**
```bash
git clone https://github.com/yourusername/RAG-Webscraping-QA-System.git
cd RAG-Webscraping-QA-System
pip install -r requirements.txt
jupyter notebook RAG_Webscraping_Integration.ipynb
```

## 💻 **Hardware Requirements**

### **Recommended Configuration:**
- **GPU**: NVIDIA RTX 4070 (8GB VRAM)
- **RAM**: 32GB DDR4/DDR5  
- **CPU**: Intel i7-13620H or equivalent
- **Storage**: 10GB free space

### **Why These Specs?**
- **RTX 4070**: Perfect for Ollama `llama3.2:1b` (uses ~3GB VRAM)
- **32GB RAM**: Handle large documents + embeddings smoothly
- **Fast CPU**: Efficient web scraping and text processing

### **Alternative Configurations:**
- **Minimum**: 16GB RAM + modern CPU (slower, CPU-only)
- **Budget GPU**: RTX 3060 12GB works great too
- **High-end**: RTX 4080/4090 enables larger models (7b, 13b)

## 🛠️ **Installation & Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/RAG-Webscraping-QA-System.git
cd RAG-Webscraping-QA-System
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Install Ollama**
**Windows:**
1. Download from [ollama.com](https://ollama.com)
2. Install and start Ollama service
3. Pull the model: `ollama pull llama3.2:1b`

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### **4. Run Application**
```bash
streamlit run streamlit_rag_app.py --server.port 8875
```

## 🌐 **Supported Websites**

### **✅ Works Great:**
- **News Sites**: BBC, Reuters, CNN, TechCrunch
- **Wikipedia**: Any topic or article
- **Blogs**: Medium, personal blogs, technical articles
- **Documentation**: Project docs, tutorials

### **❌ Won't Work:**
- **Social Media**: Twitter, Instagram, Facebook
- **Video Platforms**: YouTube, TikTok
- **Login Required**: Private sites, paywalled content
- **Heavy JavaScript**: SPAs without server-side rendering

## 🏗️ **Technical Architecture**

```
Website URL → Web Scraper → Text Processing → Chunking
                                                ↓
User Question → RAG Pipeline ← Vector Search ← ChromaDB ← Embeddings
                ↓
            Ollama LLM → Answer
```

### **Key Components:**
- **Web Scraping**: BeautifulSoup4 + advanced headers
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB for similarity search
- **LLM**: Ollama `llama3.2:1b` for local inference
- **Frontend**: Streamlit with custom CSS
- **RAG Framework**: LangChain for pipeline orchestration

## 📊 **Performance Benchmarks**

*Tested on RTX 4070 + i7-13620H + 32GB RAM:*

| Operation | Time | Notes |
|-----------|------|-------|
| Website Scraping | 2-5s | Depends on site complexity |
| Text Processing | 100-500ms | Chunking + cleaning |
| Embedding Generation | 50-200ms/chunk | ~500 tokens per chunk |
| Vector Search | 10-50ms | ChromaDB similarity query |
| LLM Response | 1-3s | Ollama llama3.2:1b |
| **Total Query Time** | **3-8s** | End-to-end experience |

### **Resource Usage:**
- **GPU**: 60-70% utilization during inference
- **VRAM**: 5-6GB typical usage
- **RAM**: 8-12GB during operation
- **CPU**: 40-60% during scraping

## 🎨 **Features**

### **Web Interface:**
- 🎯 **Smart URL Validation** - Warns about incompatible sites
- 📱 **Responsive Design** - Works on desktop and mobile
- 🔄 **Real-time Progress** - Step-by-step processing feedback
- 🌟 **Website Recommendations** - Curated list of working sites
- ❓ **Sample Questions** - Quick-start question templates

### **RAG Pipeline:**
- 🧠 **Advanced Embeddings** - Semantic similarity search
- 📚 **Smart Chunking** - Context-aware text splitting
- 🔍 **Fallback URLs** - Automatic retry with backup sites
- 🤖 **Local LLM** - Private, no-API-cost inference
- ⚡ **Caching** - Streamlit cache for faster responses

### **Bot Detection Evasion:**
- 🕵️ **Advanced Headers** - Full browser mimicry
- ⏱️ **Smart Delays** - Site-specific timing
- 🔄 **Multiple Methods** - Fallback content extraction
- 🛡️ **Error Handling** - Graceful failure recovery

## 🔧 **Development**

### **Project Structure:**
```
├── RAG_Webscraping_Integration.ipynb    # Complete development notebook
├── streamlit_rag_app.py                 # Production Streamlit app  
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore patterns
```

### **Extending the System:**

**Add New LLM Models:**
```python
# In webscrape_rag_qa function
llm = Ollama(model="llama3.2:3b")  # Larger model
```

**Add New Embedding Models:**
```python
# In scrape_website function  
model = SentenceTransformer('all-mpnet-base-v2')  # Better quality
```

**Custom Website Handlers:**
```python
# Add to scrape_website function
if 'custom-site.com' in current_url:
    # Custom extraction logic
    content = extract_custom_content(soup)
```


**⭐ Star this repo if it helped you! ⭐**


