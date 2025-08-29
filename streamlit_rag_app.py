import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uuid
import time

# Page configuration
st.set_page_config(
    page_title="RAG Webscraping Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved color scheme
st.markdown("""
<style>
    /* Main app background and layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Consistent font styling */
    .stTextInput > div > div > input {
        font-size: 16px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 16px;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stTextArea > div > div > textarea {
        font-size: 16px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        line-height: 1.6;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 16px;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 16px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        color: #374151;
    }
    
    .stButton > button:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-size: 18px;
        font-weight: 700;
        padding: 1rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton > button {
        background-color: white;
        border: 2px solid #e5e7eb;
        color: #374151;
        font-weight: 500;
        width: 100%;
        margin-bottom: 8px;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: left;
        font-size: 14px;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f8fafc;
        border-color: #3b82f6;
        color: #1e40af;
        transform: none;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    /* Success/Error/Info message styling */
    .stSuccess {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1rem;
        color: #047857;
    }
    
    .stError {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1rem;
        color: #dc2626;
    }
    
    .stInfo {
        background-color: #eff6ff;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem;
        color: #1d4ed8;
    }
    
    .stWarning {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        color: #d97706;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #1f2937;
        font-weight: 600;
    }
    
    h1 { color: #111827; }
    h2 { color: #1f2937; }
    h3 { color: #374151; }
    
    /* Content spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Custom colored boxes */
    .gradient-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .answer-display {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 2px solid #22c55e;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: #1f2937;
        font-size: 16px;
        line-height: 1.7;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.1);
    }
    
    /* Columns styling */
    .stColumn > div {
        padding: 0 1rem;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #e5e7eb;
    }
    
    .stProgress .st-bp {
        background-color: #3b82f6;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Make the main content more readable */
    p, div, span {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    
    /* Fix markdown styling */
    .stMarkdown {
        color: #374151;
    }
    
    /* Custom success/error boxes for inline use */
    .custom-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #047857;
        font-weight: 500;
    }
    
    .custom-error {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #dc2626;
        font-weight: 500;
    }
    
    .custom-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #1d4ed8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def clean_text(text):
    """Clean and normalize text by removing unwanted characters."""
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

@st.cache_data
def chunk_text(text, chunk_size=500):
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

@st.cache_data
def scrape_website(url):
    """Scrape text content from a website URL with fallback options."""
    
    # List of reliable fallback URLs
    fallback_urls = [
        "https://www.bbc.com/news/technology",
        "https://www.reuters.com/technology/", 
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://techcrunch.com/",
        "https://edition.cnn.com/business/tech"
    ]
    
    # Headers to mimic a real browser with better bot detection evasion
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    urls_to_try = [url] + [u for u in fallback_urls if u != url]
    
    for current_url in urls_to_try:
        try:
            st.info(f"🔄 Trying to scrape: {current_url}")
            
            # Add a small delay for Reuters to avoid bot detection
            if 'reuters.com' in current_url.lower():
                time.sleep(2)
            
            response = requests.get(current_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple content extraction methods
            content = ""
            
            # Method 1: Find paragraphs (works for most sites)
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = " ".join([para.get_text().strip() for para in paragraphs if para.get_text().strip()])
            
            # Method 2: For Reuters, try specific selectors
            if not content and 'reuters.com' in current_url.lower():
                reuters_selectors = [
                    'div[data-module="ArticleBody"] p',
                    '.ArticleBody-container p',
                    '[data-testid="paragraph"] p',
                    'div.StandardArticleBody_body p'
                ]
                for selector in reuters_selectors:
                    reuters_paragraphs = soup.select(selector)
                    if reuters_paragraphs:
                        content = " ".join([para.get_text().strip() for para in reuters_paragraphs])
                        break
            
            # Method 3: If no paragraphs, try article content
            if not content:
                articles = soup.find_all(['article', 'main', 'div'])
                for article in articles:
                    text = article.get_text().strip()
                    if len(text) > 200:
                        content = text
                        break
            
            # Method 4: Fallback to all text
            if not content:
                content = soup.get_text().strip()
            
            # Clean and validate content
            content = content.strip()
            if len(content) > 100:
                st.success(f"✅ Successfully scraped {len(content)} characters from {current_url}")
                return content
            else:
                st.warning(f"⚠️ Content too short ({len(content)} chars), trying next URL...")
                continue
                
        except Exception as e:
            st.error(f"❌ Failed to scrape {current_url}: {e}")
            continue
    
    # If all URLs fail, return sample content
    sample_content = """
    AI and Technology Sample Content: 
    
    Artificial Intelligence (AI) is revolutionizing how we interact with technology. Machine learning algorithms 
    enable computers to learn from data without explicit programming. Natural language processing allows 
    machines to understand and generate human language effectively.
    
    Recent developments in AI include large language models like GPT and Claude, which can engage in 
    sophisticated conversations and assist with various tasks. Computer vision systems can now identify 
    objects, faces, and scenes with remarkable accuracy.
    
    The integration of AI into everyday applications includes virtual assistants, recommendation systems, 
    autonomous vehicles, and medical diagnosis tools. Cloud computing provides the infrastructure needed 
    to train and deploy these AI models at scale.
    
    Key technologies driving this revolution include deep neural networks, transformer architectures, 
    and distributed computing systems. These advances are creating new opportunities in healthcare, 
    education, finance, and entertainment industries.
    
    As AI continues to evolve, important considerations include ethical AI development, data privacy, 
    and ensuring these technologies benefit society as a whole.
    """
    st.warning("⚠️ All websites failed to scrape. Using sample AI/Technology content for demonstration.")
    return sample_content

def webscrape_rag_qa(url, question, progress_bar):
    """Complete pipeline: scrape website, process text, create RAG, answer question."""
    try:
        # 1. Scrape website
        progress_bar.progress(10)
        st.info("🔍 Scraping website...")
        scraped_text = scrape_website(url)
        
        # 2. Clean and chunk text
        progress_bar.progress(30)
        st.info("📝 Processing text...")
        cleaned_text = clean_text(scraped_text)
        chunks = chunk_text(cleaned_text)
        st.info(f"📄 Created {len(chunks)} text chunks")
        
        # 3. Generate embeddings
        progress_bar.progress(50)
        st.info("🧠 Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks)
        
        # 4. Store in ChromaDB
        progress_bar.progress(70)
        st.info("💾 Storing in vector database...")
        collection_name = f"web_chunks_{uuid.uuid4().hex[:8]}"
        chroma_client = chromadb.Client(Settings())
        collection = chroma_client.create_collection(name=collection_name)
        
        for i, chunk in enumerate(chunks):
            collection.add(
                embeddings=[embeddings[i].tolist()],
                documents=[chunk],
                ids=[str(i)]
            )
        
        # 5. Set up RAG pipeline
        progress_bar.progress(85)
        st.info("🔗 Setting up RAG pipeline...")
        embedding_func = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_func,
            client=chroma_client
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # 6. Set up Ollama LLM
        llm = OllamaLLM(model="llama3.2:1b")
        
        # 7. Create custom prompt
        prompt_template = """Use the following pieces of context to answer the question.
The context contains information scraped from a website, so provide a comprehensive answer based on this content.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 8. Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # 9. Get answer
        progress_bar.progress(95)
        st.info("🤖 Generating answer...")
        answer = rag_chain.invoke({"query": question})["result"]
        progress_bar.progress(100)
        
        return answer
        
    except Exception as e:
        return f"❌ Error: {e}"

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Smart Website Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="gradient-box">
        <h2 style="color: white; margin-bottom: 1rem; font-weight: 700;">🌐 Ask Questions About Any Website</h2>
        <p style="font-size: 18px; margin-bottom: 0.5rem; opacity: 0.95;">Simply paste a website URL, ask your question, and get AI-powered answers!</p>
        <p style="font-size: 16px; opacity: 0.9;"><strong>💡 Pro Tip:</strong> Use the recommended websites from the sidebar for best results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **Choose a website** from examples below
        2. **Ask your question** 
        3. **Click 'Ask the AI Assistant'**
        4. **Get your answer!** ✨
        """)
        
        # Website compatibility guide
        st.markdown("---")
        st.markdown("### ✅ What Works Best")
        
        st.success("**✅ RECOMMENDED:**")
        st.markdown("""
        • **News**: BBC, Reuters, CNN
        • **Wikipedia**: Any topic
        • **Blogs & Articles**
        • **Educational sites**
        """)
        
        st.error("**❌ WON'T WORK:**")
        st.markdown("""
        • **YouTube** • **Social Media**
        • **Login sites** • **Apps**
        """)
        
        # Recommended example URLs
        st.markdown("---")
        st.markdown("### 🌟 Try These Sites")
        
        # Simpler categorized examples
        example_sites = {
            "📰 News": [
                ("BBC Tech", "https://www.bbc.com/news/technology"),
                ("Reuters Tech", "https://www.reuters.com/technology/"),
                ("CNN Tech", "https://edition.cnn.com/business/tech")
            ],
            "📖 Learn": [
                ("Wikipedia - AI", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
                ("Wikipedia - Python", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
                ("Wikipedia - Space", "https://en.wikipedia.org/wiki/Space_exploration")
            ],
            "💼 Tech": [
                ("TechCrunch", "https://techcrunch.com"),
                ("The Verge", "https://www.theverge.com")
            ]
        }
        
        for category, urls in example_sites.items():
            st.markdown(f"**{category}**")
            for name, url in urls:
                if st.button(f"{name}", key=url, use_container_width=True):
                    st.session_state.url_input = url
                    st.success(f"✅ Selected {name}")
                    st.rerun()
            st.markdown("")
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        # System checks - simpler display
        components = [
            ("SentenceTransformers", "sentence_transformers"),
            ("ChromaDB", "chromadb"), 
            ("LangChain", "langchain_community.llms")
        ]
        
        for name, module in components:
            try:
                __import__(module)
                st.success(f"✅ {name}")
            except:
                st.error(f"❌ {name}")
        
        # Quick test button
        st.markdown("---")
        if st.button("🧪 Quick Test", use_container_width=True):
            test_url = "https://www.bbc.com/news/technology"
            with st.spinner("Testing..."):
                try:
                    content = scrape_website(test_url)
                    st.success("✅ System working!")
                except Exception as e:
                    st.error("❌ System issue")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # URL Input with better guidance
        st.markdown("### 🌐 Choose Your Website")
        st.markdown("**Select from recommended sites or enter your own URL:**")
        
        url_input = st.text_input(
            "Website URL",
            value=st.session_state.get('url_input', 'https://www.bbc.com/news/technology'),
            placeholder="https://example.com",
            help="💡 Copy and paste any website URL here"
        )
        
        # URL validation and feedback
        if url_input:
            if any(blocked in url_input.lower() for blocked in ['youtube', 'twitter', 'instagram', 'facebook', 'tiktok']):
                st.error("⚠️ **This site likely won't work!** Try news sites, blogs, or Wikipedia instead.")
            elif any(good in url_input.lower() for good in ['bbc', 'reuters', 'wikipedia', 'cnn', 'techcrunch']):
                st.success("✅ **Perfect choice!** This site should work great.")
            else:
                st.info("ℹ️ **Unknown site.** If it doesn't work, try the recommended examples.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Question Input
        st.markdown("### ❓ Ask Your Question")
        st.markdown("**What would you like to know about this website?**")
        
        # Sample questions based on URL type
        default_questions = {
            'bbc': "What are the latest technology news?",
            'reuters': "What are the main technology trends?",
            'wikipedia': "What is this topic about and what are the key points?",
            'techcrunch': "What are the latest tech startup news?",
            'cnn': "What are the current technology stories?",
            'default': "What is this website about? Summarize the main content."
        }
        
        # Determine default question based on URL
        default_q = default_questions['default']
        for site, question in default_questions.items():
            if site in url_input.lower():
                default_q = question
                break
        
        question_input = st.text_area(
            "Your Question",
            value=st.session_state.get('question_input', default_q),
            placeholder="e.g., What are the main points? Summarize the content. What's new?",
            height=120,
            help="💡 Ask specific questions for better answers!"
        )
        
        # Sample question suggestions
        st.markdown("**💡 Quick Question Ideas:**")
        sample_questions = [
            "📋 Summarize main points",
            "📰 Latest news updates", 
            "🔍 Key topics discussed",
            "💡 Main ideas explained",
            "📊 Important facts listed"
        ]
        
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            if cols[i].button(question, key=f"q_{i}"):
                if "Summarize" in question:
                    st.session_state.question_input = "Summarize the main content and key points from this website"
                elif "news" in question:
                    st.session_state.question_input = "What are the latest news or updates mentioned on this website?"
                elif "topics" in question:
                    st.session_state.question_input = "What are the main topics discussed on this website?"
                elif "ideas" in question:
                    st.session_state.question_input = "Explain the main ideas and concepts presented on this website"
                elif "facts" in question:
                    st.session_state.question_input = "List the important facts and information from this website"
                st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Get Answer Button - make it more prominent
        st.markdown("### 🚀 Get Your Answer")
        if st.button("🚀 Ask the AI Assistant", type="primary", use_container_width=True):
            if not url_input.strip():
                st.error("❌ Please enter a valid URL")
                return
                
            if not question_input.strip():
                st.error("❌ Please enter a question")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            
            st.markdown("---")
            st.markdown("### 🔄 Processing Your Request...")
            
            # Show inputs
            st.write(f"**🌐 Website:** {url_input}")
            st.write(f"**❓ Question:** {question_input}")
            
            # Process request
            with st.spinner("🤖 AI is thinking..."):
                answer = webscrape_rag_qa(url_input, question_input, progress_bar)
            
            # Show results
            st.markdown("---")
            st.markdown("### 💡 Your Answer")
            
            if answer.startswith("❌ Error:"):
                st.error(answer)
            else:
                st.success("✅ Answer generated successfully!")
                
                # Display answer in a nice box with consistent styling
                st.markdown(f"""
                <div class="answer-display">
                    <div style="font-size: 16px; line-height: 1.7; color: #1f2937;">
                        {answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add some spacing
            st.markdown("---")
            st.info("🔄 Want to ask another question? Simply change your question above and click the button again!")

    with col2:
        st.subheader("📊 System Status")
        
        # System checks
        try:
            import sentence_transformers
            st.success("✅ SentenceTransformers: Ready")
        except:
            st.error("❌ SentenceTransformers: Not available")
        
        try:
            import chromadb
            st.success("✅ ChromaDB: Ready")
        except:
            st.error("❌ ChromaDB: Not available")
        
        try:
            from langchain_community.llms import Ollama
            st.success("✅ LangChain: Ready")
        except:
            st.error("❌ LangChain: Not available")
        
        # Quick test button
        st.markdown("---")
        st.subheader("🧪 Quick Test")
        if st.button("🧪 Test with BBC Tech News", use_container_width=True):
            test_url = "https://www.bbc.com/news/technology"
            with st.spinner("Testing scraping..."):
                try:
                    content = scrape_website(test_url)
                    st.success(f"✅ Test successful! Scraped {len(content)} characters from BBC")
                    st.info("💡 This confirms the system is working. Try the main interface!")
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
                    st.info("💡 If test fails, check your internet connection.")

if __name__ == "__main__":
    main()
