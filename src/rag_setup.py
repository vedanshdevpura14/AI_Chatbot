import os # OS functions ke liye, jaise folder banana ya paths join karna
import re # Regular Expressions (regex) ke liye, text clean karne mein kaam aata hai
import json # JSON data padhne aur likhne ke liye
import requests # Internet se web pages ya data fetch karne ke liye (HTTP requests)
from bs4 import BeautifulSoup # HTML pages ko parse aur unse text nikalne ke liye library
import chromadb # ChromaDB import kar rahe hain, jo ek vector database hai embeddings store karne ke liye

# --- Configuration ---
# BASE_DIR humare project ka main folder path nikalta hai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Raw aur processed data store karne ke paths define kar rahe hain
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# Chroma vector DB kahan save hoga uska path
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

CHUNK_SIZE_WORDS = 200 # Ek tukda (chunk) kitne words ka hoga
CHUNK_OVERLAP_WORDS = 40 # Do chunks ke beech mein kitne words common honge (context loss bachane ke liye)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Sentence transformer ka local model name (agar use karein toh)
CHROMA_COLLECTION_NAME = "chatbot_knowledge" # Vector DB mein collection ka naam

# Wikipedia ke URLs jinki knowledge chatbot ko deni hai
URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
]

# Ensure directories exist
# Agar yeh data folders pehle se nahi hain, toh unhe bana dega (exist_ok=True error nahi aane dega agar pehle se hain)
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- 1. Scraper ---
def clean_filename(url: str) -> str:
    # URL se 'http://' ya 'https://' hata deta hai
    name = re.sub(r"https?://", "", url)
    # Special characters ko underscore (_) se replace karta hai takki valid file name ban sake
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    # Shuru aur aakhir ke underscores hatake '.txt' laga deta hai
    return name.strip("_") + ".txt"

def scrape_page(url: str) -> str:
    # Ek fake User-Agent bhej rahe hain takki website ko lage ki real browser se request aa rahi hai
    headers = {"User-Agent": "Mozilla/5.0 (educational project bot)"}
    # URL pe GET request bhejte hain
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status() # Agar request fail ho gayi (jaise 404 error), toh exception throw karega
    
    # BeautifulSoup HTML content ko parse karne ke liye
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Bekar tags jinme actual content nahi hota, unko hata (decompose) rahe hain
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
        
    # Sirf visible text nikal rahe hain, words ke beech space lagake
    text = soup.get_text(separator=" ")
    # Extra spaces aur newlines ko ek single space mein badal rahe hain
    return re.sub(r"\s+", " ", text).strip()

def run_scraper():
    print("--- STEP 1: Scraping ---")
    # Har ek URL ke liye loop chalayenge
    for url in URLS:
        print(f"Scraping: {url}")
        try:
            text = scrape_page(url) # Page ka text scrape karenge
        except Exception as e:
            print(f"  Failed: {e}") # Agar error aaya toh print karenge
            continue
            
        # File ka pura path banayenge jahan save karna hai
        filepath = os.path.join(RAW_DATA_DIR, clean_filename(url))
        # Text file open karke usme scraped text likh denge
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved {len(text)} characters -> {filepath}")

# --- 2. Chunker ---
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split() # Text ko words mein tod lete hain
    chunks = []
    start = 0
    # Jab tak words bache hain, chunks banate rahenge
    while start < len(words):
        end = start + chunk_size # Chunk ka end point set karenge
        chunks.append(" ".join(words[start:end])) # Words jod kar wapas string banayenge aur list mein dalenge
        if end >= len(words): break # Agar aakhir tak pahunch gaye toh loop tod do
        start = end - overlap # Agla chunk thoda overlap karke start hoga
    return chunks

def run_chunker():
    print("\n--- STEP 2: Chunking ---")
    all_chunks = []
    chunk_id = 0
    # Raw data directory ki har ek file pe loop
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".txt"): continue # Sirf text files padhenge
        
        # File khol ke pura text read kar rahe hain
        with open(os.path.join(RAW_DATA_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()
            
        # Text ko chote chote chunks mein baat (split) rahe hain
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        
        # Har chunk ko ek dictionary mein metadata (jaise id, source file) ke saath save kar rahe hain
        for chunk in chunks:
            all_chunks.append({"id": f"chunk_{chunk_id}", "source": filename, "text": chunk})
            chunk_id += 1
        print(f"{filename}: {len(chunks)} chunks created")
    
    # Saare chunks ek JSON file mein save karenge
    output_path = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Total chunks saved: {len(all_chunks)}")

# --- 3. Embed Store ---
def run_embed_store():
    print("\n--- STEP 3: Embedding and Storing ---")
    chunks_path = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print("No chunks found. Did you run the chunker?")
        return
    
    # JSON file se banaye huye chunks load kar rahe hain
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Gemini AI ka API key environment se nikal rahe hain (Embeddings ke liye zaroori hai)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY environment variable is missing!")
        print("Please set it in your terminal or .env file before running this.")
        return
        
    print("Connecting to Google Gemini API for embeddings...")
    # ChromaDB ke saath Google Generative AI ki embedding function jod rahe hain
    import chromadb.utils.embedding_functions as embedding_functions
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/gemini-embedding-2"
    )
    
    # Text, ID aur Metadata ko alag-alag lists mein daal rahe hain
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    
    # ChromaDB database ka persistent client bana rahe hain jo disk pe data save karta hai
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Purana collection delete kar rahe hain takki dimensions mein koi conflicts na ho (clean slate)
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except:
        pass
        
    # Naya collection create kar rahe hain Google embedding function ke saath
    collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=google_ef)
    
    print("Generating embeddings via API and storing in Chroma...")
    import time
    
    # API ke rate limits cross na ho (free tier), isliye chunks ko 25-25 ke batch mein bhej rahe hain
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        print(f"  Processing batch {i} to {end}...")
        
        # Batch ko vector database (ChromaDB) mein add kar rahe hain, text embeddings mein convert ho jayega
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        time.sleep(5) # Har batch ke baad 5 second ruk rahe hain takki Gemini API rate limit na lagaye
        
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

def main():
    # Saare steps ek ke baad ek chala rahe hain
    run_scraper() # Step 1: Internet se data lao
    run_chunker() # Step 2: Data ko tukdo mein baanto
    run_embed_store() # Step 3: Embeddings nikalo aur VectorDB mein save karo
    print("\nSetup complete! You can now run main.py")

if __name__ == "__main__":
    # Agar directly run kiye, toh ye script main function ko call karegi
    main()
