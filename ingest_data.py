import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- KONFIGURATION ---
DATA_PATH = "data"              # Wo liegen unsere generierten Textfiles?
CHROMA_PATH = "chroma_db"       # Wo soll die Datenbank gespeichert werden?

# Wir nutzen ein kleines, schnelles Modell für die Embeddings (läuft lokal auf deiner CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"🚀 Starte Ingestion-Prozess...")
    
    # 1. Datenbank bereinigen (Falls sie schon existiert, löschen wir sie für einen sauberen Start)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"   Alte Datenbank in '{CHROMA_PATH}' gelöscht.")

    # 2. Dokumente laden
    # Wir laden alle .txt Dateien aus dem data Ordner
    print(f"📂 Lade Dokumente aus '{DATA_PATH}'...")
    loader = DirectoryLoader(
            DATA_PATH, 
            glob="*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}  # <--- WICHTIG! Das behebt die Ã¼ Fehler
        )
        
    documents = loader.load()

    # 3. Text Splitting (Chunking)
    # Ein ganzes Dokument ist oft zu lang für das LLM. Wir zerteilen es in Häppchen ("Chunks").
    # chunk_size=1000: Ein Block ist ca. 1000 Zeichen lang.
    # chunk_overlap=200: Die Blöcke überlappen sich, damit der Kontext nicht abreißt.
    print(f"✂️  Zerteile Texte in Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True, # Hilft später, die Stelle im Original zu finden
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   {len(chunks)} Text-Chunks erstellt.")

    # 4. Embeddings erstellen & Speichern
    # Hier passiert die Magie: Text wird in Vektoren umgewandelt.
    print(f"🧠 Initialisiere Embedding-Modell ({EMBEDDING_MODEL_NAME})...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"💾 Speichere in ChromaDB (das kann einen Moment dauern)...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ FERTIG! Datenbank wurde in '{CHROMA_PATH}' gespeichert.")
    print("---")
    
    # Kleiner Test, ob es geklappt hat
    print("🔍 Test-Suche: 'Glasbruch'")
    results = db.similarity_search("Glasbruch", k=1)
    if results:
        print(f"   Gefundenes Dokument: {results[0].metadata['source']}")
        print(f"   Inhalt-Vorschau: {results[0].page_content[:100]}...")
    else:
        print("   Nichts gefunden.")

if __name__ == "__main__":
    main()