__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import uuid
import time
from typing import List, TypedDict, Literal
from dotenv import load_dotenv

# LangChain / AI Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph

# --- 1. UMGEBUNGSVARIABLEN LADEN ---
load_dotenv()

# --- KONFIGURATION ---
CHROMA_PATH = "chroma_db"
LLM_MODEL = "gemma-3-27b-it"

# --- DEFINITION DER DEMO-SZENARIEN (Für das Dropdown) ---
DEMO_SCENARIOS = {
    "--- Bitte wählen ---": None,
    "🧪 Glasbruch (SOP vorhanden)": "Ich habe Glasbruch im Isolator der Abfülllinie. Was tun?",
    "📉 pH-Wert Abfall (SOP vorhanden)": "Der pH-Wert im Bioreaktor ist auf 6.5 gesunken.",
    "⚠️ Unbekannter Fehler (Eskalation)": "Fehlercode E-999: Flux-Kompensator defekt.",
    "🚫 Wetter / Off-Topic (Eskalation)": "Wie wird das Wetter morgen in Frankfurt?"
}

# Seite konfigurieren
st.set_page_config(
    page_title="Sanofi AI Assistant", 
    page_icon="🏭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "escalation_logs" not in st.session_state:
    st.session_state.escalation_logs = []
if "pending_escalation" not in st.session_state:
    st.session_state.pending_escalation = None

# --- UI HEADER & STATUS ---
col_header, col_status = st.columns([3, 1])

with col_header:
    st.title("🏭 Smart Manufacturing & Supply Agent")
    st.markdown("""
    © 2025 | Dr. Eike Bent Preuß | Sanofi Demo 1.5
    
    **Architektur & Funktion:**
    * 🧠 **LLM:** Google Gemma 3 (27B) für Reasoning.
    * 📚 **RAG:** ChromaDB für SOPs & Historische Daten.
    * 🔄 **LangGraph:** Zyklischer "Auditor-Loop" zur Qualitätskontrolle.
    """)

with col_status:
    st.caption("SYSTEM STATUS")
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        st.success("🟢 Datenbank Online")
    else:
        st.error("🔴 Datenbank Offline")
    
    if os.environ.get("GOOGLE_API_KEY"):
        st.success("🟢 LLM Verbunden")
    else:
        st.warning("🟠 API Key fehlt")

# --- ARCHITEKTUR VISUALISIERUNG (Ersetzt den alten Expander) ---
with st.expander("ℹ️ Architektur & Workflow anzeigen (Live-Graph)", expanded=False):
    st.markdown("### 🧠 Der interne Denkprozess (LangGraph)")
    st.write("Im Gegensatz zu einfachen Chatbots nutzt dieses System einen **zyklischen Graphen** mit Qualitätskontrolle:")
    
    # Graphviz Diagramm - Sieht sehr technisch und professionell aus
    # Graphviz Diagramm - Optimiert für Light & Dark Mode
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            bgcolor="transparent";
            
            # Global: Nutze SlateGrey für Texte/Linien (Lesbar auf Weiß & Dunkel)
            graph [fontname="Arial", fontsize=10];
            edge [fontname="Arial", fontsize=10, color="#607D8B", fontcolor="#607D8B"];
            
            # Nodes: Immer schwarzer Text auf Pastell-Hintergrund (Maximaler Kontrast)
            node [shape=box, style="filled,rounded", fontname="Arial", fontsize=12, fontcolor="#000000", margin=0.1];

            # Nodes definieren
            Start [shape=oval, fillcolor="#e0e0e0", label="Start\n(User Frage)"];
            Retrieve [label="🔍 Retrieve\n(ChromaDB)", fillcolor="#d1c4e9"];
            Generate [label="📝 Generate\n(Gemma 27B)", fillcolor="#bbdefb"];
            Auditor [label="⚖️ Auditor\n(GMP Check)", fillcolor="#ffcc80", shape=diamond, style="filled"];
            End [shape=oval, fillcolor="#c8e6c9", label="✅ Output\n(Antwort)"];
            Escalate [shape=oval, fillcolor="#ffcdd2", label="🛑 Eskalation\n(Supervisor)"];

            # Standard-Fluss (SlateGrey)
            Start -> Retrieve;
            Retrieve -> Generate [label="Kontext"];
            Generate -> Auditor [label="Entwurf"];
            
            # Entscheidungs-Wege (Explizite Farben, die überall wirken)
            Auditor -> End [label="Pass", color="#2e7d32", fontcolor="#2e7d32", penwidth=2.0];     # Dunkelgrün
            Auditor -> Generate [label="Fail / Retry", color="#d32f2f", fontcolor="#d32f2f", style="dashed"]; # Dunkelrot
            Auditor -> Escalate [label="No Data", color="#e65100", fontcolor="#e65100"];           # Dunkelorange
        }
    """)

    st.info("""
    **Legende der Logik:**
    1. **Retrieve:** Zuerst sucht der Agent passende SOPs und historische Fehlerberichte.
    2. **Generate:** Er entwirft basierend darauf eine Lösung.
    3. **Auditor (The Loop):** Dies ist der kritische Schritt. Ein virtueller Prüfer validiert die Antwort. 
       - Ist sie falsch, zwingt er den Agenten zur **Selbstkorrektur** (roter Pfeil).
       - Ist sie korrekt, wird sie freigegeben (grüner Pfeil).
    """)

# --- SIDEBAR (Supervisor Dashboard) ---
with st.sidebar:
    st.header("👨‍💼 Supervisor Dashboard")
    st.info("Eingehende Tickets aus der Produktion:")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        api_key = st.text_input("Google API Key", type="password")
        if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("---")
    
    if not st.session_state.escalation_logs:
        st.caption("Keine offenen Tickets.")
    else:
        for log in st.session_state.escalation_logs:
            st.error(f"**TICKET #{log['id']}**\n\n{log['msg']}")

# --- RESSOURCEN ---
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(CHROMA_PATH): return None
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

@st.cache_resource
def get_llm():
    if not os.environ.get("GOOGLE_API_KEY"): return None
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, temperature=0, max_tokens=1024,
        safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
    )

# --- LANGGRAPH STATE ---
class AgentState(TypedDict):
    question: str
    context: List[str]
    metadata: List[dict]
    answer: str
    revision_number: int
    critique: str
    status: str 

# --- NODES ---
def build_graph():
    vectorstore = get_vectorstore()
    llm = get_llm()
    if not vectorstore or not llm: return None

    # 1. RETRIEVE
    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = vectorstore.similarity_search(question, k=3)
        return {
            "context": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs],
            "revision_number": 0
        }

    # 2. GENERATE
    def generate_node(state: AgentState):
        question = state["question"]
        context = state["context"]
        metadata = state["metadata"]
        critique = state.get("critique", "")
        
        system_msg = """
        Du bist ein erfahrener QA-Manager bei Sanofi.
        
        REGELN:
        1. Nutze NUR den Kontext. Erfinde nichts.
        2. Wenn der Kontext leer ist oder die Information zur Beantwortung fehlt, antworte EXAKT mit: "NO_DATA".
        
        STRUKTUR DER ANTWORT (Nur wenn Daten vorhanden):
        1. **Problemverständnis:** Kurze Zusammenfassung.
        2. **Analyse:** Was sagt die SOP/Historie?
        3. **Maßnahme:** Was muss JETZT getan werden?
        4. **Referenz:** Nenne SOP-Nummern oder DEV-IDs.
        """
        
        if critique:
            system_msg += f"\n\nFEEDBACK VOM AUDITOR: {critique}. Du musst die Antwort korrigieren!"

        prompt = ChatPromptTemplate.from_template(system_msg + """
        KONTEXT: {context}
        QUELLE: {metadata}
        FRAGE: {question}
        ANTWORT:
        """)
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": "\n\n".join(context), "question": question, "metadata": str(metadata)
        })
        return {"answer": response}

    # 3. AUDITOR
    def auditor_node(state: AgentState):
        answer = state["answer"]
        question = state["question"]
        
        if "NO_DATA" in answer:
            return {"status": "ESCALATE", "critique": "Keine Daten."}

        auditor_prompt = ChatPromptTemplate.from_template("""
        Prüfe diese Antwort auf GMP-Konformität.
        Frage: {question}
        Antwort: {answer}
        
        Kriterien:
        1. Wurde die Struktur eingehalten?
        2. Wirkt die Antwort logisch?
        
        Antworte mit "PASS" oder "FAIL: [Grund]".
        """)
        chain = auditor_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "answer": answer})
        
        if "PASS" in result:
            return {"status": "SUCCESS", "critique": ""}
        else:
            return {"status": "RETRY", "critique": result, "revision_number": state["revision_number"] + 1}

    # ROUTING
    def router(state: AgentState) -> Literal["generate", "end_success", "end_escalate"]:
        if state["status"] == "ESCALATE": return "end_escalate"
        if state["status"] == "SUCCESS": return "end_success"
        if state["status"] == "RETRY":
            return "generate" if state["revision_number"] <= 1 else "end_success"
        return "end_success"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "auditor")
    workflow.add_conditional_edges("auditor", router, {
        "generate": "generate", "end_success": END, "end_escalate": END
    })
    return workflow.compile()

# --- CHAT & LOGIC FLOW ---

# 1. Chat Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Logic für Bestätigungs-Buttons (Vorgesetzte Eskalation)
if st.session_state.pending_escalation:
    with st.chat_message("assistant"):
        st.warning(f"⚠️ **Keine Datenbank-Treffer für:** '{st.session_state.pending_escalation}'")
        st.write("Dies scheint ein unbekannter Fehler oder ein Off-Topic Thema zu sein. Soll ein Ticket für den Supervisor erstellt werden?")
        
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Ja, melden"):
            ticket_id = str(uuid.uuid4())[:4].upper()
            st.session_state.escalation_logs.append({
                "id": ticket_id,
                "msg": st.session_state.pending_escalation
            })
            st.session_state.messages.append({"role": "assistant", "content": f"✅ Ticket #{ticket_id} an Supervisor gesendet."})
            st.session_state.pending_escalation = None
            st.rerun()

        if col_no.button("❌ Nein, verwerfen"):
            st.session_state.messages.append({"role": "assistant", "content": "Meldung verworfen."})
            st.session_state.pending_escalation = None
            st.rerun()

# 3. INPUT BEREICH (Dropdown ODER Chat Input)
if not st.session_state.pending_escalation:
    
    # --- NEU: Szenarien Auswahl ---
    st.markdown("### 💬 Neue Anfrage starten")
    col_sel, col_btn = st.columns([4, 1])
    
    with col_sel:
        selected_scenario = st.selectbox(
            "Wähle ein Test-Szenario (Optional):", 
            options=list(DEMO_SCENARIOS.keys()),
            label_visibility="collapsed"
        )
    
    start_scenario = False
    with col_btn:
        if st.button("▶️ Szenario starten", type="primary"):
            start_scenario = True

    # Chat Input (Manuelle Eingabe)
    chat_input = st.chat_input("...oder beschreibe dein Problem hier manuell")

    # --- ENTSCHEIDUNG: Woher kommt der Prompt? ---
    final_prompt = None
    
    if start_scenario and selected_scenario and DEMO_SCENARIOS[selected_scenario]:
        final_prompt = DEMO_SCENARIOS[selected_scenario]
    elif chat_input:
        final_prompt = chat_input

    # --- AUSFÜHRUNG ---
    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        st.rerun() # Rerun, um die User-Nachricht oben anzuzeigen und Input zu clearen

    # (Nach dem Rerun wird die letzte Nachricht verarbeitet, wenn sie vom User ist und der Bot noch nicht geantwortet hat)
    # Logik-Trick: Wir prüfen, ob die letzte Nachricht vom User ist. Wenn ja, antwortet der Bot.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        last_user_prompt = st.session_state.messages[-1]["content"]
        
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Bitte API Key eingeben!")
        else:
            app = build_graph()
            if app:
                with st.chat_message("assistant"):
                    status_container = st.status("🚀 Agent startet...", expanded=True)
                    try:
                        # 1. Retrieve
                        status_container.write("📚 Schritt 1: Suche in GMP-Datenbank...")
                        inputs = {"question": last_user_prompt, "revision_number": 0, "critique": ""}
                        final_state = app.invoke(inputs)

                        # 2. Generate
                        status_container.write("📝 Schritt 2: Antwort wird generiert...")
                        
                        # 3. Auditor Logic
                        if final_state["status"] == "RETRY" or final_state["revision_number"] > 0:
                            status_container.write("⚖️ **Auditor:** ⚠️ Mängel entdeckt!")
                            status_container.write(f"🔧 *Korrektur:* {final_state.get('critique', 'GMP Anpassung')}")
                            status_container.update(label="✅ Antwort nach Korrektur freigegeben", state="complete")
                        elif final_state["status"] == "ESCALATE":
                            status_container.write("⚖️ **Auditor:** 🛑 Keine validen Daten gefunden.")
                            status_container.update(label="⚠️ Prozess gestoppt", state="error")
                        else:
                            status_container.write("⚖️ **Auditor:** ✅ Antwort ist GMP-konform.")
                            status_container.update(label="✅ Fertig", state="complete")

                        # --- ERGEBNIS ---
                        if final_state["status"] == "ESCALATE":
                            st.session_state.pending_escalation = last_user_prompt
                            st.rerun()
                        else:
                            # 1. Helper-Funktion für Smart-Snippets (Lokal definiert)
                            def extract_relevant_context(full_text, query):
                                """Zeigt nur Sätze/Abschnitte, die Keywords enthalten."""
                                # Keywords aus der User-Frage extrahieren (alles > 3 Buchstaben)
                                keywords = [w.lower() for w in query.split() if len(w) > 3]
                                
                                # Text in logische Blöcke teilen (Zeilenumbrüche nutzen)
                                lines = full_text.split('\n')
                                relevant_snippets = []
                                
                                for line in lines:
                                    line_clean = line.strip()
                                    if not line_clean: continue # Leere Zeilen überspringen
                                    
                                    # Check: Enthält die Zeile ein Keyword ODER ist es eine wichtige Überschrift?
                                    is_hit = any(k in line_clean.lower() for k in keywords)
                                    is_header = any(h in line_clean for h in ["URSACHE", "CAPA", "MASSNAHME", "PROBLEM", "Fehler:"])
                                    
                                    if is_hit or (is_header and len(relevant_snippets) > 0):
                                        # Markdown Header bereinigen (# entfernen)
                                        clean_line = line_clean.replace("#", "").strip()
                                        relevant_snippets.append(clean_line)
                                
                                # Fallback: Wenn gar kein Keyword-Match (da Vektorsuche semantisch ist),
                                # zeigen wir einfach die ersten 300 Zeichen.
                                if not relevant_snippets:
                                    return full_text[:300] + " [...]"
                                
                                # Wir nehmen max 4 Schnipsel, damit es nicht zu lang wird
                                display_text = "\n[...]\n".join(relevant_snippets[:5])
                                return "[...]\n" + display_text + "\n[...]"

                            # 2. Anzeige im Expander
                            with st.expander("🔍 Quellen & SOPs (Evidence)", expanded=False):
                                for i, doc in enumerate(final_state["context"]):
                                    # Metadaten
                                    meta = final_state["metadata"][i] if i < len(final_state["metadata"]) else {}
                                    source_name = os.path.basename(meta.get("source", "Unknown"))
                                    
                                    # Smart Snippet generieren
                                    smart_text = extract_relevant_context(doc, last_user_prompt)
                                    
                                    # Anzeige als Karte
                                    st.markdown(f"**📄 Quelle {i+1}:** `{source_name}`")
                                    st.markdown(f"""
                                    <div style="
                                        background-color: rgba(128, 128, 128, 0.1); 
                                        padding: 10px; 
                                        border-radius: 5px; 
                                        border-left: 4px solid #4CAF50;
                                        font-size: 0.85em; 
                                        color: inherit;
                                        font-family: monospace;
                                        line-height: 1.4;
                                        white-space: pre-wrap;">
                                        {smart_text}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.divider()
                            
                            # 3. Finale Antwort
                            st.markdown(final_state["answer"])
                            st.session_state.messages.append({"role": "assistant", "content": final_state["answer"]})

                    except Exception as e:
                        status_container.update(label="💥 Fehler", state="error")
                        st.error(str(e))
