from __init__ import *


# Définition des constantes
HF_TOKEN = "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"  # Remplacer par votre API HuggingFace
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
QUERY = "Please do a comparative study on these PDF documents"


def read_pdf(file_path):
    """Reads a PDF file and extracts text from each page."""
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def remove_references(text):
    """Removes references/citations sections from the text."""
    reference_patterns = [
        r'\bReferences\b',
        r'\bBibliography\b',
        r'\bCitations\b',
        r'\bWorks Cited\b',
        r'\bLiterature Cited\b'
    ]
    combined_pattern = '|'.join(reference_patterns)
    match = re.search(combined_pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    return text

def clean_text(text):
    """Cleans the text by removing links, dates, etc."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_text(text, sentence_count=15):
    """Summarizes the text using the LexRank algorithm."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=sentence_count)
    return summary

def extract_and_summarize_pdf(file_path, sentence_count=30):
    """Main function to execute the text extraction, cleaning, and summarization."""
    text = read_pdf(file_path)
    text = remove_references(text)
    text = clean_text(text)
    summary = summarize_text(text, sentence_count)
    summary_text = ' '.join([str(sentence) for sentence in summary])
    return summary_text

def read_pdfs(pdf_paths):
    documents = []
    i=0
    for pdf_path in pdf_paths:
        summary = extract_and_summarize_pdf(pdf_path)
        document = Document(text=summary, metadata={"source": pdf_path})
        documents.append(document)

    return documents



def setup_llm():
    return HuggingFaceInferenceAPI(model_name=MODEL_NAME, token=HF_TOKEN)

def setup_embed_model():
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))


def setup_service_context(llm):
    Settings.llm = llm
    Settings.chunk_size = 1200

def setup_storage_context():
    graph_store = SimpleGraphStore()
    return StorageContext.from_defaults(graph_store=graph_store)

def construct_knowledge_graph_index(documents, storage_context, embed_model):
    return KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=5,
        storage_context=storage_context,
        embed_model=embed_model,
        include_embeddings=True
    )

def create_query_engine(index):
    return index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
    )

def generate_response(query_engine, query):
    response = query_engine.query(query)
    return response

def save_response(response, filename="response.md"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.response.split("<|assistant|>")[-1].strip())
    print("Le document a été sauvegardé avec succès")

def visualize_knowledge_graph(index, output_html="Knowledge_graph.html"):
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.save_graph(output_html)
    display(HTML(filename=output_html))
    print("Le graphe de connaissances a été sauvegardé et affiché avec succès.")

