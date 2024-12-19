Theory of Computation Training Assistant
An AI-powered educational assistant that provides comprehensive information about microprocessors using document retrieval and natural language processing. The system combines ChromaDB for efficient document storage and retrieval with Google's Gemini AI for intelligent response generation.

🚀 Features
PDF document processing and intelligent chunking
Semantic search using sentence transformers
Vector storage with ChromaDB
Natural language query processing
Multi-language support with automatic translation
Interactive chat interface using Streamlit
Context-aware responses powered by Google's Gemini AI
📋 Prerequisites
Python 3.8+
Required Python packages (install via pip):
langchain
sentence-transformers
chromadb
streamlit
google-generativeai
deep-translator
python-dotenv
PyPDF2
tqdm
🛠️ Installation
Clone the repository:
git clone https://github.com/KaanSezen1923/TheoryofComputation_Education_Bot.git
cd TheoryofComputation_Education_Bot
Install dependencies:
pip install -r requirements.txt
Set up your environment variables:
Create a .env file in the project root
Add your Gemini API key:
GEMINI_API_KEY=your_api_key_here
📁 Project Structure
The project consists of two main components:

Document Processor (vector_database.py)

Loads and processes PDF documents
Splits documents into manageable chunks
Creates embeddings
Stores data in ChromaDB
Chat Interface (main.py)

Provides a Streamlit-based user interface
Handles query processing
Manages conversation context
Integrates with Gemini AI for response generation
🚦 Usage
First, process your documents:
python vector_database.py
This will:

Load PDFs from the HESAPLAMA DATA directory
Process and split the documents
Create embeddings
Store them in ChromaDB
Launch the chat interface:
streamlit run main.py
Access the application through your web browser (typically at http://localhost:8501)
📝 Configuration
Document Processor Settings
DATA_PATH = "HESAPLAMA DATA"
CHROMA_PATH = "Hesaplama_Chroma_Database"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Text Splitting Parameters
chunk_size = 10000
chunk_overlap = 300
Gemini AI Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

⚠️ Error Handling
The application includes comprehensive error handling for:

PDF loading failures
Document processing errors
Embedding creation issues
ChromaDB operations
API communication problems
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
LangChain for document processing utilities

Sentence Transformers for embedding generation

ChromaDB for vector storage

Google's Gemini AI for response generation

Streamlit for the user interface

📞 Support
If you encounter any issues or have questions, please open an issue in the GitHub repository.

##Results
![image](https://github.com/user-attachments/assets/5ad079f0-a006-4582-afd0-af12cd7bb49b)
