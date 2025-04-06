FinSight: Intelligent Financial News Research & Q&A Tool

FinSight is an intelligent Q&A platform that streamlines financial news research by retrieving semantically relevant insights from online articles. Built with LangChain, OpenAI Embeddings, FAISS, and Streamlit, it allows users to query financial content in real-time, enabling informed decision-making.

⸻

🚀 Features
	•	🔎 Smart Article Retrieval: Load URLs or upload text files to extract full article content from financial domains.
	•	🧠 Semantic Search: Uses OpenAI embeddings + FAISS to build an efficient vector index for accurate and fast information retrieval.
	•	💬 Real-time Q&A: Ask questions related to loaded news; receive intelligent answers sourced from processed articles.
	•	⚡ Optimized Performance: Supports LoRA-based fine-tuning for domain-specific enhancement and faster query response.
	•	🌐 User-Friendly Interface: Clean and responsive UI built using Streamlit.

⸻

🛠️ Tech Stack
	•	LangChain for document loading and LLM orchestration
	•	OpenAI Embeddings to vectorize article content
	•	FAISS for fast similarity search
	•	Streamlit for front-end interface
	•	LoRA (Low-Rank Adaptation) for optional fine-tuning of the LLM
	•	Python as the core development language

⸻

🔧 Installation
	1.	Clone the repo:

git clone https://github.com/AmeyaBhalgat/FinSight.git
cd FinSight

	2.	Install dependencies:

pip install -r requirements.txt

	3.	Add your OpenAI API key in a .env file:

OPENAI_API_KEY=your_openai_key_here



⸻

▶️ Usage

Run the Streamlit app:

streamlit run main.py

Workflow:
	•	Input URLs or upload a file with article links.
	•	Click “Process URLs” to:
	•	Extract content via LangChain’s loaders
	•	Split text and compute embeddings with OpenAI
	•	Index data using FAISS for fast retrieval
	•	Ask a question related to the articles — get accurate answers with source references.

⸻

📂 Project Structure

FinSight/
│
├── main.py                  # Streamlit app
├── requirements.txt         # Dependencies
├── faiss_store_openai.pkl   # Saved FAISS index
├── .env                     # API Keys
└── README.md                # Project documentation



⸻

🧪 Example URLs to Try

https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html



⸻

📌 Future Work (Research Extension)
	•	✨ Implement sentiment analysis on extracted articles using NER + LSTM to classify market sentiment (positive/neutral/negative)
	•	📈 Predict market trends from news using semantic signals, enabling deeper insights for financial forecasting

⸻

Let me know if you’d like a version with images, badges, or deployed links!
