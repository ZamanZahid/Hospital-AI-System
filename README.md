<div align="center">

## Hospital AI RAG System

</div>

---

The Hospital AI RAG System is a simple AI-powered question answering system that allows users to search through the patient medical records using natural language. It uses (RAG) to search through stored patient data, and find accurate data.

Instead of manually searching through files, users can simply ask questions like:
"What allergies does John Doe have?" and get an instant AI-generated response based on real stored data.

---

### Patient Data Query System
Ask natural language questions about patient records such as:

- Allergies
- Medications
- Diagnoses
- Medical history
- Lab results

Example:
```bash
python hospital_rag.py --query "What allergies does John Doe have?"
```

---

## How To Run

### First Time Setup

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

In the `.env` file add your Google Gemini API key:

```env
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

Build the knowledge base:

```bash
python hospital_rag.py --ingest
```

---

## Ask a Question

Activate the virtual environment:

```bash
source venv/bin/activate
```

Run a query:

```bash
python hospital_rag.py --query "Your question here"
```

Example:

```bash
python hospital_rag.py --query "What allergies does John Doe have?"
```

---

## Updating Patient Data

Whenever you add, edit, or remove files in the `patient_data` folder, rebuild the knowledge base by running:

```bash
python hospital_rag.py --ingest
```

This updates the vector database with the latest patient information.

---

## Technologies Used

- Python
- LangChain
- Google Gemini API
- Chroma Vector Database
- Hugging Face Embeddings
- Retrieval-Augmented Generation (RAG)

---

## Disclaimer

This project is intended for educational and demonstration purposes only and should not be used as a replacement for professional medical systems.
