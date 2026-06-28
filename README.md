- Hospital AI RAG System -


The first time your run it do 
source venv/bin/activate
pip install -r requirements.txt
Edit the .env to add your api key GOOGLE_API_KEY=YOUR_API_KEY_HERE
python hospital_rag.py --ingest


Everytime you want to ask a question

- Activate the virtual environment:  source venv/bin/activate

- Ask your question: python hospital_rag.py --query "Your question here"
exmp:  python hospital_rag.py --query "What allergies does John Doe have?"



When you add new info you have to do --ingest 
--------------------------
Run this again if you add or change files in the patient_data folder:
python hospital_rag.py --ingest


