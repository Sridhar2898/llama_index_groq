# llama_index_groq

Before starting this session \
get your groq key and put it in the environment variable \
1. For linux: \
   export GROQ_API_KEY="your-groq-api-key"
2. For Windows(command prompt): \
   set GROQ_API_KEY="your-groq-api-key"
3. For Windows(powershell): \
   $env:GROQ_API_KEY="your-groq-api-key" \

Now, \

1. With python environment: \

pip install -r requirements.txt \
streamlit run app.py \

2. Docker \
   docker build -t llama_groq_qa . \
   docker run --network=host -p 8501:8501 --name llama_groq_qa llama_groq_qa
