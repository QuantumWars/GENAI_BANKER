import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

class RAGPipeline:
    def __init__(self, name, auth_token, model_path, cache_dir='./model/'):
        self.name = name
        self.auth_token = auth_token
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, cache_dir=self.cache_dir, use_auth_token=self.auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, cache_dir=self.cache_dir, use_auth_token=self.auth_token, torch_dtype=torch.float16, rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)

    def setup_prompt(self, prompt):
        self.prompt = prompt
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_output(self):
        self.output = self.model.generate(**self.inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
        self.output_text = self.tokenizer.decode(self.output[0], skip_special_tokens=True)

    def setup_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    def setup_llm(self):
        self.llm = HuggingFaceLLM(context_window=4096, max_new_tokens=256, system_prompt=self.system_prompt, query_wrapper_prompt=self.query_wrapper_prompt, model=self.model, tokenizer=self.tokenizer)

    def setup_embeddings(self):
        self.embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    def setup_service_context(self):
        self.service_context = ServiceContext.from_defaults(chunk_size=1024, llm=self.llm, embed_model=self.embeddings)
        set_global_service_context(self.service_context)

    def load_documents(self, file_path):
        self.PyMuPDFReader = download_loader("PyMuPDFReader")
        self.loader = self.PyMuPDFReader()
        self.documents = self.loader.load(file_path=Path(file_path), metadata=True)

    def create_index(self):
        self.index = VectorStoreIndex.from_documents(self.documents)

    def setup_query_engine(self):
        self.query_engine = self.index.as_query_engine()

    def query(self, query_str):
        self.response = self.query_engine.query(query_str)
        return self.response

pipeline = RAGPipeline(name="meta-llama/Llama-2-70b-chat-hf", auth_token="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", model_path='./model/')
pipeline.setup_prompt("### User:What is the fastest car in the world and how much does it cost? ### Assistant:")
pipeline.generate_output()
pipeline.setup_system_prompt("""[INST] <>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of 
the company.<>
""")
pipeline.setup_llm()
pipeline.setup_embeddings()
pipeline.setup_service_context()
pipeline.load_documents('./data/annualreport.pdf')
pipeline.create_index()
pipeline.setup_query_engine()
response = pipeline.query("what was the FY2022 return on equity?")
print(response)
