"""
Gradio web interface for RAG-GPT
"""
import gradio as gr
import torch
import tiktoken
from train import GPT, GPTConfig
from rag.rag_retriever import RAGRetriever
import os
os.environ["HF_HUB_DISABLE_WARNING"] = "1"

# Load model
print("Loading model...")
ckpt = torch.load("model_final.pt", map_location='cpu')
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()

rag = RAGRetriever("rag_index/index.faiss", "rag_index/data.json")
enc = tiktoken.get_encoding("gpt2")
print("Model loaded!")

def chat(message, history):
    try:
        # Retrieve from RAG
        results = rag.retrieve(message, top_k=5)
        
        if not results:
            return "I don't have information about that."
        
        # Extract answer
        context = ' '.join([r['text'] for r in results[:3]])
        sentences = context.split('.')
        answer = sentences[0].strip() + '.'
        
        return answer
    
    except Exception as e:
        return f"Error: {e}"

# Create interface
demo = gr.ChatInterface(
    chat,
    title="🤖 RAG-GPT Chat",
    description="Ask me anything! Powered by 200M parameter GPT + RAG.",
    examples=[
        "How do fish breathe?",
        "What is recursion?",
        "Who was the first man on the moon?",
        "How many continents are there?"
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates public link!