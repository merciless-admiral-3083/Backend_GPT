
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from train import GPT, GPTConfig
from rag.rag_retriever import RAGRetriever
import os
import time
import re

os.environ["HF_HUB_DISABLE_WARNING"] = "1"

app = FastAPI(
    title="RAG-GPT API",
    description="Backend for RAG-GPT chat system",
    version="1.0.0"
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISTANCE_THR = 1.5
MAX_CHUNKS = 5
MIN_CTX_LEN = 5
TOP_K_RETRIEVAL = 10
MAX_CONTEXT_TOKENS = 800
MODEL_PATH = os.getenv("MODEL_PATH", "model_domain_tuned_new.pt")
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model_domain_tuned_new.pt")
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION")


def resolve_checkpoint_path():
    if os.path.exists(MODEL_PATH):
        print(f" Using local model checkpoint: {MODEL_PATH}")
        return MODEL_PATH

    if not HF_MODEL_REPO_ID:
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Set HF_MODEL_REPO_ID to download from Hugging Face."
        )

    print(f" Downloading model from Hugging Face: {HF_MODEL_REPO_ID}/{HF_MODEL_FILENAME}")
    return hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
        revision=HF_MODEL_REVISION,
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )

print(" Loading model...")
model = None
try:
    checkpoint_path = resolve_checkpoint_path()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(" Model loaded!")
except Exception as e:
    print(f" Model unavailable. Running in RAG-only mode. Reason: {e}")

rag = RAGRetriever("rag_index/index.faiss", "rag_index/data.json")
enc = tiktoken.get_encoding("gpt2")
print(" RAG loaded!")

class ChatRequest(BaseModel):
    message: str
    rag_weight: float = 0.50  
    max_results: int = 3

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source: str
    response_time: float
    retrieved_facts: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_params: str
    rag_facts: int



def clean_text(text):
    if not text: return ""
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if re.match(r'^[\s•\-*\d.]+$', line): continue
        if line.count('•') > 3: continue
        lines.append(line)
    text = ' '.join(lines)
    text = re.sub(r'[•●○]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def strip_qa_prefix(s):
    """Remove Q:/A:/Question:/Answer: prefixes."""
    return re.sub(r'^(Q|A|Question|Answer)\s*:\s*', '', s,
                  flags=re.IGNORECASE).strip()

def is_valid_sentence(s):
    s = s.strip()
    if len(s) < 8: return False
    if sum(c.isalpha() for c in s) < 6: return False
    if s.endswith('?'): return False
    if re.match(r'^(which|what|who|where|when|how|why)\b', s.lower()): return False
    if re.search(r'\bdef\s+\w+\(|return\s+\w+\s*[;\n]|==|!=|->|=>', s): return False
    if s.count(':') >= 2: return False
    if re.match(r'^[-*•]\s', s): return False
    if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.6: return False
    return True

def extract_sentences(text):
    """Split text into valid sentences, stripping Q:/A: prefixes."""
    text = clean_text(text)
    if not text: return []

    parts = re.split(r'([.!?])\s+', text)
    sentences = []
    current = ""
    for part in parts:
        if part in '.!?':
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part
    if current.strip():
        if current.strip()[-1] not in '.!?':
            current += '.'
        sentences.append(current.strip())

    valid, seen = [], set()
    for sent in sentences:
        sent = strip_qa_prefix(sent.strip())
        if not sent: continue
        if not sent[-1] in '.!?': sent += '.'
        if not is_valid_sentence(sent): continue
        norm = ' '.join(sent.lower().split())
        if norm in seen: continue
        dup = False
        for ex in seen:
            w1, w2 = set(norm.split()), set(ex.split())
            if w1 and w2 and len(w1 & w2) / max(len(w1), len(w2)) > 0.85:
                dup = True; break
        if dup: continue
        seen.add(norm)
        valid.append(sent)

    return valid

STOP = {
    'the','a','an','is','are','was','were','in','of','to','and','or',
    'for','it','this','that','be','by','as','at','on','its','with',
    'from','have','has','had','will','would','could','should','be',
    'not','no','but','so','if','then','than','there','their','they',
    'about','into','each','some','any','all','been','being','very',
    'just','even','only','both','while','during','before','after',
    'do','does','did','can','may','might','also','what','who','where',
    'when','how','why','which','am','we','you','i','my','your','our'
}

def keywords(text, extra_stop=None):
    stops = STOP | (extra_stop or set())
    return {w for w in re.findall(r'\b[a-z]{3,}\b', text.lower())
            if w not in stops}

def extract_answer(question, context, max_sentences=2):
    sentences = extract_sentences(context)
    if not sentences: return None

    ql    = question.lower().strip().rstrip('?.')
    q_kw  = keywords(question)
    stop  = {'game','play','sport','ball'} if re.search(r'\bwho\s+(invented|created|founded)\b', ql) else set()
    s_kw  = keywords(ql, extra_stop=stop | {'what','who','where','when','how','why','which',
                                              'is','are','was','were','do','does','did',
                                              'define','explain','describe','capital','tell'})

    if re.match(r'^(is|are|was|were|does|do|did|has|have|can|could|will|would)\b', ql):
        qt = 'yesno'
    elif re.match(r'^(who\s+invented|who\s+created|who\s+founded|who\s+built|who\s+made)\b', ql):
        qt = 'invention'
    elif re.match(r'^(what\s+is|what\s+are|define|explain|describe)\b', ql):
        qt = 'definition'
    elif re.match(r'^when\b', ql):
        qt = 'when'
    elif re.match(r'^where\b', ql):
        qt = 'where'
    elif re.match(r'^how\b', ql):
        qt = 'how'
    else:
        qt = 'general'

    
    s_stems = {w[:4] for w in s_kw}

    scored = []
    for sent in sentences:
        sl   = sent.lower()
        sk   = keywords(sent)
        ovlp = len(q_kw & sk)
        sovl = len(s_kw  & sk)
       
        stem_ovlp = len(s_stems & {w[:4] for w in sk})
        sc   = ovlp * 8

        if qt == 'yesno':
            if re.match(r'^(yes\b|no\b)', sl):
                sc += 50       
            if stem_ovlp > 0:
                sc += 15

        elif qt == 'definition':
            if re.search(r'\b(is a|is an|are a|refers to|means|defined as|known as|called)\b', sl):
                sc += 25
            if s_kw:
                first_word = sl.split()[0][:4] if sl else ''
                if any(first_word == w[:4] for w in s_kw):
                    sc += 15   

        elif qt == 'invention':
            if stem_ovlp == 0:
                sc -= 40       
            if re.search(r'\b(invented|created|founded|designed|introduced|developed)\b', sl):
                sc += 30 if stem_ovlp > 0 else -15

        elif qt == 'when':
            if re.search(r'\b\d{3,4}\b', sl):
                sc += 25 + (15 if stem_ovlp > 0 else 0)
            else:
                sc -= 20

        elif qt == 'where':
            if re.search(r'\b(located|situated|found in|lives in|native to|born in)\b', sl):
                sc += 25

        elif qt == 'how':
            if s_stems:
                if stem_ovlp == 0:
                    sc -= 60    
                elif stem_ovlp >= len(s_stems):
                    sc += 35   
                else:
                    sc += stem_ovlp * 12   
            if re.search(r'\b(through|using|process|by|via|allows|enables)\b', sl):
                sc += 10

        elif qt == 'general':
            if stem_ovlp > 0: sc += 10

        wc = len(sent.split())
        if 8 <= wc <= 40: sc += 6
        elif wc < 5:       sc -= 10

        if re.match(r'^(for example|such as|e\.g\.|note that|including)\b', sl):
            sc -= 20

        scored.append((sent, sc))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < 5:
        return None

    if len(q_kw & keywords(scored[0][0])) < 1 and qt not in ('yesno',):
        return None

    best    = scored[0][0]
    result  = [best]
    best_kw = keywords(best)

    for s, sc in scored[1:]:
        if sc < 5: break
        sk2 = keywords(s)
        if len(q_kw & sk2) < 1:                                  continue
        if len(best_kw & sk2) < 1:                               continue
        if len(best_kw & sk2) / max(len(sk2), 1) > 0.75:        continue  
        result.append(s)
        if len(result) >= max_sentences: break

    answer = ' '.join(s.strip() if s.strip().endswith(('.','!','?')) else s.strip()+'.' for s in result)
    return re.sub(r'\.\.+', '.', re.sub(r'\s+', ' ', answer)).strip()

def rag_confidence(answer, question):
    if not answer: return 0.0
    ql  = question.lower()
    al  = answer.lower()
    wc  = len(answer.split())
    q_kw = keywords(question)
    a_kw = keywords(answer)
    ovlp = len(q_kw & a_kw) / max(len(q_kw), 1)

    type_ok = 1.0
    if 'invented' in ql or 'created' in ql:
        if not any(w in al for w in ['invented','created','founded','designed','introduced']):
            type_ok = 0.4
    if 'when' in ql.split()[:2]:
        if not re.search(r'\b\d{3,4}\b', al):
            type_ok = 0.3
    if re.match(r'^(is|are|does|do|did|was|were)\b', ql):
        if re.match(r'^(yes\b|no\b)', al.lower()):
            type_ok = 1.5   

    conf = min(1.0, (wc / 25) * 0.3 + ovlp * 0.4 + type_ok * 0.3)
    return conf

@torch.no_grad()
def gpt_generate(context, question):
    prompt = f"Context: {context[:400]}\n\nQ: {question}\nA:"

    try:
        ids  = enc.encode(prompt)
        idx  = torch.tensor([ids], device='cpu')
        out  = []
        max_gen = 80

        for step in range(max_gen):
            logits, _ = model(idx[:, -model.config.block_size:])
            logits = logits[:, -1, :] / 0.4 
            v, _ = torch.topk(logits, 50)  
            logits[logits < v[:, [-1]]] = -float("inf")
            tok = torch.multinomial(F.softmax(logits, -1), 1)
            out.append(tok.item())
            idx = torch.cat([idx, tok], dim=1)

            if step > 5 and step % 3 == 0:
                decoded = enc.decode(out).strip()
                if len(decoded) > 20 and decoded[-1] in '.!?':
                    break
                words = decoded.lower().split()
                if len(words) >= 12:
                    if words[-6:] == words[-12:-6]:
                        out = out[:len(out)//2]
                        break

        if not out: return None
        answer = enc.decode(out).strip()
        answer = clean_text(answer)

        for pfx in ['answer:', 'a:', 'context:', 'q:']:
            if answer.lower().startswith(pfx):
                answer = answer[len(pfx):].strip()

        words = answer.lower().split()
        if len(words) >= 8:
            for i in range(len(words) - 6):
                if ' '.join(words[i:i+3]) in ' '.join(words[i+3:]):
                    answer = ' '.join(words[:i+3])
                    break

        if not answer or len(answer.split()) < 5: return None
        if sum(c.isalpha() for c in answer) / max(len(answer), 1) < 0.5: return None
        if not answer[-1] in '.!?': answer += '.'
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer

    except Exception as e:
        print(f"GPT error: {e}")
        return None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    try:
        results = rag.retrieve(request.message, top_k=TOP_K_RETRIEVAL)
        
        if not results:
            return ChatResponse(
                answer="I don't have information about that.",
                confidence=0.0,
                source="none",
                response_time=time.time() - start_time,
                retrieved_facts=[]
            )
        
        filtered = [r for r in results
                    if r['distance'] <= DISTANCE_THR
                    and len(r['text'].split()) >= MIN_CTX_LEN]
        
        if not filtered:
            return ChatResponse(
                answer="I don't have reliable information about that.",
                confidence=0.0,
                source="none",
                response_time=time.time() - start_time,
                retrieved_facts=[]
            )
        
        seen_c, unique = set(), []
        for r in sorted(filtered, key=lambda x: x['distance']):
            norm = ' '.join(r['text'].lower().split())[:120]
            if norm not in seen_c:
                seen_c.add(norm)
                unique.append(r)

        chunks = unique[:MAX_CHUNKS]
        context = ' '.join(r['text'].strip() for r in chunks)
        toks = enc.encode(context)
        if len(toks) > MAX_CONTEXT_TOKENS:
            context = enc.decode(toks[:MAX_CONTEXT_TOKENS])
        
        rag_answer = extract_answer(request.message, context)
        rag_conf = rag_confidence(rag_answer, request.message)
        
        final = None
        source = None
        
        if rag_answer and rag_conf >= request.rag_weight:
            final = rag_answer
            source = 'rag'
        
        elif model is not None and (not rag_answer or rag_conf < 0.60):
            best_dist = chunks[0]['distance'] if chunks else 999
            if best_dist <= 1.0:
                gpt_ans = gpt_generate(context, request.message)
                if gpt_ans:
                    final = gpt_ans
                    source = 'gpt'
            if final is None and rag_answer:
                final = rag_answer
                source = 'rag'
        
        elif rag_answer:
            final = rag_answer
            source = 'rag'
        
        if not final:
            for r in chunks:
                sents = extract_sentences(r['text'])
                if sents:
                    final = sents[0]
                    source = 'fallback'
                    break
        
        if not final:
            return ChatResponse(
                answer="I don't have reliable information about that.",
                confidence=0.0,
                source="none",
                response_time=time.time() - start_time,
                retrieved_facts=[]
            )
        
        if not final[-1] in '.!?': 
            final += '.'
        
        return ChatResponse(
            answer=final,
            confidence=rag_conf if source == 'rag' else 0.7,
            source=source,
            response_time=time.time() - start_time,
            retrieved_facts=[
                {"text": r['text'][:100] + "...", "distance": r['distance']} 
                for r in chunks[:3]
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_params="166M",
        rag_facts=len(getattr(rag, "data", []))
    )

@app.get("/")
async def root():
    return {
        "message": "RAG-GPT API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }