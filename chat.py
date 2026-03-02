"""
chat.py — RAG + GPT Hybrid
Based on the original architecture (rag_weight confidence model),
with fixes for: fish/whale bug, strawberry Q&A prefix, GPT repetition.
"""

import os, sys, logging, re, argparse
os.environ["HF_HUB_DISABLE_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
import tiktoken
from train import GPT, GPTConfig
from rag.rag_retriever import RAGRetriever
from rag.config import TOP_K_RETRIEVAL, MAX_CONTEXT_TOKENS

parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="model_domain_tuned_new.pt")
parser.add_argument("--device",      default="cpu")
parser.add_argument("--temperature", type=float, default=0.4)
parser.add_argument("--top_k",       type=int,   default=50)
parser.add_argument("--max_tokens",  type=int,   default=100)
parser.add_argument("--rag_weight",  type=float, default=0.50)
parser.add_argument("--debug",       action="store_true")
args = parser.parse_args()

DEBUG        = args.debug
DISTANCE_THR = 1.5
MAX_CHUNKS   = 5
MIN_CTX_LEN  = 5

# ── model ─────────────────────────────────────────────────
def load_model(path):
    try:
        ck = torch.load(path, map_location=args.device)
        m  = GPT(GPTConfig(**ck["config"]))
        m.load_state_dict(ck["model"])
        m.to(args.device); m.eval()
        return m
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Model load failed: {e}")
        return None

# ── text helpers ──────────────────────────────────────────
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
    # reject pure questions
    if s.endswith('?'): return False
    if re.match(r'^(which|what|who|where|when|how|why)\b', s.lower()): return False
    # reject code
    if re.search(r'\bdef\s+\w+\(|return\s+\w+\s*[;\n]|==|!=|->|=>', s): return False
    # reject data lists
    if s.count(':') >= 2: return False
    if re.match(r'^[-*•]\s', s): return False
    # digit ratio
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
        # near-duplicate check
        dup = False
        for ex in seen:
            w1, w2 = set(norm.split()), set(ex.split())
            if w1 and w2 and len(w1 & w2) / max(len(w1), len(w2)) > 0.85:
                dup = True; break
        if dup: continue
        seen.add(norm)
        valid.append(sent)

    return valid

# ── answer extraction ─────────────────────────────────────
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

    # detect question type
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

    if DEBUG:
        print(f"[DEBUG] Q-type: {qt} | subject_kw: {s_kw}")

    # ── subject stems for fuzzy matching (fish/fishes, play/played) ──
    s_stems = {w[:4] for w in s_kw}

    scored = []
    for sent in sentences:
        sl   = sent.lower()
        sk   = keywords(sent)
        ovlp = len(q_kw & sk)
        sovl = len(s_kw  & sk)
        # stem overlap — handles fish/fishes, breath/breathe
        stem_ovlp = len(s_stems & {w[:4] for w in sk})
        sc   = ovlp * 8

        if qt == 'yesno':
            if re.match(r'^(yes\b|no\b)', sl):
                sc += 50        # "Yes, strawberries are fruits." wins immediately
            if stem_ovlp > 0:
                sc += 15

        elif qt == 'definition':
            if re.search(r'\b(is a|is an|are a|refers to|means|defined as|known as|called)\b', sl):
                sc += 25
            if s_kw:
                first_word = sl.split()[0][:4] if sl else ''
                if any(first_word == w[:4] for w in s_kw):
                    sc += 15    # sentence starts with subject word

        elif qt == 'invention':
            if stem_ovlp == 0:
                sc -= 40        # wrong subject
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
            # Subject must appear in sentence (fish/fishes via 4-char stems)
            if s_stems:
                if stem_ovlp == 0:
                    sc -= 60    # wrong subject entirely
                elif stem_ovlp >= len(s_stems):
                    sc += 35    # all subject stems present — perfect match
                else:
                    sc += stem_ovlp * 12   # partial match
            if re.search(r'\b(through|using|process|by|via|allows|enables)\b', sl):
                sc += 10

        elif qt == 'general':
            if stem_ovlp > 0: sc += 10

        # prefer medium length sentences
        wc = len(sent.split())
        if 8 <= wc <= 40: sc += 6
        elif wc < 5:       sc -= 10

        # penalise list-starters
        if re.match(r'^(for example|such as|e\.g\.|note that|including)\b', sl):
            sc -= 20

        scored.append((sent, sc))

    scored.sort(key=lambda x: x[1], reverse=True)

    if DEBUG:
        print("[DEBUG] Top scored sentences:")
        for s, sc in scored[:5]:
            print(f"  score={sc:4d} | {s[:95]}")

    if not scored or scored[0][1] < 5:
        return None

    # top sentence must share at least 1 keyword with question
    if len(q_kw & keywords(scored[0][0])) < 1 and qt not in ('yesno',):
        if DEBUG: print("[DEBUG] Top sentence has zero question keyword overlap — refuse")
        return None

    # pick up to max_sentences
    best    = scored[0][0]
    result  = [best]
    best_kw = keywords(best)

    for s, sc in scored[1:]:
        if sc < 5: break
        sk2 = keywords(s)
        if len(q_kw & sk2) < 1:                                  continue
        if len(best_kw & sk2) < 1:                               continue
        if len(best_kw & sk2) / max(len(sk2), 1) > 0.75:        continue  # too similar
        result.append(s)
        if len(result) >= max_sentences: break

    answer = ' '.join(s.strip() if s.strip().endswith(('.','!','?')) else s.strip()+'.' for s in result)
    return re.sub(r'\.\.+', '.', re.sub(r'\s+', ' ', answer)).strip()

# ── RAG confidence ────────────────────────────────────────
def rag_confidence(answer, question):
    if not answer: return 0.0
    ql  = question.lower()
    al  = answer.lower()
    wc  = len(answer.split())
    q_kw = keywords(question)
    a_kw = keywords(answer)
    ovlp = len(q_kw & a_kw) / max(len(q_kw), 1)

    # type-specific confidence penalty
    type_ok = 1.0
    if 'invented' in ql or 'created' in ql:
        if not any(w in al for w in ['invented','created','founded','designed','introduced']):
            type_ok = 0.4
    if 'when' in ql.split()[:2]:
        if not re.search(r'\b\d{3,4}\b', al):
            type_ok = 0.3
    if re.match(r'^(is|are|does|do|did|was|were)\b', ql):
        if re.match(r'^(yes\b|no\b)', al.lower()):
            type_ok = 1.5   # direct yes/no is high confidence

    conf = min(1.0, (wc / 25) * 0.3 + ovlp * 0.4 + type_ok * 0.3)
    return conf

# ── GPT generation ────────────────────────────────────────
@torch.no_grad()
def gpt_generate(model, enc, context, question):
    if model is None: return None

    prompt = f"Context: {context[:400]}\n\nQ: {question}\nA:"

    try:
        ids  = enc.encode(prompt)
        idx  = torch.tensor([ids], device=args.device)
        out  = []
        max_gen = min(args.max_tokens, 80)

        for step in range(max_gen):
            logits, _ = model(idx[:, -model.config.block_size:])
            logits = logits[:, -1, :] / args.temperature
            if args.top_k > 0:
                v, _ = torch.topk(logits, args.top_k)
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
                        if DEBUG: print("[DEBUG] GPT: repetition, stopping")
                        out = out[:len(out)//2]
                        break

        if not out: return None
        answer = enc.decode(out).strip()
        answer = clean_text(answer)

        for pfx in ['answer:', 'a:', 'context:', 'q:']:
            if answer.lower().startswith(pfx):
                answer = answer[len(pfx):].strip()

        # repetition check on final output
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
        if DEBUG: print(f"[DEBUG] GPT error: {e}")
        return None

# ── main ──────────────────────────────────────────────────
def main():
    enc   = tiktoken.get_encoding("gpt2")
    model = load_model(args.model)

    print("=" * 60)
    print("🧠 Context Grounded GPT+RAG Hybrid  📖")
    print("=" * 60)

    if model:
        p = sum(x.numel() for x in model.parameters()) / 1e6
        print(f"GPT        : {p:.0f}M params loaded")
    else:
        print("GPT        : not available — RAG only")

    try:
        rag = RAGRetriever("rag_index/index.faiss", "rag_index/data.json")
    except Exception as e:
        print(f"RAG failed : {e}"); return

    print("\nType your question (or 'exit' to quit)")
    if DEBUG: print("Debug ON\n")

    stats = dict(total=0, answered=0, refused=0, rag=0, gpt=0, fallback=0)

    while True:
        print("\n" + "-" * 60)
        q = input("👱 You: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            print("Goodbye! 👋")
            break
        if not q: continue
        stats['total'] += 1

        # retrieve
        try:
            results = rag.retrieve(q, top_k=TOP_K_RETRIEVAL)
        except Exception as e:
            print(f"\nAssistant: Retrieval error: {e}")
            stats['refused'] += 1; continue

        if not results:
            print("\nAssistant: I don't have information about that.")
            stats['refused'] += 1; continue

        if DEBUG:
            print(f"[DEBUG] Retrieved {len(results)} chunks:")
            for i, r in enumerate(results[:5]):
                print(f"  [{i+1}] dist={r['distance']:.3f} | {r['text'][:70]}...")

        filtered = [r for r in results
                    if r['distance'] <= DISTANCE_THR
                    and len(r['text'].split()) >= MIN_CTX_LEN]

        if not filtered:
            print("\nAssistant: I don't have reliable information about that.")
            stats['refused'] += 1; continue

        # deduplicate chunks
        seen_c, unique = set(), []
        for r in sorted(filtered, key=lambda x: x['distance']):
            norm = ' '.join(r['text'].lower().split())[:120]
            if norm not in seen_c:
                seen_c.add(norm); unique.append(r)

        chunks  = unique[:MAX_CHUNKS]
        context = ' '.join(r['text'].strip() for r in chunks)
        toks    = enc.encode(context)
        if len(toks) > MAX_CONTEXT_TOKENS:
            context = enc.decode(toks[:MAX_CONTEXT_TOKENS])

        if DEBUG:
            print(f"[DEBUG] Context: {len(context)} chars, {len(chunks)} chunks")

        # RAG extraction
        rag_answer = extract_answer(q, context)
        rag_conf   = rag_confidence(rag_answer, q)

        if DEBUG:
            print(f"[DEBUG] RAG answer: {rag_answer}")
            print(f"[DEBUG] RAG confidence: {rag_conf:.2f}")

        # Decision
        final = None
        source = None

        if rag_answer and rag_conf >= args.rag_weight:
            # High confidence RAG → use directly
            final  = rag_answer
            source = 'rag'
            if DEBUG: print("[DEBUG] Using RAG (high confidence)")

        elif model is not None and (not rag_answer or rag_conf < 0.60):
            # Only try GPT if the best chunk is actually close (relevant context exists)
            best_dist = chunks[0]['distance'] if chunks else 999
            if best_dist <= 1.0:
                gpt_ans = gpt_generate(model, enc, context, q)
                if gpt_ans:
                    final  = gpt_ans
                    source = 'gpt'
                    if DEBUG: print(f"[DEBUG] Using GPT answer: {gpt_ans}")
            if final is None and rag_answer:
                final  = rag_answer
                source = 'rag'
                if DEBUG: print("[DEBUG] GPT skipped/failed, using RAG")

        elif rag_answer:
            # Medium confidence RAG
            final  = rag_answer
            source = 'rag'
            if DEBUG: print("[DEBUG] Using RAG (medium confidence)")

        if not final:
            # Last resort: first valid sentence from best chunk
            for r in chunks:
                sents = extract_sentences(r['text'])
                if sents:
                    final  = sents[0]
                    source = 'fallback'
                    if DEBUG: print("[DEBUG] Using fallback first sentence")
                    break

        if not final:
            print("\nAssistant: I don't have reliable information about that.")
            stats['refused'] += 1; continue

        if not final[-1] in '.!?': final += '.'
        print(f"\n🤖 Assistant: {final}")
        stats['answered'] += 1
        stats[source]     += 1

if __name__ == "__main__":
    main()