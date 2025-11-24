from aquiles.client import AquilesRAG
from openai import OpenAI

client = AquilesRAG(api_key="dummy-api-key")

def gen_emb(text):
    client = OpenAI()
    embedding = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

    return embedding.data[0].embedding

#print(client.create_index("docs2", embeddings_dim=1536, dtype="FLOAT32"))


TEXT_TRANSFORMERS = """
1. How Transformers Work in AI
Transformers are a type of neural network architecture that revolutionized the field of AI by introducing the self-attention mechanism, enabling scalable and efficient processing of sequences. First introduced in the 2017 paper “Attention Is All You Need”, they replaced traditional RNNs with fully parallelizable layers that process entire sequences at once.

At a high level, here's how they function:

- Tokenization and embedding: Input text is split into tokens, each converted into a numeric vector.

- Positional encoding: Because Transformers lack recurrence, they add positional information to each token.

- Multi-head self-attention: Each token attends to all others in the sequence to capture context. Multiple attention heads enable learning various relationships.

- Feed-forward layers with residual connections: Interleaved with attention layers to enhance expressivity and gradient flow.

- Stacked encoder-decoder blocks: Encoder builds a contextual representation; decoder uses that to generate outputs for tasks like translation or text generation.

- This architecture excels in tasks such as machine translation, summarization, and large language modeling (GPT, BERT). Its ability to process long-range dependencies in parallel has made it a cornerstone of modern AI
"""

TEXT_SAVE = """
2. A Firefighter Saves a Cat
Captain Elena Rivera raced onto the scene, siren wailing, as neighbors watched in distress. Flames licked the two-story townhouse’s attic windows, smoke billowing into the sky. Inside, little Misty—a fluffy gray cat—was trapped on the topmost beam, mewing in fear.

Clad in full protective gear and oxygen mask, Elena climbed the narrow ladder with practiced speed. “Hang on, Misty,” she whispered reassuringly, securing her footing. Smoke stung her eyes, but her resolve remained steady.

Reaching the attic, Elena found Misty trembling on a charred beam. Gently, she whispered, “It’s okay, sweetheart,” slowly extending a gloved hand. Misty paused uncertainly, then leapt into her arms as Elena wrapped her securely in a protective blanket.

With careful, measured steps, Elena descended. Outside, the owner rushed forward, tears glistening in gratitude. “Thank you, Captain. You saved my baby.”

Misty purred, snuggled on Elena’s chest. Amidst the chaos of the fire, one small life had been given another chance—thanks to courage, training, and a kind heart.
"""

TEXT_SET = """
3. Explaining Set Theory
Set theory is a foundational branch of mathematics that studies sets, which are well-defined collections of distinct objects. A set is typically denoted using curly braces, e.g., {1, 2, 3}.

Key concepts include:

Empty set (denoted ∅ or { }): The set with no elements.

Element membership: a ∈ A means a is an element of set A.

Subset: A ⊆ B if all members of A are also in B. A proper subset is A ⊂ B when A is strictly contained in B.

Union (A ∪ B): The set of elements belonging to A, B, or both.

Intersection (A ∩ B): Elements common to both A and B.

Difference (A / B): Elements in A that are not in B.

Complement: Given a universal set U, the complement of A is U / A.

Power set: The set of all subsets of a set. Cantor’s theorem states that any set’s power set has strictly greater cardinality than the set itself.

Set theory underpins virtually all of modern mathematics—serving as the language for functions, relations, and numbers. The modern axiomatic framework, known as Zermelo–Fraenkel set theory with the Axiom of Choice (ZFC), was introduced in the early 20th century to formalize these concepts and avoid paradoxes like Russell’s paradox 
"""

print("\n Indexing text \n")

#print(client.send_rag(gen_emb, index="docs2", name_chunk="transformers", raw_text=TEXT_TRANSFORMERS))

#print("\n")

#print(client.send_rag(gen_emb, index="docs2", name_chunk="savecat", raw_text=TEXT_SAVE))

print("\n")

#print(client.send_rag(gen_emb, index="docs2", name_chunk="settheory", raw_text=TEXT_SET))

emb = gen_emb("Transformers")

print(f"\n Making a query: Transformers \n")

print(client.query("docs2", emb))