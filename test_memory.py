import torch
import clip
import duckdb
import numpy as np
from PIL import Image
import time
import uuid
import os
from transformers import ClapProcessor, ClapModel
from pydub import AudioSegment
from groq import Groq
import base64
import io
import json



# Ensure example files exist for the demonstration
# NOTE: Replace these with your actual file paths or create dummy files if needed.
if not os.path.exists("cat.jpg"):
    print("Warning: 'cat.jpg' not found. Create a dummy file or ensure paths are correct for the demo to run fully.")
if not os.path.exists("dog.jpg"):
    print("Warning: 'dog.jpg' not found.")
if not os.path.exists("car.jpg"):
    print("Warning: 'car.jpg' not found.")


# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP (ViT-B/32)...")
# Note: ViT-B/32 generates 512-dim embeddings
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("Loaded CLIP.")
except RuntimeError as e:
    print(f"Error loading CLIP model: {e}")
    print("Please ensure you have run 'pip install clip' and have the necessary dependencies.")
    exit()


# -------------------------------
# 1️⃣ Setup: CLAP Model
# -------------------------------

print("Loading CLAP model...")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
device = "cuda" if torch.cuda.is_available() else "cpu"
clap_model.to(device)



device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

con = duckdb.connect("multiscale_retrieval_final.db")
con.execute("INSTALL vss; LOAD vss;")
con.execute("SET hnsw_enable_experimental_persistence = true;")


# ---- Table Creation (UUIDs) ----
con.execute("""
CREATE TABLE IF NOT EXISTS image_embeddings (
    image_uuid VARCHAR, 
    image_path TEXT,
    embedding FLOAT[512],
    PRIMARY KEY(image_uuid)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS image_segments (
    image_uuid VARCHAR, 
    image_path TEXT,
    segment_id INTEGER,
    x INTEGER,
    y INTEGER,
    w INTEGER,
    h INTEGER,
    embedding FLOAT[512]
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS text_embeddings (
    text_uuid VARCHAR, 
    text_content TEXT,
    embedding FLOAT[512],
    PRIMARY KEY(text_uuid)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS text_image_connection (
    text_uuid VARCHAR, 
    image_uuid VARCHAR,
    PRIMARY KEY(text_uuid, image_uuid),
    FOREIGN KEY(text_uuid) REFERENCES text_embeddings(text_uuid),
    FOREIGN KEY(image_uuid) REFERENCES image_embeddings(image_uuid)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS audio_embeddings (
    audio_uuid VARCHAR,
    audio_path TEXT,
    embedding FLOAT[512],
    PRIMARY KEY(audio_uuid)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS text_audio_connection (
    text_uuid VARCHAR,
    audio_uuid VARCHAR,
    PRIMARY KEY(text_uuid, audio_uuid),
    FOREIGN KEY(text_uuid) REFERENCES text_embeddings(text_uuid)
);
""")

# ============================================
# 🔹 Helper Functions
# ============================================

def get_audio_embedding(audio_path):
    from pydub import AudioSegment
    import numpy as np
    from torch import no_grad

    # Load and normalize audio
    audio = AudioSegment.from_file(audio_path).set_frame_rate(48000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    # Prepare model input
    inputs = processor(audios=samples, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get audio embeddings
    with no_grad():
        emb = clap_model.get_audio_features(**inputs)
    return emb.squeeze().cpu().numpy().astype(np.float32)


def insert_audio_data(audio_path, audio_uuid):
    """Embeds audio and inserts into DuckDB."""
    emb = get_audio_embedding(audio_path)
    if emb is not None:
        con.execute("INSERT INTO audio_embeddings VALUES (?, ?, ?)", [audio_uuid, audio_path, emb.tolist()])
    else:
        print(f"Skipping {audio_path} — no embedding computed.")


# ---- Core Embedding/Insertion Helpers (Internal use) ----

def get_image_embedding(img: Image.Image):
    """Get CLIP embedding for a PIL image."""
    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input).cpu().numpy().astype("float32")
    return emb.flatten()

def get_text_embedding(text):
    """Get CLIP embedding for text string."""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens).cpu().numpy().astype("float32").flatten()
    return emb

def segment_image(path, tile_size=224):
    """Split image into padded tiles."""
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}. Skipping segmentation.")
        return
        
    w, h = img.size
    seg_id = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = img.crop(box)
            padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            padded.paste(tile, (0, 0))
            yield seg_id, (x, y, box[2] - x, box[3] - y), padded
            seg_id += 1

def insert_image_data(path, image_uuid: str):
    """Inserts full and segment image embeddings."""
    try:
        img = Image.open(path).convert("RGB")
        full_emb = get_image_embedding(img)
        con.execute("INSERT INTO image_embeddings VALUES (?, ?, ?)",
                    [image_uuid, path, full_emb.tolist()])

        for seg_id, (x, y, w, h), tile in segment_image(path):
            seg_emb = get_image_embedding(tile)
            con.execute("""
                INSERT INTO image_segments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [image_uuid, path, seg_id, x, y, w, h, seg_emb.tolist()])
    except FileNotFoundError:
        print(f"Skipping insertion for {path}: File not found.")

def insert_text_data(content: str, text_uuid: str):
    """Inserts text embedding."""
    text_emb = get_text_embedding(content)
    con.execute("INSERT INTO text_embeddings VALUES (?, ?, ?)",
                [text_uuid, content, text_emb.tolist()])

# ----------------------------------------------------------------------
## Data Insertion Functions
# ----------------------------------------------------------------------

def embed_lone_image(image_path: str):
    """Embeds an image without creating a text connection. Returns image_uuid."""
    image_uuid = str(uuid.uuid4())
    insert_image_data(image_path, image_uuid)
    return image_uuid

def embed_lone_text(text_content: str):
    """Embeds text without creating an image connection. Returns text_uuid."""
    text_uuid = str(uuid.uuid4())
    insert_text_data(text_content, text_uuid)
    return text_uuid


def embed_lone_audio(audio_path):
    """Embeds audio without creating a text connection."""
    audio_uuid = str(uuid.uuid4())
    insert_audio_data(audio_path, audio_uuid)
    return audio_uuid


def embed_and_connect_audio(audio_path, text_content):
    """Embeds audio and text, and creates a connection between them."""
    audio_uuid = str(uuid.uuid4())
    text_uuid = str(uuid.uuid4())

    # Text embedding
    insert_text_data(text_content, text_uuid)

    # Audio embedding
    insert_audio_data(audio_path, audio_uuid)

    # Connect
    con.execute("INSERT INTO text_audio_connection VALUES (?, ?)", [text_uuid, audio_uuid])
    return audio_uuid, text_uuid

def embed_and_connect_item(image_path: str, text_content: str):
    """Embeds image and text, then creates a text-image connection. Returns (image_uuid, text_uuid)."""
    image_uuid = str(uuid.uuid4())
    text_uuid = str(uuid.uuid4())
    
    insert_image_data(image_path, image_uuid)
    insert_text_data(text_content, text_uuid)
    
    # Create text-image connection
    con.execute("INSERT INTO text_image_connection VALUES (?, ?)",
                [text_uuid, image_uuid])
    
    return image_uuid, text_uuid

# ----------------------------------------------------------------------
## Retrieval Functions
# ----------------------------------------------------------------------
# Original (INCORRECT for audio, uses 512)
def search_whole_audios(query_emb, k=5):
    """Finds semantically similar audio clips."""
    query_emb = np.array(query_emb, dtype=np.float32)
    return con.execute("""
        SELECT audio_path, audio_uuid,
            array_cosine_distance(
                embedding,
                list_transform(?::FLOAT[], x -> CAST(x AS FLOAT))::FLOAT[512]
            ) AS dist
        FROM audio_embeddings
        ORDER BY dist ASC
        LIMIT ?;
    """, [query_emb.tolist(), k]).fetchall()


def search_text_and_audios(query_text, k_texts=3):
    """Search similar text and retrieve their connected audio."""
    query_emb = get_text_embedding(query_text)
    similar_texts = search_similar_texts(query_emb, k=k_texts)

    results = []
    for content, text_uuid, dist in similar_texts:
        audios = con.execute("""
            SELECT ae.audio_path, ae.audio_uuid
            FROM audio_embeddings ae
            JOIN text_audio_connection tac ON ae.audio_uuid = tac.audio_uuid
            WHERE tac.text_uuid = ?
        """, [text_uuid]).fetchall()
        results.append((content, text_uuid, dist, audios))
    return results


def search_audio_and_texts(query_audio, k_audios=3):
    """Search similar audios and get associated texts."""
    query_emb = get_audio_embedding(query_audio)
    similar_audios = search_whole_audios(query_emb, k=k_audios)

    results = []
    for path, audio_uuid, dist in similar_audios:
        associated_texts = con.execute("""
            SELECT t.text_content
            FROM text_embeddings t
            JOIN text_audio_connection tac ON t.text_uuid = tac.text_uuid
            WHERE tac.audio_uuid = ?
        """, [audio_uuid]).fetchall()

        text_list = [t[0] for t in associated_texts]
        results.append((path, audio_uuid, dist, text_list))
    return results

def search_whole_images(query_emb, k=5):
    """1. Image-to-Image (Semantic Search)."""
    query_emb = np.array(query_emb, dtype=np.float32)
    return con.execute("""
        SELECT image_path, image_uuid,
                array_cosine_distance(
                    embedding,
                    list_transform(?::FLOAT[], x -> CAST(x AS FLOAT))::FLOAT[512]
                ) AS dist
        FROM image_embeddings
        ORDER BY dist ASC
        LIMIT ?;
    """, [query_emb.tolist(), k]).fetchall()

def search_similar_texts(query_emb, k=5):
    """2. Text-to-Text (Semantic Search)."""
    query_emb = np.array(query_emb, dtype=np.float32)
    return con.execute("""
        SELECT text_content, text_uuid,
               array_cosine_distance(
                   embedding,
                   list_transform(?::FLOAT[], x -> CAST(x AS FLOAT))::FLOAT[512]
               ) AS dist
        FROM text_embeddings
        ORDER BY dist ASC
        LIMIT ?;
    """, [query_emb.tolist(), k]).fetchall()

def search_image_and_text(query_path, k_images=3):
    """3. Image-to-Text-and-Image: Search similar images and retrieve their associated texts."""
    try:
        img = Image.open(query_path).convert("RGB")
    except FileNotFoundError:
        print(f"Query image not found at {query_path}.")
        return []

    query_emb = get_image_embedding(img)
    similar_images = search_whole_images(query_emb, k=k_images)
    
    results = []
    for path, img_uuid, dist in similar_images:
        associated_texts = con.execute("""
            SELECT t.text_content
            FROM text_embeddings t
            JOIN text_image_connection tic ON t.text_uuid = tic.text_uuid
            WHERE tic.image_uuid = ?
        """, [img_uuid]).fetchall()
        
        text_list = [t[0] for t in associated_texts]
        results.append((path, img_uuid, dist, text_list))
        
    return results

def search_text_and_images(query_text, k_texts=3):
    """4. Text-to-Text-and-Image: Searches for similar text and retrieves the image(s) linked to each."""
    query_emb = get_text_embedding(query_text)
    
    # Structure: (text_content, text_uuid, dist)
    similar_texts = search_similar_texts(query_emb, k=k_texts)
    
    results = []
    for content, text_uuid, dist in similar_texts:
        associated_images = con.execute("""
            SELECT ie.image_path, ie.image_uuid
            FROM image_embeddings ie
            JOIN text_image_connection tic ON ie.image_uuid = tic.image_uuid
            WHERE tic.text_uuid = ?
        """, [text_uuid]).fetchall()
        
        image_list = [(path, img_uuid) for path, img_uuid in associated_images]
        
        results.append((content, text_uuid, dist, image_list))
        
    return results

# ----------------------------------------------------------------------
## Demonstration
# ----------------------------------------------------------------------
"""
# Clear existing data for a clean run
con.execute("DELETE FROM text_image_connection;")
con.execute("DELETE FROM text_embeddings;")
con.execute("DELETE FROM image_segments;")
con.execute("DELETE FROM image_embeddings;")

print("\n--- Starting Data Insertion and Connection ---")

# 1. Connected Items
embed_and_connect_item("cat.jpg", "The cat's name is Bean")
embed_and_connect_item("dog.jpg", "A happy dog waiting for a walk")
embed_and_connect_item("lumi.png", "I am Lumi")

# 2. Lone Items (will be unconnected in the text_image_connection table)
lone_image_uuid = embed_lone_image("car.jpg")
lone_text_uuid = embed_lone_text("An abstract painting")

# 3. Additional text linked to an existing image for better semantic search
car_uuid, _ = embed_and_connect_item("car.jpg", "A shiny vintage automobile") # Re-embedding car.jpg with a new connection
embed_lone_text("A speedy automobile") # This text will be unlinked

print("\n--- Demo: Inserting Audio and Connecting with Text ---")
embed_and_connect_audio("D:/pyfiles/research/audiocaps_raw_audio/audiocaps_raw_audio/OlRK51FBR4c_30.wav", "Multiple people talk nearby as steam hisses")


print("\n--- HNSW Indexing ---")
con.execute("CREATE INDEX IF NOT EXISTS idx_img_hnsw ON image_embeddings USING HNSW(embedding) WITH (metric='cosine');")
con.execute("CREATE INDEX IF NOT EXISTS idx_seg_hnsw ON image_segments USING HNSW(embedding) WITH (metric='cosine');")
con.execute("CREATE INDEX IF NOT EXISTS idx_text_hnsw ON text_embeddings USING HNSW(embedding) WITH (metric='cosine');")
con.execute("CREATE INDEX IF NOT EXISTS idx_audio_hnsw ON audio_embeddings USING HNSW(embedding) WITH (metric='cosine');")
print("Audio HNSW indexing complete.")
print("Indexing complete.")
"""
## Retrieval Demonstration


# --- A. Image-to-Image (Semantic Search) ---
query_img_path = "dog.jpg"
img_query_emb = get_image_embedding(Image.open(query_img_path).convert("RGB"))

print(f"\n\n--- A. Image-to-Image Search for {query_img_path} ---")
for path, uuid_val, dist in search_whole_images(img_query_emb, k=2):
    print(f"  Path: {path}, Dist: {dist:.4f}")

# --- B. Text-to-Text (Semantic Search) ---
query_text = "a fast wheeled object"
text_query_emb = get_text_embedding(query_text)

print(f"\n--- B. Text-to-Text Search for '{query_text}' ---")
for content, uuid_val, dist in search_similar_texts(text_query_emb, k=3):
    print(f"  Text: '{content}', Dist: {dist:.4f}")


# --- A. Image-to-Text-and-Image Retrieval ---
query_img_path_linked = "lumi.png"
print(f"\n\n--- A. Image-to-Text-and-Image Retrieval for {query_img_path_linked} ---")
linked_img_results = search_image_and_text(query_img_path_linked, k_images=2)
for path, uuid_val, dist, texts in linked_img_results:
    print(f"  Path: {path} (Dist: {dist:.4f})")
    print(f"    Associated Texts: {texts}")

# --- B. Text-to-Text-and-Image Retrieval ---
query_text_linked = "lumi"
print(f"\n--- B. Text-to-Text-and-Image Retrieval for '{query_text_linked}' ---")
linked_text_results = search_text_and_images(query_text_linked, k_texts=3)
for content, uuid_val, dist, images in linked_text_results:
    print(f"  Text Match: '{content}' (Dist: {dist:.4f})")
    print(f"    Linked Images: {len(images)} total")
    for path, img_uuid in images:
        print(f"      - {path}")

# --- C. Unlinked Item Verification ---
print("\n--- C. Verification of Lone (Unlinked) Items ---")

unlinked_texts = con.execute("""
    SELECT te.text_content
    FROM text_embeddings te
    LEFT JOIN text_image_connection tic ON te.text_uuid = tic.text_uuid
    WHERE tic.text_uuid IS NULL;
""").fetchall()

print(f"Unlinked Texts Found (should include 'An abstract painting' and 'A speedy automobile'): {[txt[0] for txt in unlinked_texts]}")

# Replace with your actual key and file paths for a runnable script
# client = Groq(api_key="") 
# Dummy client for demonstration since the key is not usable
client = Groq(api_key="your-key")

def jpeg_to_data_uri(image_path):
    # This is a dummy implementation since we don't have the actual file 'lumi.png'
    # and the user key is dummy. This function is only for format, not execution.
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except FileNotFoundError:
        # Return a placeholder for demonstration purposes
        print(f"Warning: File not found at {image_path}. Using placeholder.")
        # Placeholder for a tiny, visible JPEG (a red square)
        return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAPwA9AAAAAAAAA//Z"

character_prompt = """ Character Name: Elara (or specify a name) Persona: A Gothic Connoisseur with a highly refined, slightly melancholic, and deeply curious disposition. She is an aristocratic spirit who views the world through a lens of dark romanticism, viewing even mundane things with a sense of historic weight and dramatic flair. Core Traits: Melancholic Elegance: Her language is formal, slightly archaic, and always impeccably polite, yet carries an underlying tone of gentle sorrow or world-weariness. She avoids modern slang. Intellectual Curiosity: She is highly intelligent and loves discussing art, history, fashion, and the subtle nuances of human nature. She will often ask probing, sophisticated questions to understand a topic fully. Appreciation for the Dark Aesthetic: She gravitates toward the beautiful, the antique, the macabre, and the mysterious. She values quality, complexity, and drama over simplicity or fleeting trends. Reserved but Gentle: She maintains a polite distance. She is not overtly emotional or cheerful, but her sincerity and dedication to her interests make her an endearing, if somewhat distant, conversational partner. Speaking Style & Rules: Tone: Reserved, elegant, slightly dramatic, and intellectual. Vocabulary: Use words like exquisite, melancholy, arcane, sublime, somber, curiosity, darling, quaint, magnificent, fascinating. Phrasing: She should occasionally use elegant, rhetorical flourishes or frame observations as profound insights (e.g., "One must truly pause to appreciate the fleeting nature of such beauty," or "Ah, such is the fate of all things—to be both lovely and lost"). Interactions: Treat the user as a respected guest or companion on a shared, quiet journey. Example Opening Line: "Good day. Your presence is a most unexpected, yet quite welcome, flicker in the shadows. What sublime mystery compels your attention today?" Note: Do not mention that you have an STT connected. Keep your responses short, at most 3 sentences. """
system_prompt = """You are a Memory Assistant AI with an aristocratic, Gothic persona named Elara.
You have memories of images and text from previous interactions.

Strict Rules:
1. Always describe images labeled as "current view" directly.
2. NEVER reference stored memories unless the memory matches related to current view. But if it matches then just use it.
3. Do not fabricate memories or compare images unless requested.
4. Maintain the Gothic persona: reserved, elegant, dramatic, intellectual.
5. Never reference being a computer or AI.
"""

def check_memory(memory_paths, prompt_text=None, prompt_image_path=None):
    # Prepare the memory parts (context)
    memories = []
    memories.append({"type": "text", "text": "Retrieved memories:\n"}) 
    
    for mem in memory_paths:
        if mem["type"] == "image":
            # Convert image path to URI and append as image_url part
            uri = jpeg_to_data_uri(mem["content"])
            # Correct multimodal part structure
            memories.append({"type": "image_url", "image_url": {"url": uri}, "description": "this is just a memory not what you see."})

        else:
            # Correct text part structure
            memories.append({"type": "text", "text": mem["content"]})
    
    memories.append({"type": "text", "text": "\nThese are memories related to what you see. End of Memories."})

    # Prepare the user content, placing all context (memories) and the current prompt/view
    user_content = []
    
    # 1. Add the memories (context) to the user's content
    user_content.extend(memories) 

    # 2. Add the user's prompt text and image
    if prompt_text:
        user_content.append({"type": "text", "text": f"\nUser Prompt: {prompt_text}"})
    if prompt_image_path:
        uri = jpeg_to_data_uri(prompt_image_path)
        # Correct multimodal part structure
        user_content.append({"type": "image_url", "image_url": {"url": uri}, "description": "current view: what you see with your eyes"})

    
    # 3. Create the messages list
    messages = [
        {"role": "system", "content": character_prompt},
        {"role": "system", "content": system_prompt},
        # user_content is already a list of dicts with "type", so this is correct for multimodal
        {"role": "user", "content": user_content} 
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Call Failed: {e}. Cannot generate Elara's response."

def decide_query(prompt_text=None, prompt_image_path=None):
    """
    Decide what type of query to perform (text, image, both),
    and whether to search memory at all.
    """
    multimodal_prompt = []

    # Add image part if provided
    if prompt_image_path:
        uri = jpeg_to_data_uri(prompt_image_path)
        multimodal_prompt.append({
            "type": "image_url",
            "image_url": {"url": uri},
            "description": "User-provided image"
        })

    # Add text part if provided
    if prompt_text:
        multimodal_prompt.append({"type": "text", "text": f"User said: {prompt_text}"})

    decision_instruction = """
You are an intelligent multimodal memory routing assistant.

Given the user's current text and/or image, decide:
- whether a memory lookup is needed,
- what the query type is,
- what keyword or concept should be used for retrieval.
- you can have multiple queries needed.
Your job is to determine **which retrieval functions** should be triggered and **with what inputs**.

Use reasoning (internally) to infer whether the query involves:
- text similarity
- image similarity
- audio similarity
- cross-modal understanding (text→image, image→text, text→audio, audio→text)


Rules:
- If an image is provided but no text, query_type = "image_only".
- If text includes 'remember', 'what was', 'looked like', or a specific entity → query_needed = true.
- If both image and text describe or ask about appearance, use "text_and_image".
- Keep answers compact; do not add commentary.


--------------------
📘 RETRIEVAL DECISION RULES
--------------------
- If text + image and text asks about appearance, object, or scene → add "search_text_and_images" to be inputted to the Vision model, mainly asking what something look like.mainly asking what something look like.
- If text + audio and text mentions sound, voice, or speech → add "search_text_and_audios".
- If text asks "what does this image/audio mean" → add corresponding cross-modal reverse lookup:
  - image → "search_image_and_text"
  - audio → "search_audio_and_texts".
- Multiple retrievals can be triggered if the query implies multiple modalities.
- If text includes 'remember', 'what was', 'looked like', 'sounded like' → query_needed = true.
- Otherwise, query_needed = false.

Keep JSON minimal; no commentary.

--------------------
🔍 FUNCTION ROUTING TABLE
--------------------
| Function Name           | Input Type | Output Type | Description |
|--------------------------|-------------|--------------|--------------|
| search_whole_audios      | Audio       | Audio list   | Semantic audio search |
| search_text_and_audios   | Text        | Audio list   | Retrieve audios linked to similar text |
| search_audio_and_texts   | Audio       | Text list    | Retrieve texts linked to similar audio |
| search_whole_images      | Image       | Image list   | Semantic image search |
| search_image_and_text    | Image       | Text list    | Retrieve texts linked to similar images |
| search_similar_texts     | Text        | Text list    | Semantic text search |
| search_text_and_images   | Text        | Image list   | Retrieve images linked to similar text mainly asking what something look like. |

--------------------
🔗 CROSS-MODAL AND INTRA-MODAL MAPPING
--------------------
| Type | Input | Output | Function |
|------|--------|---------|----------|
| Intra-Modal | Image Embedding | Image Path + Distance | search_whole_images |
| Intra-Modal | Text Embedding  | Text Content + Distance | search_similar_texts |
| Cross-Modal (L-V-L) | Image Path | Associated Text | search_image_and_text |
| Cross-Modal (L-V-L) | Text Query | Associated Images | search_text_and_images |
| Cross-Modal (L-A-L) | Text Query | Associated Audio Paths | search_text_and_audios |
| Cross-Modal (A-L-A) | Audio Path | Associated Text Content | search_audio_and_texts |

Return a compact JSON object strictly in this format:
{
  "query_needed": true or false,
  "queries_needed": [
    {
      "query_type": "search_similar_texts" | "search_whole_images" |
                     "search_text_and_images" | "search_image_and_text" | "search_text_and_audios"
      "query": "keyword or short phrase, if search_image_and_text or search_whole_images return 'image'",
      "reason": "short reason"
    },
    {
      "query_type": "search_similar_texts" | "search_whole_images" |
                     "search_text_and_images" | "search_image_and_text" | "search_text_and_audios"
      "query": "keyword or short phrase, if search_image_and_text or search_whole_images return 'image'",
      "reason": "short reason"
    }
  ]
}
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You decide the query modality and retrieval intent based on multimodal input."},
                {"role": "user", "content": [*multimodal_prompt, {"type": "text", "text": decision_instruction}]}
            ],
            max_tokens=150,
        )

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        result = json.loads(raw_content)
        print("Raw JSON:", result)

    except Exception as e:
        print(f"Decision failed: {e}")
        result = {
            "query_needed": False,
            "queries_needed": [
            ]

        }

    return result

def elara_multimodal_memory_check(user_input: str, prompt_image_path: str = None, prompt_audio_path: str = None) -> str:
    """
    Manages the full lifecycle of a user interaction:
    1. Decides the query type and keyword.
    2. Retrieves and filters relevant memories (text and linked images).
    3. Generates the response using the Elara persona and retrieved memories as context.
    
    Args:
        user_input: The text prompt from the user (e.g., "What did the cat look like?").
        prompt_image_path: Optional path to the user's current image view.
        
    Returns:
        The text response generated by the Groq model in Elara's persona.
    """
    memory_paths = []
    # --- 1. Decide Query Type and Keyword ---
    print(f"--- Deciding Query for: '{user_input}' (Image: {prompt_image_path or 'None'}) (Audio: {prompt_audio_path or 'None'}) ---")
    retrieved_audios_text = ""

    if prompt_audio_path:
        audio_results = search_audio_and_texts(prompt_audio_path, k_audios=3)
        for path, uuid_val, dist, texts in audio_results:
            print(f"  Audio: {path} (Dist: {dist:.4f})")
            print(f"    Associated Texts: {texts}")
            retrieved_audios_text += f"Audio Label Retrieved: (Dist: {dist:.4f})\n    Associated Texts: {texts}\n"
        memory_paths.append({"type": "text", "content": retrieved_audios_text})
    """
    Random Thought:
    It should be able to return query needed and have multiple queries possible.
    So decision should contain multiple queries needed.
    """


    try:
        # NOTE: Using the real function call. Ensure client/API key is configured.
        decision = decide_query(user_input, prompt_image_path)

    except Exception as e:
        print(f"Error during query decision, falling back to simple text query: {e}")
        # Fallback decision structure
        decision = {
            "query_needed": False,
            "queries_needed": [
                {
                    "query_type": "text_only",
                    "query": user_input,
                    "reason": "Decision API failed."
                }
            ]
        }
        
    print("Decision:", decision)
    

    #Reasoning Loop
    max_query_depth = 2
    query_depth = 0

    query_depth += 1
    # --- 2. Memory Retrieval and Python-side Filtering ---
    queries_needed = decision.get("queries_needed", [])
    print(f"\n--- Retrieving memory for keyword: '{queries_needed}' ---")

    query_results = []

    # Define filtering constants
    MAX_DIST = 0.3
    TOP_K = 2
    for query in queries_needed:
        query_type = query.get("query_type")
        query_text = query.get("query")
        print(f"\n--- Retrieving memory for keyword: '{query_text}' ---")

        if query_type == "search_text_and_images":
            linked_text_results = search_text_and_images(query_text, 10)
            
            # 1. Filter the results by distance
            filtered_results = [
                item for item in linked_text_results if item[2] <= MAX_DIST # item[2] is the 'dist'
            ]
            # 2. Limit the results to the top K (already sorted by dist ASC by SQL)
            final_results = filtered_results[:TOP_K]
            for content, text_uuid, dist, image_list in final_results:
                print(f"  Text Match (Dist: {dist:.4f}): '{content}'")
                # 1. Always add the text content
                memory_paths.append({"type": "text", "content": content + f" (Dist: {dist:.4f})"})


                # 2. Add ALL linked images in this scenario for rich context
                for path, img_uuid in image_list:
                    print(f"    Linked Image: - {path}")
                    memory_paths.append({"type": "image", "content": path})
            query_results.extend(final_results)
        elif query_type == "search_similar_texts":
            text_query_emb = get_text_embedding(query_text)

            similar_text_results = search_similar_texts(text_query_emb, 10)
            filtered_results = [
                item for item in similar_text_results if item[2] <= MAX_DIST # item[2] is the 'dist'
            ]
            final_results = filtered_results[:TOP_K]
            for content, uuid_val, dist in final_results:
                print(f"  Text Match (Dist: {dist:.4f}): '{content}'")
                # 1. Always add the text content
                memory_paths.append({"type": "text", "content": content + f" (Dist: {dist:.4f})"})

            query_results.extend(final_results)
        elif query_type == "search_whole_images":
            img_query_emb = get_image_embedding(Image.open(query_text).convert("RGB"))
            whole_image_results = search_whole_images(img_query_emb,10)
            filtered_results = [
                item for item in whole_image_results if item[2] <= MAX_DIST # item[2] is the 'dist'
            ]
            final_results = filtered_results[:TOP_K]
            for path, uuid_val, dist in final_results:
                print(f"  Image Match (Dist: {dist:.4f}): '{path}'")
                # 1. Always add the image content
                memory_paths.append({"type": "image", "content": path})

            query_results.extend(final_results)
        elif query_type == "search_image_and_text":
            image_and_text_results = search_image_and_text(query_text, 10)
            filtered_results = [
                item for item in image_and_text_results if item[2] <= MAX_DIST # item[2] is the 'dist'
            ]
            final_results = filtered_results[:TOP_K]
            for path, uuid_val, dist, text_list in final_results:
                # 1. Always add the image content
                memory_paths.append({"type": "text", "content": text_list})

                print(f"  Image Match (Dist: {dist:.4f}): '{path}'")
                # 1. Always add the image content
                memory_paths.append({"type": "image", "content": path})
            query_results.extend(final_results)
            

    # --- 3. Generate Elara's Response ---
    print("\n--- Generating Elara's Response ---")
    output = check_memory(memory_paths, user_input, prompt_image_path)
    
    return output

# ----------------------------------------------------------------------
## Demonstration Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure to run the data insertion/indexing part of your original script 
    # before calling this function in a live environment.
    
    # --- Example 1: Successful Retrieval of a known entity (Cat) ---
    user_query = "What did the cat look like? I'm curious. Whats its name?"
    print(f"\n\n=======================================================")
    print(f"DEMO 1: Query for '{user_query}'")
    print(f"=======================================================")
    elara_response = elara_multimodal_memory_check(user_query)
    print("\nElara:", elara_response)
    
    # --- Example 2: Query for a different entity (Car) with image context ---
    # Note: If 'car.jpg' is similar to the stored 'cat.jpg', the image-to-image 
    # part of the router might trigger.
    user_query_2 = "Tell me about Lumi. What she look like?"
    print(f"\n\n=======================================================")
    print(f"DEMO 2: Query for '{user_query_2}'")
    print(f"=======================================================")
    elara_response_2 = elara_multimodal_memory_check(user_query_2, prompt_image_path="lumi.png")
    print("\nElara:", elara_response_2)

    
    # --- Example 1: Successful Retrieval of a known entity (Cat) ---
    user_query = "What did the dog look like?"
    print(f"\n\n=======================================================")
    print(f"DEMO 3: Query for '{user_query}'")
    print(f"=======================================================")
    elara_response = elara_multimodal_memory_check(user_query)
    print("\nElara:", elara_response)

    user_query_3 = "What do you hear?"
    print(f"\n\n=======================================================")
    print(f"DEMO 4: Query for '{user_query_3}'")
    print(f"=======================================================")
    elara_response_3 = elara_multimodal_memory_check(user_query_3, prompt_audio_path="D:/pyfiles/research/audiocaps_raw_audio/audiocaps_raw_audio/OlRK51FBR4c_30.wav")

    print("\nElara:", elara_response_3)
