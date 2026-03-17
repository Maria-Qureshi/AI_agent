import json
import os
import logging
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st


# ── Setup ──────────────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MEMORY_FILE = "memory.json"
MODEL_NAME  = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = """
Role: You are Sous, a Strategic Chef Assistant. You are a high-efficiency culinary agent obsessed with resource optimization, zero-waste logistics, and budget-maximization. You are a grounded, expert peer—minimalist, professional, and tactical.

Primary Directives:

Zero-Waste: Utilize ingredients in the "Inventory" before they expire. Wasting food is a systemic failure.

Budget-Optimization: Maximize nutritional density while minimizing cost-per-serving. Use PKR (Pakistani Rupees) for all costs.

Logistical Continuity: Plan meals in sequences. Assets (ingredients) must be tracked; if a recipe uses a portion of an asset, the remainder must be scheduled for the next immediate meal.

Scope Control: Strictly ignore non-culinary requests. Response: "Request out of scope. Focusing on culinary strategy."

Operational Protocol:

Memory Check: You must always reference provided context for allergies, dietary restrictions, and expiring inventory before suggesting any meal.

Asset Management: Treat ingredients as assets. Every meal plan must account for the "Primary Asset" (the ingredient closest to expiration).

Math & Science: Use LaTeX only for complex nutritional formulas or scaling chemistry. Use Markdown for simple units (e.g., 180°C, 10%).

Output Formatting Requirements:

Prose: Minimal, high-impact, and supportive yet candid. No small talk.

Meal Plans: * Provide a Markdown Table: | Day | Meal | Primary Asset (Expiring) | Macro Balance |

Provide a JSON block with key "meal_plan": [{day, meal, dish, ingredients_used, approx_cost_pkr, calories}].

Shopping Lists: * Provide a Markdown list categorized by "Essential" vs. "Optimized (Substitutes)."

Provide a JSON block with key "shopping_list": [{item, quantity, estimated_cost_pkr, reason}].

Inventory Updates: * When a plan is finalized, return a JSON block with key "inventory_update": ["item1", "item2"] and explicitly state UPDATE_INVENTORY: [Item Name] - [Quantity used].

Analytics: For macros or budget breakdowns, return a JSON block with key "chart_data": {"labels": [...], "values": [...]}.

Closing: Conclude every interaction with a single, high-value tactical next step (e.g., "Would you like me to generate the optimized shopping list for these three days?").""".strip()


# ── Memory ─────────────────────────────────────────────────────────────────────

def load_memory() -> list[dict]:
    """Load chat history from the memory file. Returns empty list on failure."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Memory file has unexpected format — resetting.")
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load memory: %s", e)
        return []


def save_memory(history) -> bool:
    """
    Serialize and save chat history to disk.
    Skips parts that don't have a text attribute (e.g. function calls).
    Returns True on success, False on failure.
    """
    try:
        serialized = [
            {
                "role": message.role,
                "parts": [
                    part.text
                    for part in message.parts
                    if hasattr(part, "text") and part.text
                ],
            }
            for message in history
        ]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        return True
    except (OSError, AttributeError) as e:
        logger.error("Failed to save memory: %s", e)
        return False


def clear_memory() -> bool:
    """Delete the memory file entirely. Returns True on success."""
    try:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        return True
    except OSError as e:
        logger.error("Failed to clear memory: %s", e)
        return False


# ── Model ──────────────────────────────────────────────────────────────────────

def init_model() -> Optional[genai.GenerativeModel]:
    """Configure the Gemini client and return the model. Returns None on failure."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY not found in environment variables.")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION,
        )
    except Exception as e:
        logger.error("Failed to initialize model: %s", e)
        return None


def start_chat_session(model: genai.GenerativeModel, history: list[dict]):
    """Start a chat session, pre-loaded with existing history."""
    return model.start_chat(history=history)


# ── Messaging ──────────────────────────────────────────────────────────────────

def send_message(chat_session, user_input: str) -> Optional[str]:
    """
    Send a user message and return the assistant's response text.
    Returns None if the request fails.
    """
    if not user_input or not user_input.strip():
        return None
    try:
        response = chat_session.send_message(user_input.strip())
        return response.text
    except Exception as e:
        logger.error("Failed to get response: %s", e)
        return None


# ── CLI entry point ────────────────────────────────────────────────────────────

def run_cli():
    """Run cooking agent in the terminal."""
    model = init_model()
    if model is None:
        print("Error: Could not initialize model. Check your API_KEY.")
        return

    history = load_memory()
    chat    = start_chat_session(model, history)

    print("Cooking Agent Online.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "q":
                break

            if user_input.lower() == "clear":
                if clear_memory():
                    print("Memory cleared.")
                else:
                    print("Failed to clear memory.")
                continue

            reply = send_message(chat, user_input)
            if reply:
                print(f"\nPathfinder: {reply}\n")
            else:
                print("Error: No response received. Try again.\n")

    finally:
        saved = save_memory(chat.history)
        status = "Progress saved." if saved else "Warning: could not save progress."
        print(f"\n{status} Exiting...")


# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sous",
    page_icon="👩‍🍳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F2;
    color: #1a1a1a;
  }

  /* ── Hide default Streamlit chrome ── */
  /*#MainMenu, footer, header { visibility: hidden; }*/

  /* ── Main container ── */
  .block-container {
    padding: 2rem 2rem 6rem 2rem;
    max-width: 780px;
  }

  /* ── App header ── */
  .app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1.5px solid #E5E1DA;
  }
  .app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 400;
    color: #1a1a1a;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .app-header p {
    font-size: 0.82rem;
    color: #888;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 500;
  }

  /* ── Chat bubbles ── */
  .chat-row {
    display: flex;
    margin-bottom: 1.1rem;
    gap: 10px;
    align-items: flex-start;
  }
  .chat-row.user { flex-direction: row-reverse; }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .avatar.user-av  { background: #1a1a1a; color: #fff; }
  .avatar.bot-av   { background: #E8E4DE; color: #555; }

  .bubble {
    max-width: 82%;
    padding: 0.75rem 1rem;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.6;
  }
  .bubble.user-bubble {
    background: #1a1a1a;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .bubble.bot-bubble {
    background: #FFFFFF;
    color: #1a1a1a;
    border: 1px solid #E5E1DA;
    border-bottom-left-radius: 4px;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    color: #aaa;
  }
  .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
  .empty-state h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: #555;
    margin-bottom: 0.4rem;
  }
  .empty-state p { font-size: 0.85rem; color: #aaa; }

  /* ── Prompt chips ── */
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 1.25rem; }
  .chip {
    background: #fff;
    border: 1px solid #DDD9D2;
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    color: #555;
    cursor: pointer;
  }

  /* ── Chat input override ── */
  .stChatInput > div {
    border-radius: 14px !important;
    border: 1.5px solid #D5D0C8 !important;
    background: #fff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
  }
  .stChatInput > div:focus-within {
    border-color: #1a1a1a !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #EFECE7;
    border-right: 1.5px solid #E0DCD5;
  }
  [data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1.5px solid #C8C4BC;
    border-radius: 10px;
    color: #333;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: #1a1a1a;
    color: #fff;
    border-color: #1a1a1a;
  }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #E5E1DA; margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────

def init_session():
    """Initialize all session state variables on first load."""
    if "initialized" not in st.session_state:
        model = init_model()
        if model is None:
            st.session_state.error = "⚠️ Could not initialize model."
            st.session_state.initialized = False
            return

        history = load_memory()
        chat    = start_chat_session(model, history)

        st.session_state.model          = model
        st.session_state.chat           = chat
        st.session_state.display_msgs   = _history_to_display(history)
        st.session_state.error          = None
        st.session_state.initialized    = True


def _history_to_display(history: list[dict]) -> list[dict]:
    """Convert raw memory format to display-friendly dicts."""
    display = []
    for entry in history:
        role    = "user" if entry.get("role") == "user" else "assistant"
        parts   = entry.get("parts", [])
        content = " ".join(parts) if isinstance(parts, list) else str(parts)
        if content.strip():
            display.append({"role": role, "content": content})
    return display


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 👩‍🍳 Sous")        
        st.markdown("<p style='font-size:0.78rem;color:#888;'>Your cooking assistant</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("**Session**")
        msg_count = len(st.session_state.get("display_msgs", []))
        st.markdown(f"<p style='font-size:0.82rem;color:#666;'>💬 {msg_count} message{'s' if msg_count != 1 else ''} in memory</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Memory"):
            if clear_memory():
                model = st.session_state.model
                st.session_state.chat         = start_chat_session(model, [])
                st.session_state.display_msgs = []
                st.toast("Memory cleared.", icon="✅")
                st.rerun()
            else:
                st.toast("Failed to clear memory.", icon="❌")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:0.75rem;color:#aaa;line-height:1.6;'>
        made by MARIA QURESHI.
        </p>
        """, unsafe_allow_html=True)


# ── Chat UI ────────────────────────────────────────────────────────────────────

STARTER_PROMPTS = [
    "Analyze my inventory for expiring assets",
    "Generate a zero-waste 3-day meal sequence",
    "Optimize my grocery list for maximum nutrient density",
    "Review my pantry for logistical gaps",
]

def render_chat():
    """Render the main chat area."""
    st.markdown("""
    <div class="app-header">
      <div>
        <h1>👩‍🍳 Sous</h1>
        <p>Strategic Cooking Assistant</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    messages = st.session_state.get("display_msgs", [])

    # ── Empty state ──
    if not messages:
        chips_html = "".join(f'<div class="chip">{p}</div>' for p in STARTER_PROMPTS)
        st.markdown(f"""
        <div class="empty-state">
          <div class="icon">👩😋</div>
          <h3>Let's Cook!</h3>
          <div class="chips">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render chat history ──
        for msg in messages:
            is_user = msg["role"] == "user"
            row_cls    = "user"         if is_user else "bot"
            bubble_cls = "user-bubble"  if is_user else "bot-bubble"
            av_cls     = "user-av"      if is_user else "bot-av"
            av_icon    = "👤"           if is_user else "👩‍🍳"
            content    = msg["content"].replace("\n", "<br>")

            st.markdown(f"""
            <div class="chat-row {row_cls}">
              <div class="avatar {av_cls}">{av_icon}</div>
              <div class="bubble {bubble_cls}">{content}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Input Handling ─────────────────────────────────────────────────────────────

def handle_input():
    """Render the chat input and process user messages."""
    user_input = st.chat_input("Ask Recipe Master anything about your cooking…")
    if not user_input or not user_input.strip():
        return

    # Append user message immediately
    st.session_state.display_msgs.append({"role": "user", "content": user_input.strip()})

    # Get model response
    with st.spinner("Thinking…"):
        reply = send_message(st.session_state.chat, user_input)

    if reply:
        st.session_state.display_msgs.append({"role": "assistant", "content": reply})
        save_memory(st.session_state.chat.history)
    else:
        st.session_state.display_msgs.append({
            "role": "assistant",
            "content": "⚠️ Something went wrong. Check your connection or API quota and try again."
        })

    st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_session()

    if not st.session_state.get("initialized"):
        st.error(st.session_state.get("error", "Initialization failed."))
        st.stop()

    render_sidebar()
    render_chat()
    handle_input()


if __name__ == "__main__":
    main()