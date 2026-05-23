# Chatbot-for-FAQs
FAQs related to a product

import tkinter as tk
import customtkinter as ctk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# seting up FAQ dataset
# we can expand this dictionary with any questions and answers relevant to our topic.
FAQ_DATA = {
    "What is your return policy?":
        "You can return any product within 30 days of purchase for a full refund. Items must be in their original packaging.",
    "How long does shipping take?":
        "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "Do you ship internationally?":
        "Yes, we ship to over 50 countries worldwide. Shipping fees and delivery times vary by destination.",
    "How can I track my order?":
        "Once your order ships, we will email you a tracking link. You can also track it via the 'My Orders' section on our website.",
    "What payment methods do you accept?":
        "We accept all major credit cards (Visa, MasterCard, Amex), PayPal, and Apple Pay.",
    "How do I contact customer support?":
        "You can reach our support team via email at support@example.com or call us at 1-800-555-0199 between 9 AM and 5 PM EST."
}
# Extract questions into a list for semantic comparison
FAQ_QUESTIONS = list(FAQ_DATA.keys())
print("Loading NLP Model... (This may take a moment on the first run)")
# Load a lightweight, high-performance embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Pre-compute embeddings for all of our known FAQ questions
FAQ_EMBEDDINGS = model.encode(FAQ_QUESTIONS)
# NLP matching
def get_bot_response(user_query):
    """Finds the closest matching FAQ question using cosine similarity."""
    if not user_query.strip():
        return "Please type something so I can help you!"
    # Generate vector embedding for what the user just typed
    user_embedding = model.encode([user_query])
    # Calculate similarity scores between user query and all FAQ questions
    # cosine_similarity returns an array of arrays, so we flatten it with [0]
    similarity_scores = cosine_similarity(user_embedding, FAQ_EMBEDDINGS)[0]
    # Find the index of the highest similarity score
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]
    # Set a confidence threshold (0.4 means the phrase must match reasonably well)
    if highest_score > 0.4:
        matched_question = FAQ_QUESTIONS[best_match_idx]
        return FAQ_DATA[matched_question]
    else:
        return "I'm sorry, I couldn't find an exact match for that question. Could you try rephrasing it, or contact our support team at support@example.com?"
# chat interactive UI (CustomTkinter)
class ChatbotWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Configure window settings
        self.title("FAQ Support Chatbot")
        self.geometry("450x600")
        ctk.set_appearance_mode("dark")  # Modes: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")
        # Grid layout configuration
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # Chat History Log Textbox
        self.chat_history = ctk.CTkTextbox(self, state="disabled", wrap="word", font=("Arial", 13))
        self.chat_history.grid(row=0, column=0, padx=20, pady=(20, 10), columnspan=2, sticky="nsew")
        # User Input Box
        self.user_input = ctk.CTkEntry(self, placeholder_text="Ask me a question...")
        self.user_input.grid(row=1, column=0, padx=(20, 10), pady=20, sticky="ew")
        self.user_input.bind("<Return>", lambda event: self.send_message())
        # Send Button
        self.send_button = ctk.CTkButton(self, text="Send", width=80, command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=(10, 20), pady=20, sticky="w")
        # Display greeting message
        self.append_to_chat("Bot",
                            "Hello! I am your automated assistant. Ask me anything about our shipping, returns, or payment options.")
    def append_to_chat(self, sender, text):
        """Helper function to append text blocks cleanly to the chat display."""
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"{sender}: {text}\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see(tk.END)  # Auto-scrolls to the bottom
    def send_message(self):
        query = self.user_input.get()
        if not query.strip():
            return
        # Display user message
        self.append_to_chat("You", query)
        self.user_input.delete(0, tk.END)
        # Fetch and display bot response
        response = get_bot_response(query)
        self.append_to_chat("Bot", response)
if __name__ == "__main__":
    app = ChatbotWindow()
    app.mainloop()
