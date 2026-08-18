"""Conversation module for visitor interactions."""

import threading
import time
import stt
import tts
import llm
import logger as log_module

logger = log_module.logging.getLogger("openhouse")


class VisitorConversation:
    def __init__(self, visitor_id: int, name: str, set_status_cb):
        self.visitor_id = visitor_id
        self.name = name
        self.set_status = set_status_cb
        self.conversation_history = []
        self.is_active = False
    
    def start(self):
        """Start interactive conversation with visitor."""
        self.is_active = True
        self.set_status(f"💬 Listening to {self.name}...", "blue")
        
        def _listen_and_respond():
            try:
                # Listen for visitor speech
                visitor_speech = stt.capture_name(timeout=15)
                
                if visitor_speech:
                    self.conversation_history.append({"role": "visitor", "text": visitor_speech})
                    self.set_status(f"🎤 You said: {visitor_speech}", "gray")
                    
                    # Generate response from LLM
                    response = llm.generate_chat_response(visitor_speech)
                    self.conversation_history.append({"role": "host", "text": response})
                    
                    self.set_status(f"🤖 Host: {response}", "green")
                    tts.speak(response)
                    
                    # Continue conversation
                    self._prompt_continue()
                else:
                    self.set_status("⏱️ No response detected", "orange")
                    self.is_active = False
            except Exception as e:
                logger.error("Conversation error: %s", e)
                self.set_status(f"❌ Error: {e}", "red")
                self.is_active = False
        
        threading.Thread(target=_listen_and_respond, daemon=True).start()
    
    def _prompt_continue(self):
        """Ask if visitor wants to continue conversation."""
        time.sleep(1)
        tts.speak("Would you like to continue chatting?")
        self.set_status("🎤 Say yes to continue, no to end...", "blue")
        
        def _listen_continue():
            response = stt.capture_name(timeout=5)
            if response and ('yes' in response.lower() or 'yeah' in response.lower()):
                self.start()
            else:
                self.is_active = False
                self.set_status(f"👋 Goodbye {self.name}!", "green")
                tts.speak(f"Goodbye {self.name}! Thanks for visiting!")
        
        threading.Thread(target=_listen_continue, daemon=True).start()
    
    def ask_for_name(self, encoding):
        """Ask new visitor for their preferred name."""
        self.set_status("🎤 What would you like to be called?", "orange")
        tts.speak("Would you like to provide a nice name to be called?")
        
        def _get_name():
            name = stt.capture_name(timeout=10)
            if name and len(name.strip()) > 1:
                self.name = name.capitalize()
                self.set_status(f"✅ Nice to meet you, {self.name}!", "green")
                tts.speak(f"Nice to meet you, {self.name}!")
            else:
                self.set_status(f"Using default name: {self.name}", "gray")
        
        threading.Thread(target=_get_name, daemon=True).start()
