import streamlit as st
import requests
from typing import Optional


DEFAULT_API_URL = "http://localhost:8000"
TRANSLATE_ENDPOINT = "/translate"


def translate_text(text: str, api_url: str) -> tuple[Optional[str], Optional[str]]:
    if not text.strip():
        return None, "Please enter some text to translate."
    
    try:
        response = requests.post(
            f"{api_url}{TRANSLATE_ENDPOINT}",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        translated_text = response.text
        return translated_text, None
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to API at {api_url}. Make sure the FastAPI server is running."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return None, f"API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return None, f"An error occurred: {str(e)}"


def main():
    st.set_page_config(
        page_title="English to Spanish Translator",
        page_icon="🌐",
        layout="centered"
    )
    
    st.title("🌐 English to Spanish Translator")
    st.markdown("Translate English text to Spanish using our ML-powered translation API.")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_url = st.text_input(
            "API Base URL",
            value=DEFAULT_API_URL,
            help="Base URL of the FastAPI translation service"
        )
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Enter English text in the text area below
        2. Click the **Translate** button
        3. View the Spanish translation
        
        Make sure the FastAPI server is running on the configured port!
        """)
    
    st.markdown("---")
    
    input_text = st.text_area(
        "Enter English text to translate:",
        height=150,
        placeholder="Type or paste your English text here...",
        help="Enter the text you want to translate from English to Spanish"
    )
    
    if input_text:
        char_count = len(input_text)
        word_count = len(input_text.split())
        st.caption(f"📊 Characters: {char_count} | Words: {word_count}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        translate_button = st.button(
            "🔄 Translate",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )
    
    if clear_button:
        st.rerun()
    
    if translate_button:
        if not input_text or not input_text.strip():
            st.warning("⚠️ Please enter some text to translate.")
        else:
            with st.spinner("🔄 Translating... Please wait."):
                translated_text, error = translate_text(input_text, api_url)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Translation successful!")
                st.markdown("---")
                st.markdown("### Spanish Translation:")
                st.text_area(
                    "Translated text:",
                    value=translated_text,
                    height=150,
                    key="translation_output",
                    disabled=False
                )
                
                if st.button("📋 Copy Translation", use_container_width=True):
                    st.write("💡 Tip: Select and copy the text above, or use Cmd/Ctrl+C")
    
    st.markdown("---")
    st.caption("Powered by FastAPI and Streamlit | English → Spanish Translation")


if __name__ == "__main__":
    main()
