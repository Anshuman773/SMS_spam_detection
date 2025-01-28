import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit UI
st.title("📩 SMS Spam Detection App")
st.markdown(
    """
    Welcome to the **SMS Spam Detection App**!  
    Enter an SMS message below, and we'll tell you whether it's **Spam** or **Not Spam**.  
    *(Built by Anshuman Mahapatra)*  
    ---
    """
)

# Input text
st.subheader("Enter SMS:")
input_sms = st.text_area(
    "Type your message here 👇",
    placeholder="E.g., 'Congratulations! You've won a free ticket to Paris. Click here to claim.'",
    height=150,
)

if st.button('🔍 Predict'):
    if input_sms.strip():  # Check if input is not empty
        # Preprocess the input SMS
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the input
        vector_input = tk.transform([transformed_sms])
        
        # Predict the result
        result = model.predict(vector_input)[0]
        
        # Display the result with styling
        if result == 1:
            st.markdown("### 🚨 **Spam Message Detected!**")
            st.error("Be cautious! This message appears to be spam.")
        else:
            st.markdown("### ✅ **Not Spam**")
            st.success("This message looks safe.")
    else:
        st.warning("⚠️ Please enter a valid SMS message to analyze.")

# Footer
st.markdown(
    """
    ---
    Made with ❤️ using **Streamlit**.  
    Have feedback? [Contact Anshuman Mahapatra](https://www.linkedin.com/in/anshuman-mahapatra1704/ ) 
    """
)
    