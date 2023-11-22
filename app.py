import streamlit as st

# Streamlit app
def main():
    st.title("Face Recognition Using SVM & PCA")

 # Add HTML and CSS to customize the file uploader color
    st.markdown(
        """
        <style>
            .st-emotion-cache-1gulkj5 { 
                background-color: #E8F89F !important;
            }
            # .st-emotion-cache-7ym5gk:hover {
            #     color: #000 !important;
            #     border-color: blue;
            # }
        </style>
        """,
        unsafe_allow_html=True
    )

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
         # Display the custom file uploader display using Markdown
        st.markdown(f"### Uploaded Image: {uploaded_file.name}")
        # Display the uploaded image
        st.image(uploaded_file, caption=f"Original Image: {uploaded_file.name}", use_column_width=True)

        # Display the prediction result
        st.subheader("Prediction: ")

if __name__ == '__main__':
    main()
