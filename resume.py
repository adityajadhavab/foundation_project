## importing required libraries
from pycaret.clustering import *
import streamlit as st
import pandas as pd
from PIL import Image
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

## loading the kmeans model
model = load_model('Final Kmeans Model')

## defining a function to make predictions
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['Cluster'][0]
    return predictions

## defining the main function
def run():

    ## loading an image
    image = Image.open('Resume-Ranking.jpg')

    ## adding the image to the webapp
    st.image(image, use_column_width=True)

    ## adding a selectbox making a choice between two broadways to predict new data points
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch")
    )

    ## adding some information about the app's functioning to the sidebar
    st.sidebar.info('This app is created for resume ranking')

    ## adding the title for the streamlit app
    st.title("Resume Ranking System App")

    ## adding steps to be followed if the user selects Online mode of prediction 
    if add_selectbox == 'Online':

        ## adding a text input box to get skills
        skills = st.text_input('Skills')

        ## adding a text input box to get experience
        experience = st.text_input('Experience')

        ## adding a text input box to get degree
        degree = st.text_input('Degree')

        ## adding a text input box to get job description
        job_description = st.text_area('Job Description')

        ## defining the output variable 
        output = ""

        ## creating an input dictionary with all the input features
        input_dict = {'Skills': skills, 'Experience': experience, 'Degree': degree, 'Job_Description': job_description}

        ## converting the input dictionary into a pandas DataFrame
        input_df = pd.DataFrame([input_dict])

        ## adding a button to make predictions when clicked on by the user
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        ## displaying the output after successful prediction
        st.success('The output is {}'.format(output))

    ## adding steps to be followed if the user selects Batch mode of prediction
    if add_selectbox == 'Batch':

        ## adding a file uploader button for the user to upload the CSV file containing data points
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])

        ## block of code to be run once a CSV file is uploaded by the user
        if file_upload is not None:

            ## reading the CSV file using pandas
            data = pd.read_csv(file_upload)

            ## making predictions
            predictions = predict_model(model, data=data)
            
            lst = list(predictions['Cluster'])
            counter = Counter(lst)
            max_occurrence = max(counter, key=counter.get)
            max_occurrence = str(max_occurrence)  # Print the variable with the maximum occurrence

            predictions = predictions[predictions['Cluster'] == max_occurrence]

            #predictions = predictions[['Person','About','Cluster','cosine_similarity']]

            job_description = predictions['job_description'][0]
            
            tfidf_vectorizer = TfidfVectorizer()

            skills_tfidf = tfidf_vectorizer.fit_transform(predictions['description'])
            #degree_tfidf = tfidf_vectorizer.fit_transform(predictions['degree'])
            #degree_experience = tfidf_vectorizer.fit_transform(predictions['About'])

            jd_tfidf = tfidf_vectorizer.transform([job_description])


            cosine_similarities_des = cosine_similarity(jd_tfidf, skills_tfidf).flatten()
            #cosine_similarities_degree = cosine_similarity(jd_tfidf, degree_tfidf).flatten()
            #cosine_similarities_exp = cosine_similarity(jd_tfidf, degree_experience).flatten()


            predictions['cosine_similarities_des'] = cosine_similarities_des
            #predictions['cosine_similarities_deg'] = cosine_similarities_degree
            #predictions['cosine_similarities_exp'] = cosine_similarities_exp



            # Create an instance of CountVectorizer
            vectorizer = CountVectorizer()

            # Fit and transform the skills data
            skills_bow = vectorizer.fit_transform(predictions['degree'])

            # Transform the jd data
            jd_bow = vectorizer.transform([job_description])

            # Calculate cosine similarities
            cosine_similarities_bow = cosine_similarity(jd_bow, skills_bow).flatten()

            # Assign cosine similarities to the data
            predictions['cosine_similarity_bow'] = cosine_similarities_bow

            predictions['Score'] = predictions['cosine_similarities_des'] * 0.75 + predictions['cosine_similarity_bow'] *.25 #+ predictions['cosine_similarities_exp']*.3
            
            predictions = predictions[predictions['Score'] > 0.3]
            predictions = predictions.sort_values(by=['Score'], ascending=False)


            predictions = predictions[['Person','About','Cluster','Score']]


            ## writing the predictions
            st.write(predictions)

## calling the main function
if __name__ == '__main__':
    run()
