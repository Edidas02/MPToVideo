import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RELEVANCE_THRESHOLD = 0.25

# Step 1: Preprocess the Data
def preprocess_text(text):
    """
    Preprocesses the given text.

    Parameters:
    text (str): The text to be processed.

    Returns:
    str: The processed text.
    """
    return text.lower()

# Load CSV
df = pd.read_csv('./data/lecturedata.csv') # Update with the actual path
#print(df)
transcripts = df['Transcript'].apply(preprocess_text)

# Step 2: Vectorizing the Text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(transcripts)

# Step 3 & 4: Search Function
def find_relevant_lectures(query, top_n=3):
    """
    Finds and returns the most relevant lectures for a given query.

    Parameters:
    query (str): The search query (assignment description).
    top_n (int, optional): The number of top relevant lectures to return. Defaults to 5.

    Returns:
    list: A list of the most relevant lecture numbers.
    """
    query_vec = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Filter based on threshold
    relevant_lectures = [(index, score) for index, score in enumerate(cosine_similarities) if score >= RELEVANCE_THRESHOLD]
    
    # Sort based on scores and get top_n indices
    relevant_lectures.sort(key=lambda x: x[1], reverse=True)
    relevant_lectures_indices = [index for index, score in relevant_lectures][:top_n]

    return df['Lecture'].iloc[relevant_lectures_indices].to_list()


# Example Usage
query1 = '''
In this MP, you will get familiar with building and evaluating Search Engines.

Part 1
In this part, you will use the MeTA toolkit to do the following:

create a search engine over a dataset
investigate the effect of parameter values for a standard retrieval function
write the InL2 retrieval function
investigate the effect of the parameter value for InL2
Choose one of the above retrieval functions and one of its parameters (don’t choose BM25 + k3, it’s not interesting). For example, you could choose Dirichlet Prior and mu.

Change the ranker to your method and parameters. In the example, it is set to bm25. Use at least 10 different values for the parameter you chose; try to choose the values such that you can find a maximum MAP.
'''

query2 = '''
In this part, you will develop a classifier and participate in a classification competition. Your classifier will be evaluated using F1 scores on 2 datasets: metamia dataset (created from metamia.com), MP3.1 Analogy dataset collected and annotated by you and your classmates.

You are free to use any classifier (Hint: The web pages are really long, so models suitable for long document classification (e.g., Longformer) might work better) and fine-tune various hyper-parameters (e.g., learning rate, number of epochs). We recommend you start by experimenting with learning rate and number of epochs.

To see how well you perform in the leaderboard, you need to run the prediction code on the test sets of both the datasets. For metamia dataset, save the results as metamia_results.csv and for mp3.1, save as mp3.1_results.csv. These two files must be located in the root folder in your repo, i.e., it should not be under any sub-folders.
'''

query3 = '''
molecular biology
'''

relevant_lectures = find_relevant_lectures(query1)
print("Lecture Numbers: ", relevant_lectures)