from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

lecture_transcripts = [] # TODO
assignment_descriptions = [] # TODO


# preprocess the data so its uniform
def preprocess(text):
    # lowercase and remove punctuation
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])

    return text

lecture_transcripts = [preprocess(lecture) for lecture in lecture_transcripts]
assignment_descriptions = [preprocess(assignment) for assignment in assignment_descriptions]


vectorizer = make_pipeline(
    TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS),
    FunctionTransformer(lambda x: x.astype('float'), validate=False)
)

tfidf_matrix = vectorizer.fit_transform(lecture_transcripts + assignment_descriptions)

# Calculate cosine similarity, could use other similarity function as well
cosine_similarities = cosine_similarity(tfidf_matrix[-len(assignment_descriptions):], tfidf_matrix[:-len(assignment_descriptions)])

# Match assignments to lectures based on cosine similarity
for i, assignment in enumerate(assignment_descriptions):
    relevant_lectures = cosine_similarities[i].argsort()[:-4:-1]  # Get top 3 most similar lectures
    print(f"\nAssignment: {assignment}")
    print("Relevant Lecture Videos:")
    for j in relevant_lectures:
        print(f"- Lecture {j + 1} (Similarity Score: {cosine_similarities[i][j]:.4f})")