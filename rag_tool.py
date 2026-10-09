### COMPLETE THE CODE  ###

from policy_loader import load_policy_documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

document_vectors = vectorizer.fit_transform(
    [doc.page_content for doc in documents]
)


def retrieve_documents(question, k=3):

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:k]:
        if scores[index] > 0:
            results.append(documents[index])

    return results