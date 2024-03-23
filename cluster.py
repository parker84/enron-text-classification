from openai import OpenAI
from decouple import config
import pandas as pd
import hdbscan
from tqdm import tqdm
import os
import numpy as np
import coloredlogs, logging
from decouple import config

# --------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'))
client = OpenAI(api_key=config('OPENAI_API_KEY'))

# --------helpers
def get_cluster_summary(texts_in_cluster: list):
    system_prompt = """
    You are an expert at deciphering clusters of text by finding the common themes in them.
    I'm going to give you a number of emails that are all in the same cluster and you need to explain why these all fall into the same cluster.
    
    Do not just summarize all the values - but find the themes that common within every ticket in the cluster.
    
    Your job is to read through them and give me:
    1. A name for this cluster - based on the common themes that are present in every ticket in this cluster.
    2. Pull out the 3-5 main themes that are common for every ticket in this cluster and list them as 'tags' for this cluster.
    3. Give me an (extremely brief) summary of the common themes (present in every ticket) in this cluster.

    Here's the details from the first email in the cluster:
    """
    user_prompt = "\n\n------------------Next ticket in the cluster:\n".join(texts_in_cluster)
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    logger.info(f"Cluster summary: \n{response.choices[0].message.content}")
    return response.choices[0].message.content



def get_all_embeddings(df, model="text-embedding-3-small"):
    logger.info('Getting embeddings...')
    embeddings = client.embeddings.create(input=df['text'].tolist(), model=model).data
    df['embedding'] = [val.embedding for val in embeddings]
    logger.info('Done getting embeddings! ✅')
    return df

def get_clusters(df, min_cluster_size=5):
    logger.info('Clustering...')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(np.array([row for row in df['embedding']]))
    df['cluster'] = clusterer.labels_
    df['probs'] = clusterer.probabilities_
    logger.info('Done clustering! ✅')
    return df

def interpret_clusters(df):
    logger.info('Interpreting clusters...')
    cluster_counts = df.cluster.value_counts()
    logger.info(f'Cluster counts: {cluster_counts}')
    cluster_df = pd.DataFrame(cluster_counts)
    summaries = []
    top_texts_per_cluster = []
    for i in tqdm(cluster_counts.index):
        top_texts_per_cluster_df = df[df.cluster == i].sort_values(by='probs', ascending=False).head(10)
        summary = get_cluster_summary(top_texts_per_cluster_df['text'].tolist())
        summaries.append(summary)
        top_texts_per_cluster.append(top_texts_per_cluster_df['text'].tolist())
    cluster_df['top_texts'] = top_texts_per_cluster
    cluster_df['summary'] = summaries
    logger.info('Done interpreting clusters! ✅')
    logger.info(f'Cluster summaries: \n{cluster_df}')
    return cluster_df


if __name__ == "__main__":
    MAX_CHARACTERS_FOR_EMBEDDING = 5000
    TEXT_CSV_PATH = './data/processed/train.csv'
    EMBEDDING_CSV_PATH = './data/processed/embedded_emails.csv'
    if os.path.exists(EMBEDDING_CSV_PATH):
        logger.info('Embeddings already exist. Skipping embedding...')
        df = pd.read_csv(EMBEDDING_CSV_PATH)
    else:
        df = pd.read_csv(TEXT_CSV_PATH).sample(1000)
        df = df.rename(columns={'email_contents': 'text'})
        df['text'] = [val[:MAX_CHARACTERS_FOR_EMBEDDING] for val in df['text']]
        df = df.drop_duplicates(subset=['email_id'])
        logger.info(f'Input df shape: {df.shape}')
        df = get_all_embeddings(df)
        df.to_csv(EMBEDDING_CSV_PATH)
    df = get_clusters(df) # TODO: can we get more granular clusters?
    df.to_csv('data/processed/cluster_assignments.csv')
    cluster_df = interpret_clusters(df)
    cluster_df.to_csv('data/processed/cluster_summaries.csv')
    logger.info("Done! ✅")