import plotly.express as px
import numpy as np
import pandas as pd
import hdbscan
import umap
import re
from sklearn.metrics.pairwise import pairwise_distances

class TopicExtractor:
    def __init__(self,wv,threshold) -> None:
        self.wv = wv
        self.word_vecs = None
        self.pre_trained = self._get_word_vec_df(self.wv.index_to_key,threshold)
        self.word_vecs= pd.DataFrame()
        
    def load_words(self, word_list, threshold):
        df = pd.DataFrame()
        df['word'] = word_list
        self.word_vecs = pd.concat([self.word_vecs,df.merge(self.pre_trained, on = 'word', how = 'left').dropna().query(f'mag > {threshold}').sort_values('mag')]).reset_index(drop=True)
        self.cluster()
    
    def load_seed_clusters(self, seed_words, known_labels,threshold):
        df = pd.DataFrame()
        df['word'] = seed_words
        df['pillar'] = known_labels
        pillars = df.merge(self.pre_trained, on = 'word', how = 'left').dropna().query(f'mag > {threshold}').sort_values('mag')
        self.word_vecs = pd.concat([self.word_vecs,pillars])
        self.word_vecs = self.word_vecs.drop_duplicates(keep='last',subset=['word']).reset_index(drop=True)
        self.cluster()
        
    def _get_word_vec_df(self,word_list,threshold):
        df = pd.DataFrame()
        df['word'] = word_list
        vectors = []
        for word in df['word']:
            vectors.append(self.wv[word])
        vectors = np.array(vectors)
        vecs = {f'v{i}':vector for i,vector in enumerate(vectors.T)}
        vecs=df.from_dict(vecs)
        df = pd.concat([df,vecs],axis=1)
        df['mag'] = [vector.dot(vector) for vector in df.iloc[:,1:].to_numpy()]
        df = df.sort_values(by='mag',ascending=True)
        return df.query(f'mag > {threshold}')
    
    def get_current_vecs(self):
        return self.word_vecs
    def clear_current_vecs(self):
        self.word_vecs = pd.DataFrame()
    def cluster(self):
        distance_matrix = pairwise_distances(self.word_vecs.loc[:,'v0':'v299'],metric='cosine')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4,metric= 'precomputed')
        self.word_vecs['labels'] = clusterer.fit_predict(distance_matrix.astype(np.double))
        self.clusterer = clusterer
    def get_clusterer(self):
        return self.clusterer

    def view_clusters(self):
        fit = umap.UMAP()
        if any(self.word_vecs.labels > -1):
            temp = self.word_vecs[self.word_vecs.labels > -1]
        else:
            temp = self.word_vecs
        if 'pillar' in temp:
            temp['pillar'] = temp.pillar.replace(np.NaN,'None')
        else:
            temp['pillar'] = 'None'
        u = fit.fit_transform(temp.loc[:,'v0':'v299'])
        return px.scatter(temp,x = u.T[0],y = u.T[1],hover_name='word',color='labels',symbol='pillar',labels={
                     "x": "principal component of text embeddings",
                     "y": "secondary component of text embeddings"
                 }).update_layout(legend_orientation="h")
    def list_clusters(self):
        labels = self.word_vecs.labels.unique()
        labels.sort()
        entries = []
        for label in labels:
            data_for_label = self.word_vecs[self.word_vecs.labels == label]
            try: 
                pillar = data_for_label[~data_for_label['pillar'].isna()].groupby('pillar').count().sort_values(by='word',ascending=False).iloc[0].name
            except:
                pillar = "UNK"
            top_tree_words = np.array(self.wv.most_similar(data_for_label.loc[:,'v0':'v299'].mean(axis=0).values)).T[0][:3]
            entries.append( {'label' : label, 'pillar' : pillar, 'words': top_tree_words})
        return pd.DataFrame(entries)
            
    def get_sentence_topics(self, sentence, threshold, learn_novel = False):
        sentence_vector = self.get_important_words(sentence,threshold).loc[:,'v0':'v299'].to_numpy().sum(axis=0)
        topics =  self.word_vecs[self.word_vecs.word.isin(np.array(self.wv.most_similar(sentence_vector,topn=10)).T[0])]
        if (not learn_novel) or (all(topics.labels>-1) and topics.shape[0] > 0):
            print('known-topics')
            return topics
        elif topics.shape[0] == 0:
            new_topic = self.pre_trained[self.pre_trained.word.isin(np.array(self.wv.most_similar(sentence_vector)).T[0])].sort_values(by='mag',ascending=False).iloc[0].word
            print('pre-trained')
        else:
            new_topic = topics[topics.labels == -1].iloc[0].word
            print('known-noise')
        print(new_topic)
        new_cluster_seed = self.pre_trained[self.pre_trained.word.isin(np.append(np.array(self.wv.most_similar(new_topic)).T[0],new_topic))].query(f'mag > {threshold-1}')
        new_cluster_seed[['pillar','target_words']] = new_topic
        self.word_vecs = pd.concat([self.word_vecs,new_cluster_seed])
        self.word_vecs = self.word_vecs.drop_duplicates(keep='last',subset=['word'])
        self.cluster()
        return self.get_sentence_topics(sentence,threshold)
    
    def get_important_words(self,sentence,threshold):
        df = pd.DataFrame()
        df['word'] = re.sub(r'[.,\'"]',r"",sentence).strip().split(" ")
        known_words = pd.Series(self.wv.index_to_key)
        df['found_words'] = [list(known_words[known_words.str.lower() == word.lower()]) for word in df['word']]
        df = df.explode('found_words')
        df['mag'] = [np.square(self.wv[str(word)]).sum() for word in df['found_words']]
        df = df.groupby('word').median('mag').query(f'mag > {threshold}').drop('mag',axis=1)
        return df.merge(self.pre_trained, left_on= 'word',right_on='word', how = 'left').dropna().sort_values('mag',ascending=False)