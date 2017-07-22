import json
import pandas as pd
import datetime
import re
import pickle
from geopy.geocoders import Nominatim
import pdb
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import trigrams
from collections import Counter
from textblob import TextBlob
from tqdm import tqdm


STATE_DICT = {'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AS': 'American Samoa','AZ': 'Arizona','CA': 'California',
               'CO': 'Colorado','CT': 'Connecticut','DC': 'District of Columbia','DE': 'Delaware','FL': 'Florida','GA': 'Georgia','GU': 'Guam',
               'HI': 'Hawaii','IA': 'Iowa','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana',
               'MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan','MN': 'Minnesota','MO': 'Missouri','MP': 'Northern Mariana Islands',
               'MS': 'Mississippi','MT': 'Montana','NA': 'National','NC': 'North Carolina','ND': 'North Dakota','NE': 'Nebraska','NH': 'New Hampshire',
               'NJ': 'New Jersey','NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio','OK': 'Oklahoma','OR': 'Oregon','PA': 'Pennsylvania',
               'PR': 'Puerto Rico','RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
               'UT': 'Utah','VA': 'Virginia','VI': 'Virgin Islands','VT': 'Vermont','WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia','WY': 'Wyoming'}

ABR = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

STATES = ['Alaska','Alabama','Arkansas','American Samoa',
          'Arizona','California','Colorado','Connecticut','District of Columbia','Delaware','Florida','Georgia','Guam',
          'Hawaii','Iowa','Idaho','Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland',
          'Maine','Michigan','Minnesota','Missouri','Northern Mariana Islands','Mississippi','Montana','National',
          'North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey','New Mexico','Nevada','New York',
          'Ohio','Oklahoma','Oregon','Pennsylvania','Puerto Rico','Rhode Island','South Carolina','South Dakota',
          'Tennessee','Texas','Utah','Virginia','Virgin Islands','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

POPULATIONS = {'Alabama':4858979,'Alaska':738432,'Arizona':6828065,'Arkansas':2978204,'California':39144818,'Colorado':5456574,
               'Connecticut':3590886,'Delaware':945934,'Florida':20271272,'Georgia':10214860,'Hawaii':1431603,'Idaho':1654930,'Illinois':12859995,
               'Indiana':6619680,'Iowa':3123899,'Kansas':2911641,'Kentucky':4425092,'Louisiana':4670724,'Maine':1329328,'Maryland':6006401,
               'Massachusetts':6794422,'Michigan':9922576,'Minnesota':5489594,'Mississippi':2992333,'Missouri':6083672,'Montana':1032949,
               'Nebraska':1896190,'Nevada':2890845,'New Hampshire':1330608,'New Jersey':8958013,'New Mexico':2085109,'New York':19795791,'North Carolina':10042802,
               'North Dakota':756927,'Ohio':11613423,'Oklahoma':3911338,'Oregon':4028977,'Pennsylvania':12802503,'Rhode Island':1056298,'South Carolina':4896146,
               'South Dakota':858469,'Tennessee':6600299,'Texas':27469114,'Utah':2995919,'Vermont':626042,'Virginia':8382993,'Washington':7170351,
               'West Virginia':1844128,'Wisconsin':5771337,'Wyoming':586107,'Puerto Rico': 3411307,'District of Columbia':681170}

PARTISAN = {'Alabama':('Republican', 14),'Alaska':('Republican', 13),'Arizona':('Republican', 5),'Arkansas':('Republican', 7),'California':('Democratic', 15),'Colorado':('Republican', 2),
               'Connecticut':('Democratic', 16),'Delaware':('Democratic', 20),'Florida':('Democratic', 3),'Georgia':('Republican', 4),'Hawaii':('Democratic', 14),'Idaho':('Republican', 25),'Illinois':('Democratic', 12),
               'Indiana':('Republican', 7),'Iowa':('Republican', 1),'Kansas':('Republican', 20),'Kentucky':('Democratic', 14),'Louisiana':('Democratic', 19),'Maine':('Democratic', 5),'Maryland':('Democratic', 26),
               'Massachusetts':('Democratic', 24),'Michigan':('Democratic', 7),'Minnesota':('Democratic', 5),'Mississippi':('Republican', 8),'Missouri':('Republican', 5),'Montana':('Republican', 18),
               'Nebraska':('Republican', 17),'Nevada':('Democratic', 3),'New Hampshire':('Republican', 3),'New Jersey':('Democratic', 13),'New Mexico':('Democratic', 15),'New York':('Democratic', 25),'North Carolina':('Democratic', 10),
               'North Dakota':('Republican', 11),'Ohio':('Republican', 1),'Oklahoma':('Democratic', 1),'Oregon':('Democratic', 9),'Pennsylvania':('Democratic', 12),'Rhode Island':('Democratic', 30),'South Carolina':('Republican', 5),
               'South Dakota':('Republican', 12),'Tennessee':('Republican', 12),'Texas':('Republican', 4),'Utah':('Republican', 33),'Vermont':('Democratic', 16),'Virginia':('Republican', 2),'Washington':('Democratic', 8),
               'West Virginia':('Democratic', 20),'Wisconsin':('Democratic', 2),'Wyoming':('Republican', 47),'Puerto Rico': ('Democratic', 20),'District of Columbia':('Democratic', 20)}


def load_tweets(filename):
    tweets = []
    for line in open('raw_datainc.json', 'r'):
        try:
            tweets.append(json.loads(line))
        except:
            continue
    return tweets


def create_df(tweets):
    """ We will look at the following tweet data: creation time, language, hashtags, userid and location
        Tweets which do not have required information are removed from list.
        Remaining tweets stored into a dataframe
    """
    for tweet in tweets:
        try:
            tweet['created_at']
            tweet['lang']
            tweet['entities']['hashtags']
            tweet['user']['id']
            tweet['user']['location']
        except:
            tweets.remove(tweet)

    df = pd.DataFrame()
    df['time'] = [tweet['created_at'] for tweet in tweets]
    df['time'] = pd.to_datetime(df['time'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
    df['hashtags'] = [[x['text'] for x in tweet['entities']['hashtags']] for tweet in tweets]
    df['lang'] = [tweet['lang'] for tweet in tweets]
    df['text'] = [tweet['text'] for tweet in tweets]
    df['userid'] = [tweet['user']['id'] for tweet in tweets]
    df['location'] = [tweet['user']['location'] for tweet in tweets]
    return df[df['lang']=='en']    # remove non english tweets


def extract_location(df):
    """ Trying to catch states the sender says to be living in (most tweets not geolocated) """
    strlist_1 = [x.lower() for x in STATES]
    strlist_2 =  [x.lower() for x in ABR]

    regex_1 = r"(?=\s("+'|'.join(strlist_1)+r"))"
    regex_2 = r"(?=\s("+'|'.join(strlist_2)+r")$)"
    regex_3 = r"(?=\s("+'|'.join(strlist_2)+r")[\s|\W])"
    regex_4 = r"(?=("+'|'.join(strlist_1)+r"),)"
    regex_5 = r"(?=,("+'|'.join(strlist_1)+r"))"
    regex_6 = r"(?=,s*("+'|'.join(strlist_2)+r"))"

    df['location'] = df['location'].fillna(value='unknown')
    locations = list(pd.unique(df['location']))
    location_dict = {}
    for item in locations:
        first_pass = re.findall(regex_1, item.lower())
        if first_pass != []:
            location_dict[item] = first_pass[0]
            continue
        second_pass = re.findall(regex_2, item.lower())
        if second_pass != []:
            location_dict[item] = STATE_DICT[second_pass[0].upper()]
            continue
        third_pass = re.findall(regex_3, item.lower())
        if third_pass != []:
            location_dict[item] = STATE_DICT[third_pass[0].upper()]
            continue
        fourth_pass = re.findall(regex_4, item.lower())
        if fourth_pass != []:
            location_dict[item] = fourth_pass[0]
            continue
        fifth_pass = re.findall(regex_5, item.lower())
        if fifth_pass != []:
            location_dict[item] = fifth_pass[0]
            continue
        sixth_pass = re.findall(regex_6, item.lower())
        if sixth_pass != []:
            location_dict[item] = sixth_pass[0]
            continue
        if item.lower() in strlist_1:
            location_dict[item] = item.lower()
            continue
        if item.lower() in strlist_2:
            location_dict[item] = STATE_DICT[item.upper()]
            continue
    df['location'] = df['location'].map(location_dict)
    tagged = df[~pd.isnull(df['location'])]

    loc = []
    for item in tagged['location']:
        if (len(item.split()) == 1) and (len(item) != 2):
            loc.append(item[0].upper() + item[1:])
        elif (len(item.split()) == 2) and (len(item) != 2):
            loc.append(item.split()[0][0].upper() + item.split()[0][1:] +\
                       ' ' + item.split()[1][0].upper() + item.split()[1][1:])
        elif len(item) == 2:
            loc.append(STATE_DICT[item.upper()])
        elif len(item.split()) == 3:
            loc.append('District of Columbia')
        else:
            loc.append(item)
    tagged['location'] = loc
    return df, tagged


def state_population(df):
    df['population'] = df['location']
    df['population'] = df['population'].map(POPULATIONS)
    return df


def state_corpus(tagged):
    """ Create a separate corpus for each state, that concatenate all its tweets """
    corpus = {}
    for state in pd.unique(tagged['location']):
        corpus[state] = '\n'.join(tagged[tagged['location']==state]['text'].tolist())
    return corpus


def pickle_corpus():
    filename = 'raw_datainc.json'
    tweets = load_tweets(filename)
    df = create_df(tweets)
    df, tagged = extract_location(df)
    df = state_population(df)
    corpus = state_corpus(tagged)

    with open('corpus.pck', 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def tokenize(corpus):
    tokens = {}
    tknzr = TweetTokenizer()
    for item in tqdm(corpus):
        #remove punctuation
        punctuation = str.maketrans(dict.fromkeys('…,.?!:;/-&"'))
        corpus[item] = corpus[item].translate(punctuation)
        tokens[item] = tknzr.tokenize(corpus[item].lower())
        #remove stopwords
        tokens[item] = [x for x in tokens[item] if x!='rt']
        tokens[item] = [word for word in tokens[item] if word not in stopwords.words('english')]
    return tokens


def bokeh_df(tagged, tokens, corpus):
    """ dataframe to be visualized: contains tweet count for each state, sentiment polarity, and most
        frequent words and trigrams """
    #tweet count
    tweets_nbr = tagged.groupby('location', as_index=False)['userid'].count().rename(columns={'userid': 'count'})
    tagged = pd.merge(tagged, tweets_nbr, how='left', on='location')
    # term frequency
    term = {}
    trigram = {}
    for state in tokens:
        term_counts = Counter(tokens[state])
        tgrams = trigrams(tokens[state])
        tgrams_counts = Counter(tgrams)
        term[state] = term_counts.most_common(3)
        trigram[state] = tgrams_counts.most_common(3)
        
    wordfreq = tweets_nbr.copy()
    wordfreq['1rst term'] = [term[wordfreq['location'].iloc[i]][0][0] for i in range(len(wordfreq))]
    wordfreq['2nd term'] = [term[wordfreq['location'].iloc[i]][1][0] for i in range(len(wordfreq))]
    wordfreq['3rd term'] = [term[wordfreq['location'].iloc[i]][2][0] for i in range(len(wordfreq))]
    wordfreq['1rst wcount'] = [term[wordfreq['location'].iloc[i]][0][1] for i in range(len(wordfreq))]
    wordfreq['2nd wcount'] = [term[wordfreq['location'].iloc[i]][1][1] for i in range(len(wordfreq))]
    wordfreq['3rd wcount'] = [term[wordfreq['location'].iloc[i]][2][1] for i in range(len(wordfreq))]
    wordfreq['1rst tgrm'] = [trigram[wordfreq['location'].iloc[i]][0][0] for i in range(len(wordfreq))]
    wordfreq['2nd tgrm'] = [trigram[wordfreq['location'].iloc[i]][1][0] for i in range(len(wordfreq))]
    wordfreq['3rd tgrm'] = [trigram[wordfreq['location'].iloc[i]][2][0] for i in range(len(wordfreq))]
    wordfreq['1rst tcount'] = [trigram[wordfreq['location'].iloc[i]][0][1] for i in range(len(wordfreq))]
    wordfreq['2nd tcount'] = [trigram[wordfreq['location'].iloc[i]][1][1] for i in range(len(wordfreq))]
    wordfreq['3rd tcount'] = [trigram[wordfreq['location'].iloc[i]][2][1] for i in range(len(wordfreq))]
    wordfreq.drop('count', axis=1, inplace=True)
    tagged = pd.merge(tagged, wordfreq, how='left', on='location')
    # sentiment analysis
    sentiment = {}
    for state in tqdm(pd.unique(tagged['location'])):
        blob = TextBlob(corpus[state])
        sentiment[state] = blob.sentiment.polarity
    tagged['sentiment'] = [sentiment[tagged['location'].iloc[i]] for i in range(len(tagged))]

    # partisan split from Gallup
    tagged = tagged[~pd.isnull(tagged['population'])]
    tagged['main_party'] = [PARTISAN[tagged['location'].iloc[i]][0] for i in range(len(tagged))]
    tagged['partisan_split'] = [PARTISAN[tagged['location'].iloc[i]][1] for i in range(len(tagged))]

    return tagged


#pickle_corpus()
with open('corpus.pck', 'rb') as handle:
    corpus = pickle.load(handle)

#tokens = tokenize(corpus)
with open('tokens.pck', 'rb') as handle:
    tokens = pickle.load(handle)

filename = 'raw_datainc.json'
tweets = load_tweets(filename)
df = create_df(tweets)
df, tagged = extract_location(df)
tagged = state_population(tagged)

final = bokeh_df(tagged, tokens, corpus)

#final.to_pickle('final.pck')
final.to_csv('final.csv', encoding = 'utf-8-sig')

#final = pd.read_pickle('final.pck')