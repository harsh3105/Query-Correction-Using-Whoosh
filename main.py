import os,os.path
import codecs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import operator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh import index,searching
from whoosh import qparser
from whoosh import highlight

path1="documents"
path="docs"
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
ix = index.create_in(path,schema)
writer=ix.writer()
dir=os.listdir(path1)
st= set(stopwords.words('english'))
hf = highlight.HtmlFormatter()

def queryformation(qstring,content):
    query_content=[]
    i=""
    wd=word_tokenize(qstring)
    for w in wd:
        if w not in st:
         i=i+" "+w
    query_content.append(i)
    X=CountVectorizer().fit_transform(query_content)
    t=TfidfTransformer(smooth_idf=False).fit_transform(X)
    te = []
    for i in t:
        te.append(list(i.A[0]))
    te=te[0]
    tef = [ '%.2f' % x for x in te]
    s=set(tef)
    #print(s)
    return s
    
    

def documentformation():
    content={}
    for item in dir:
        i=""
        k=[]
        xpath=os.path.join(path1,item)
        fill=codecs.open(xpath,encoding="utf-8",errors='ignore')
        wd=word_tokenize(fill.read())
        for w in wd:
            if w not in st:
                i=i+" "+w
        k.append(i)
        content[item]=k
        X=CountVectorizer().fit_transform(content[item])
        j=TfidfTransformer(smooth_idf=False).fit_transform(X)
        te=[]
        for i in j:
            te.append(list(i.A[0]))
        te=te[0]
        tef = [ '%.2f' % x for x in te]
        s=set(tef)
        content[item]=s
    return content

def cosine(content,tfidf_query):
    for key,value in content.items():
        #print(value)
        intersection = tfidf_query.intersection(value)
        union = tfidf_query or value
        x=float(len(union))
        c=float(len(intersection)/x)
        content[key]=c
    return content        

#make the query index
def queryindex():
    for item in dir:
        i=""
        xpath=os.path.join(path1,item)
        stry=codecs.open(xpath,encoding="utf-8",errors='ignore')
        wd=word_tokenize(stry.read())
        for w in wd:
            if w not in st:
                i=i+" "+w
        fileInfo={
        'title':item,
        'path':os.path.join(path,item),
        'content':i
        }
        writer.add_document(title=u'%s'%fileInfo['title'],path=u'%s'%fileInfo['path'],content=u'%s'%fileInfo['content'])
    writer.commit()

# Parse the user query string
def queryparse():
    c=True
    while c:
        print("Enter string")
        qstring=input()
        qp = qparser.QueryParser("content", ix.schema)
        q = qp.parse(qstring)
        
        with ix.searcher() as s:
            corrected = s.correct_query(q, qstring,)
            if corrected.query != q:
                print("Did you mean:", corrected.string)
                print(corrected.string)
                print("Enter yes or no [Y/N]")
                x=input()
                if(x=="Y"):
                    c=False
                    return corrected.string
            else:
                print(qstring)
                c=False
                return qstring
    
def main():
    queryindex()
    b=queryparse()
    content=documentformation()
    tfidf_query=queryformation(b,content)
    a=cosine(content,tfidf_query)
    sorted_a=sorted(a.items(),key=operator.itemgetter(1),reverse=True)
    for key,val in sorted_a:
        print ('{0:30} {1}'.format(key, val))
    
if __name__=="__main__": 
    main() 
    
    
