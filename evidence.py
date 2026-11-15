"""Bounded read-only queries and local TF-IDF source retrieval."""
import datetime as dt
import json
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer

class Evidence:
    def __init__(self, db_path='data/fire.sqlite', corpus_path=None):
        self.db_path = Path(db_path).resolve()
        self.corpus = json.loads(Path(corpus_path or Path(__file__).parent/'docs/corpus.json').read_text())
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform([p['section']+' '+p['text'] for p in self.corpus])

    def _query(self, sql, params=()):
        with sqlite3.connect(self.db_path.as_uri()+'?mode=ro',uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql,params)]

    def get_weekly_counts(self,start,end,region='study-box'):
        if region != 'study-box': raise ValueError('Supported geography: study-box, 37–42 N, 125–119 W')
        a,b = dt.date.fromisoformat(start),dt.date.fromisoformat(end)
        if a>b or a<dt.date(2023,1,1) or b>dt.date(2025,12,31) or (b-a).days>1095:
            raise ValueError('Use an ordered date window within 2023–2025')
        return self._query('SELECT * FROM weekly WHERE date(week_start) BETWEEN ? AND ? ORDER BY week_start LIMIT 156',(a.isoformat(),b.isoformat()))

    def get_model_metrics(self):
        return self._query('SELECT * FROM metrics ORDER BY model LIMIT 3')

    def search_docs(self,query,top_k=3):
        if not isinstance(query,str) or not query.strip() or len(query)>2000: raise ValueError('Invalid search query')
        if type(top_k) is not int or not 1<=top_k<=3: raise ValueError('top_k must be 1–3')
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return [{**self.corpus[i],'score':float(scores[i])} for i in scores.argsort()[::-1][:top_k] if scores[i]>0]
