from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_sqlalchemy import SQLAlchemy
import spacy
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import os

# Créer l'application Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///articles.db'
db = SQLAlchemy(app)
nlp = spacy.load("fr_core_news_sm")

# Définir les modèles de base de données
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)

class IgnoredWord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(80), unique=True, nullable=False)

# Route pour afficher la page principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour ajouter un article
@app.route('/add_article', methods=['POST'])
def add_article():
    content = request.json.get('content')
    if content:
        article = Article(content=content)
        db.session.add(article)
        db.session.commit()
        return jsonify({'message': 'Article added successfully!'}), 201
    return jsonify({'message': 'Content is missing!'}), 400

# Route pour ajouter un mot à ignorer
@app.route('/add_ignored_word', methods=['POST'])
def add_ignored_word():
    word = request.json.get('word')
    if word:
        ignored_word = IgnoredWord(word=word)
        db.session.add(ignored_word)
        db.session.commit()
        return jsonify({'message': 'Word added to ignored list!'}), 201
    return jsonify({'message': 'Word is missing!'}), 400

# Route pour supprimer un mot de la liste des mots à ignorer
@app.route('/remove_ignored_word', methods=['POST'])
def remove_ignored_word():
    word = request.json.get('word')
    if word:
        ignored_word = IgnoredWord.query.filter_by(word=word).first()
        if ignored_word:
            db.session.delete(ignored_word)
            db.session.commit()
            return jsonify({'message': 'Word removed from ignored list!'}), 200
        return jsonify({'message': 'Word not found!'}), 404
    return jsonify({'message': 'Word is missing!'}), 400

# Route pour analyser les articles
@app.route('/analyze', methods=['GET'])
def analyze():
    articles = Article.query.all()
    all_texts = ' '.join([article.content for article in articles])
    doc = nlp(all_texts)
    ignored_words = [iw.word for iw in IgnoredWord.query.all()]
    common_nouns = [token.text for token in doc if token.pos_ == 'NOUN' and token.text not in ignored_words]
    noun_freq = Counter(common_nouns)
    
    G = nx.Graph()
    for noun, freq in noun_freq.items():
        G.add_node(noun, size=freq)
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    sizes = [G.nodes[node]['size'] * 100 for node in G]
    
    nx.draw(G, pos, with_labels=True, node_size=sizes, node_color='skyblue', edge_color='gray', font_size=10, font_weight='bold')
    graph_path = 'static/graph.png'
    plt.savefig(graph_path)
    
    return jsonify({'message': 'Analysis complete! Graph saved as ' + graph_path}), 200

# Route pour servir les fichiers statiques
@app.route('/static/<path:filename>')
def send_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
