# preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from typing import Dict, Set

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
romanian_stopwords = set(stopwords.words('romanian'))

positive_words = set([
    'minunat', 'perfect', 'excelent', 'bun', 'frumos', 'superb', 'fantastic', 'grozav',
    'admirabil', 'agreabil', 'amabil', 'amuzant', 'atractiv', 'avantajos', 'binecuvântat',
    'benefic', 'brav', 'bravo', 'bucuros', 'calm', 'cald', 'celebru', 'chipeș', 'competent',
    'confortabil', 'convenabil', 'curajos', 'delicios', 'demn', 'deosebit', 'deștept', 'devotat',
    'dibaci', 'distractiv', 'divin', 'drăguț', 'eficient', 'elegant', 'emoționant', 'entuziast',
    'excepțional', 'extraordinar', 'fabulos', 'față', 'favorabil', 'fericire', 'fericit', 'fermecător',
    'fidel', 'folositor', 'formidabil', 'frumusețe', 'generos', 'genial', 'glorios', 'harnic',
    'ideal', 'ilustru', 'impecabil', 'impresionant', 'încântător', 'încrezător', 'ingenios',
    'inspirat', 'inteligent', 'interesant', 'întâi', 'învingător', 'înviorat', 'lăudabil',
    'liber', 'loial', 'luminos', 'luxos', 'magic', 'magnific', 'măreț', 'minune', 'miraculos',
    'modest', 'nobil', 'nou', 'onest', 'optimist', 'original', 'pasionant', 'perfect',
    'plăcut', 'popular', 'pozitiv', 'practic', 'prețios', 'priceput', 'productiv', 'profitabil',
    'profund', 'promițător', 'prosper', 'puternic', 'rapid', 'rafinat', 'realist', 'recunoscător',
    'remarcabil', 'respectabil', 'reușit', 'sănătos', 'satisfăcător', 'sărbătoare', 'seducător',
    'senzațional', 'serios', 'sigur', 'simpatic', 'sincer', 'smerit', 'special', 'spectaculos',
    'splendid', 'strălucit', 'sublim', 'succes', 'superior', 'surprinzător', 'talentat', 'tânăr',
    'tandru', 'uimitor', 'util', 'valoros', 'vesel', 'victorios', 'vioi', 'viu', 'voios', 'zâmbitor',
    'realizare', 'avantaj', 'beneficiu', 'binecuvântare', 'bucurie', 'câștig', 'cinste', 'curaj',
    'devotament', 'dragoste', 'entuziasm', 'fericire', 'generozitate', 'glorie', 'grație',
    'iubire', 'libertate', 'liniște', 'noroc', 'onoare', 'pace', 'plăcere', 'progres', 'prosperitate',
    'respect', 'satisfacție', 'speranță', 'succes', 'triumf', 'victorie', 'virtute', 'viață',
    'a admira', 'a ajuta', 'a aprecia', 'a bucura', 'a câștiga', 'a crea', 'a dezvolta', 'a excela',
    'a ferici', 'a iubi', 'a îmbunătăți', 'a împlini', 'a încuraja', 'a progresa', 'a prospera',
    'a proteja', 'a realiza', 'a respecta', 'a recomanda', 'a reuși', 'a salva', 'a sprijini',
    'foarte bun', 'cel mai bun', 'de top', 'de calitate', 'de încredere', 'merită', 'recomand'
])

negative_words = set([
    'rău', 'groaznic', 'prost', 'urât', 'teribil', 'jalnic', 'înfricoșător', 'dezgustător',
    'abuziv', 'agresiv', 'amenințător', 'anacronic', 'anormal', 'antipatic', 'arogant', 'aspru',
    'atroce', 'avar', 'banal', 'barbar', 'bizar', 'blestemat', 'bolnav', 'brutal', 'catastrofal',
    'chinuit', 'cinic', 'corupt', 'crud', 'cumplit', 'dăunător', 'debil', 'decadent', 'defect',
    'deplasat', 'deplorabil', 'deprimant', 'deranjant', 'deștept', 'detestabil', 'dezastruos',
    'dezgustător', 'dezolant', 'dificil', 'distructiv', 'dramatic', 'dubios', 'dureros', 'egoist',
    'enervant', 'escroc', 'exasperant', 'execrabil', 'fals', 'fatal', 'fragil', 'fricos', 'furios',
    'gol', 'grav', 'greșit', 'greu', 'hidos', 'hoț', 'îngrozitor', 'îngrijorat', 'idiot', 'ignorant',
    'ilogic', 'imatur', 'imoral', 'imperfect', 'imposibil', 'imprudent', 'inadecvat', 'inacceptabil',
    'incapabil', 'incompetent', 'incomplet', 'inconsistent', 'incorect', 'incredibil', 'indecent',
    'inferior', 'infernal', 'ingrat', 'injust', 'insuportabil', 'intolerabil', 'inutil', 'iresponsabil',
    'ironic', 'jegos', 'josnic', 'lacom', 'laș', 'leneș', 'lipsit', 'mânios', 'malefic', 'mediocru',
    'meschin', 'mincinos', 'mizerabil', 'mohorât', 'monoton', 'monstruos', 'murdar', 'naiv',
    'neadecvat', 'neatent', 'nebun', 'necinstit', 'necompetent', 'neconvingător', 'necuviincios',
    'nefericit', 'negativ', 'neglijent', 'neîndemânatic', 'neînsemnat', 'nelalocul', 'nemilos',
    'nenorocit', 'neplăcut', 'neproductiv', 'nervos', 'neserios', 'nesigur', 'nesociabil',
    'nesprijinit', 'netrebnic', 'nociv', 'obositor', 'obscur', 'odios', 'ofensator', 'oribil',
    'ostil', 'păcătos', 'penibil', 'periculos', 'pesimist', 'plictisitor', 'posac', 'praf',
    'prea', 'prejudicios', 'primitiv', 'problematic', 'răuvoitor', 'răzbunător', 'reacționar',
    'regretabil', 'respingător', 'retardat', 'ridicol', 'rigid', 'rușinos', 'sărac', 'sălbatic',
    'scandalos', 'scelerat', 'sfidător', 'sinistru', 'slab', 'sordid', 'spăimos', 'stupid',
    'supărător', 'suspect', 'tâmp', 'tensionat', 'trist', 'tulburat', 'turbulent', 'umilit',
    'urat', 'urâcios', 'urât', 'vexant', 'vicios', 'vinovat', 'violent', 'vulgar', 'zadarnic',
    'abandon', 'abuz', 'accident', 'acuzație', 'agresiune', 'amenințare', 'anarhie', 'anxietate',
    'bătaie', 'beznă', 'blestem', 'boală', 'brutalitate', 'calomnie', 'catastrofă', 'chin',
    'conflict', 'corupție', 'criză', 'cruzime', 'decădere', 'defect', 'depresie', 'dezamăgire',
    'dezastru', 'dezgust', 'distrugere', 'durere', 'eșec', 'exploatare', 'frică', 'frustare',
    'greșeală', 'haos', 'înfrângere', 'înșelăciune', 'lăcomie', 'lipsă', 'mânie', 'minciună',
    'mizerie', 'moarte', 'nedreptate', 'oboseală', 'pagubă', 'panică', 'pericol', 'pierdere',
    'problemă', 'rușine', 'scandal', 'slăbiciune', 'suferință', 'teamă', 'teroare', 'tragedie',
    'tristețe', 'ură', 'urâțenie', 'vină', 'violență',
    'a abandona', 'a abuzá', 'a acuza', 'a amenința', 'a ataca', 'a bate', 'a chinui', 'a critica',
    'a dăuna', 'a defăima', 'a denunța', 'a deprima', 'a deranja', 'a deteriora', 'a disprețui',
    'a distruge', 'a divide', 'a durea', 'a eșua', 'a exploata', 'a frânge', 'a fura', 'a îmbolnăvi',
    'a încălca', 'a înfrânge', 'a înjosi', 'a înșela', 'a întrista', 'a învinui', 'a jigni',
    'a lipsi', 'a minți', 'a murdări', 'a neglija', 'a părăsi', 'a pedepsi', 'a pierde', 'a provoca',
    'a răni', 'a respinge', 'a ruina', 'a sabota', 'a sfâșia', 'a speria', 'a suferi', 'a trăda',
    'a ucide', 'a umili', 'a urî',
    'deloc', 'nu este', 'lipsă de', 'fără sens', 'prea mult', 'prea puțin', 'nu recomand'
])

neutral_words = set([
    'este', 'sunt', 'avea', 'face', 'există', 'poate', 'trebuie', 'fi', 'vrea', 'putea',
    'știe', 'lua', 'da', 'vedea', 'merge', 'veni', 'spune', 'lucra', 'găsi', 'crede',
    'om', 'femeie', 'bărbat', 'copil', 'persoană', 'zi', 'an', 'timp', 'loc', 'mână',
    'ochi', 'cap', 'parte', 'număr', 'grup', 'problemă', 'punct', 'guvern', 'companie',
    'eu', 'tu', 'el', 'ea', 'noi', 'voi', 'ei', 'ele', 'mine', 'tine', 'lui', 'lor',
    'acesta', 'aceasta', 'acela', 'aceea', 'acestea', 'acelea', 'cine', 'ce', 'care', 'unde',
    'un', 'o', 'unei', 'unui', 'unor', 'la', 'în', 'pe', 'cu', 'de', 'din', 'pentru',
    'prin', 'peste', 'sub', 'între', 'către', 'despre', 'fără', 'până', 'după', 'înainte',
    'și', 'sau', 'dar', 'însă', 'ci', 'deși', 'dacă', 'când', 'unde', 'cum', 'că', 'ca',
    'unu', 'doi', 'trei', 'patru', 'cinci', 'șase', 'șapte', 'opt', 'nouă', 'zece',
    'primul', 'doilea', 'treilea', 'sută', 'mie', 'milion', 'miliard',
    'azi', 'ieri', 'mâine', 'luni', 'marți', 'miercuri', 'joi', 'vineri', 'sâmbătă', 'duminică',
    'ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august',
    'septembrie', 'octombrie', 'noiembrie', 'decembrie', 'oră', 'minut', 'secundă'
])

irony_patterns = set([
    'desigur', 'evident', 'sigur că', 'normal că', 'ce surpriză', 'la naiba', 'aproape că',
    'vai de mine', 'bravo', 'felicitări', 'minunat', 'extraordinar', 'fantastic', 'incredibil',
    'clar', 'logic', 'perfect', 'excelent', 'genial', 'superb', 'oh da', 'bineînțeles',
    'mare lucru', 'ce noroc', 'vai', 'doamne', 'dumnezeule', 'ce minune', 'ce frumos',
    'ce bine', 'foarte bine', 'super', 'wow', 'uau', 'no comment', 'fără cuvinte',
    'ce să zic', 'ce să spun', 'ce să mai zic', 'nu se poate', 'imposibil', 'incredibil',
    'fantastic de', 'minunat de', 'perfect de', 'sigur', 'cum să nu', 'da sigur',
    'evident că', 'normal', 'firește', 'poate', 'probabil', 'posibil', 'eventual',
    'vai și amar', 'săracul', 'bietul', 'mă rog', 'ce să faci', 'asta e', 'așa e viața',
    'nu mă miră', 'mare mirare', 'foarte surprins', 'cine ar fi crezut', 'să vezi și să nu crezi',
    'felicitări pentru', 'bravo ție', 'foarte deștept', 'ce inteligent', 'ce isteț',
    'foarte original', 'foarte util', 'foarte interesant', 'foarte important',
    'cel mai', 'foarte foarte', 'super super', 'mega', 'ultra', 'extra', 'hiper',
    'absolut', 'total', 'complet', 'în întregime', 'sută la sută', '100%',
    'nu-i așa?', 'da?', 'corect?', 'bine?', 'ok?', 'serios?', 'pe bune?', 'zău?',
    'chiar așa?', 'adevărat?', 'într-adevăr?', 'cu adevărat?'
])


def extract_key_features(text: str) -> Dict[str, float]:
    features = {}

    # Basic text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['unique_words_ratio'] = len(set(text.split())) / len(text.split()) if text.split() else 0

    # Punctuation features
    features['exclamation_marks'] = text.count('!')
    features['question_marks'] = text.count('?')
    features['ellipsis'] = text.count('...')
    features['quotes'] = text.count('"') + text.count("'") + text.count('„') + text.count('"')
    features['parentheses'] = text.count('(') + text.count('[') + text.count('{')
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    features['semicolon_count'] = text.count(';')
    features['colon_count'] = text.count(':')

    # Capitalization features
    words = text.split()
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1) / len(
        words) if words else 0
    features['title_case_words'] = sum(1 for word in words if word[0].isupper() and word[1:].islower()) / len(
        words) if words else 0

    # Sentiment features
    text_lower = text.lower()
    words_lower = text_lower.split()

    # Count sentiment words considering word boundaries
    positive_count = sum(1 for word in words_lower if word in positive_words)
    negative_count = sum(1 for word in words_lower if word in negative_words)
    neutral_count = sum(1 for word in words_lower if word in neutral_words)

    features['positive_word_count'] = positive_count
    features['negative_word_count'] = negative_count
    features['neutral_word_count'] = neutral_count
    features['positive_ratio'] = positive_count / len(words) if words else 0
    features['negative_ratio'] = negative_count / len(words) if words else 0
    features['sentiment_polarity'] = (positive_count - negative_count) / (positive_count + negative_count + 1)
    features['sentiment_subjectivity'] = (positive_count + negative_count) / (len(words) + 1)

    # Irony detection
    irony_count = 0
    for pattern in irony_patterns:
        if ' ' in pattern:  # Multi-word patterns
            irony_count += text_lower.count(pattern)
        else:  # Single word patterns
            irony_count += sum(1 for word in words_lower if word == pattern)

    features['irony_indicator_count'] = irony_count
    features['irony_ratio'] = irony_count / len(words) if words else 0

    # Additional linguistic features
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features[
                                                                                                 'sentence_count'] > 0 else 0

    # Repetition detection
    word_frequencies = {}
    for word in words_lower:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1

    features['word_repetition_ratio'] = sum(1 for count in word_frequencies.values() if count > 1) / len(
        word_frequencies) if word_frequencies else 0
    features['max_word_frequency'] = max(word_frequencies.values()) if word_frequencies else 0

    # Special character features
    features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(
        text) if text else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0

    # Emphasis detection
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))  # Characters repeated 3+ times
    features['emphasis_words'] = len(re.findall(r'\b\w*[A-Z]{2,}\w*\b', text))  # Words with consecutive caps

    return features


def get_external_sentiment_score(text: str) -> float:
    # Simple rule-based scoring as fallback
    words = text.lower().split()
    if not words:
        return 0.0

    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    # Consider intensifiers and negations
    intensifiers = {'foarte', 'extrem', 'deosebit', 'absolut', 'complet', 'mai'}
    negations = {'nu', 'niciodata', 'nici', 'fara', 'deloc'}

    for i, word in enumerate(words):
        if word in intensifiers and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in positive_words:
                positive_score += 0.5
            elif next_word in negative_words:
                negative_score += 0.5

        if word in negations and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in positive_words:
                positive_score -= 1
                negative_score += 0.5
            elif next_word in negative_words:
                negative_score -= 1
                positive_score += 0.5

    # Normalize score to [-1, 1]
    total = positive_score + negative_score
    if total == 0:
        return 0.0
    return (positive_score - negative_score) / total


def preprocess_for_tfidf(text: str) -> str:
    # Preserve emoticons before cleaning
    emoticon_mapping = {
        ':)': 'emoticon_happy',
        ':D': 'emoticon_veryhappy',
        ':(': 'emoticon_sad',
        ':|': 'emoticon_neutral',
        ';)': 'emoticon_wink',
        ':P': 'emoticon_tongue',
        ':o': 'emoticon_surprised',
        ':/': 'emoticon_confused',
        '<3': 'emoticon_heart',
        '</3': 'emoticon_brokenheart',
    }

    for emoticon, replacement in emoticon_mapping.items():
        text = text.replace(emoticon, f' {replacement} ')

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Keep some punctuation for context
    text = re.sub(r'[^\w\s!?.,;:\'\"-]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords but keep important sentiment words
    important_words = positive_words | negative_words | irony_patterns
    filtered = [w for w in words if w not in romanian_stopwords or w in important_words]

    return ' '.join(filtered)


# TextBlob for Romanian
def get_textblob_sentiment(text: str) -> Dict[str, float]:
    try:
        from textblob import TextBlob
        from textblob_ro import TextBlobRO

        blob = TextBlobRO(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except ImportError:
        print("TextBlob Romanian not installed. Using fallback method.")
        return {
            'polarity': get_external_sentiment_score(text),
            'subjectivity': 0.5
        }


def main():
    print("Loading data...")
    train_data = pd.read_csv('data/train.csv')
    validation_data = pd.read_csv('data/test.csv')
    test_data = pd.read_csv('data/stiri_digi24_alegeri.csv')

    train_data['content'] = train_data['content'].fillna('')
    test_data['contents'] = test_data['contents'].fillna('')
    validation_data['content'] = validation_data['content'].fillna('')

    print("Extracting enhanced features...")
    train_features = train_data['content'].apply(extract_key_features).apply(pd.Series)
    test_features = test_data['contents'].apply(extract_key_features).apply(pd.Series)
    validation_features = validation_data['content'].apply(extract_key_features).apply(pd.Series)

    # Add external sentiment scores
    print("Computing sentiment scores...")
    train_features['external_sentiment'] = train_data['content'].apply(get_external_sentiment_score)
    test_features['external_sentiment'] = test_data['contents'].apply(get_external_sentiment_score)
    validation_features['external_sentiment'] = validation_data['content'].apply(get_external_sentiment_score)

    print("Preprocessing text...")
    train_data['processed_text'] = train_data['content'].apply(preprocess_for_tfidf)
    test_data['processed_text'] = test_data['contents'].apply(preprocess_for_tfidf)
    validation_data['processed_text'] = validation_data['content'].apply(preprocess_for_tfidf)

    print("Creating enhanced TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True
    )

    X_train_tfidf = tfidf.fit_transform(train_data['processed_text'])
    X_test_tfidf = tfidf.transform(test_data['processed_text'])
    X_validation_tfidf = tfidf.transform(validation_data['processed_text'])

    # Combine all features
    X_train = np.hstack([X_train_tfidf.toarray(), train_features.values])
    X_validation = np.hstack([X_validation_tfidf.toarray(), validation_features.values])
    X_test = np.hstack([X_test_tfidf.toarray(), test_features.values])

    # Save processed data
    os.makedirs("processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv('processed/X_train.csv', index=False)
    pd.DataFrame(X_validation).to_csv('processed/X_validation.csv', index=False)
    pd.DataFrame(X_test).to_csv('processed/X_test.csv', index=False)
    train_data[['label']].to_csv('processed/y_train.csv', index=False)
    validation_data[['label']].to_csv('processed/y_validation.csv', index=False)
    test_data[['headline']].to_csv('processed/headlines_test.csv', index=False)

    # Save feature names for interpretability
    feature_names = list(tfidf.get_feature_names_out()) + list(train_features.columns)
    pd.DataFrame(feature_names, columns=['feature_name']).to_csv('processed/feature_names.csv', index=False)

    print(f"Preprocessing complete. Created {len(feature_names)} features.")
    print("Saved to 'processed/' folder.")


if __name__ == '__main__':
    main()