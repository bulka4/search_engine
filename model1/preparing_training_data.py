


def clean_text(string: str, 
               punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
               stop_words = stopwords.words('english'),
               porter = PorterStemmer()
              ):
    """
    A method to clean text 
    """
    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])
    
    # stemming words. That means changing word to its basic format, for example
    # words 'fishing', 'fished', 'fischer' will be changed into a word 'fisch'
    string = ' '.join([porter.stem(word) for word in string.split()])

    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    return string

