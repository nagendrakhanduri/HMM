import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    hwords = test_set.get_all_Xlengths()
    

    for idWord in range (0, len(test_set.get_all_sequences())):
        pbWord = {}
        bestScore =  float ('-inf')
        
        guessWord = None
        X, lenWord = hwords [idWord]
        
        for word, model in models.items():
            score = float ('-inf')
            try:
                score = model.score(X, lenWord)
            except:
                #print ('Value erro')
                #print (score)
                #raise
                pass    
            pbWord[word] = score
            if score > bestScore:
                guessWord = word
                bestScore = score
        
        #print ("Adding " + str(pbWord))   
        probabilities.append(pbWord)
        guesses.append(guessWord)

                
    #print (len(probabilities)) 
    return probabilities, guesses
    
    
