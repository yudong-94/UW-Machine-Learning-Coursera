'''
Classification
    -Analyzing product sentiment
'''

import graphlab

##read some product review data
products = graphlab.SFrame('amazon_baby.gl/')

##explore the data
#print products.head()

##build the word count vector for each review
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
#print products.head()
#products['name'].show()

##explore Vulli Sophie (the most frequent item)
giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
#print len(giraffe_reviews)
##list frequency for each rating value
#giraffe_reviews['rating'].show(view='Categorical')

##build a sentiment classifier
#products['rating'].show(view='Categorical')
    ##most comments are positive
##define what's a postive and a negative sentiment
##ignore all 3-star reviews (neither positive nor negative)
products = products[products['rating'] != 3]
##positive sentiment = 4-star or 5-star reviews (if rating >=4, sentiment = 1)
products['sentiment'] = products['rating'] >= 4

##let's train the sentiment classifier
train_data, test_data = products.random_split(.8, seed=0)
sentiment_model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                                      features=['word_count'],validation_set=test_data)
                                                     
##evaluate the sentiment model
#print sentiment_model.evaluate(test_data, metric='roc_curve')
##shows number of false positive and false negative and confusion matrix                                                    
#sentiment_model.show(view='Evaluation')
    ##can see the relationship betwen FP and FN

##apply the learned model to understand sentiment of the Giraffe
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
#print giraffe_reviews.head()
    ##include a column of predicted sentiment

##sort the reviews based on the predicted sentiment (list the most positive reviews)                                    
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
#print giraffe_reviews.head()
##see the most positive review
#print giraffe_reviews[0]['review']
##see the most negative review
#print giraffe_reviews[-1]['review']
