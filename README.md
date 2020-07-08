# AlphabetSoup

From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received various amounts of funding from Alphabet Soup over the years. We will build our own machine learning model that will be able to predict the success of a venture paid by Alphabet Soup. 

1. Import, analyze, clean, and preprocess the classification dataset.
2. Select, design, and train a binary classification model of your choosing.
3. Optimize model training and input data to achieve desired model performance.

## Resources

- Data Source: charity_data.csv
- Software: Python 3.7.6, Jupyter Lab 1.2.6
- Libraries: Sklearn, Pandas, Tensorflow

## Summary

As part of cleaning and preprocessing the data, we made the following changes to our dataset:

        - Removed the *NAME, CLASSIFICATION, STATUS, AND SPECIAL CONSIDERATIONS* columns
        - Checked if there were any null values
        - Binned the *ASK_AMT* into two categories (5000 and Other)
        - Encoded the *ASK_AMT* column
        - One Hot Encoded the rest of the columns we identified as features
        - Checked if we have any unique values

After the changes were made, the data was trained, scaled and ready for our model.

### 1. How many neurons and layers did you select for your neural network model? Why?

In our initial attempt to run the model, we had 43 input features. As such, we had chosen to reflect that number in the number of neurons used. We also decided to use 2 hidden layers as there were a lot more data to filter through. As well, by adding another layer, it will allow the model to better classify the data and hopefully with higher accuracy. We split the number of neurons so that the first layer had 30 and the second layer had 13.

### 2. Were you able to achieve the target model performance? What steps did you take to try and increase model performance?

With 2 layers and 43 neurons, the model was not able to achieve a target predictive accuracy highe than 75%. Our model was only able to achieve 72.55% accuracy and 56.90% for our loss metric.

In attempt to increase model performance, we took the following steps:

   1. Instead of 1 output layer, we encoded the column *IS_SUCCESSFUL* resulting in 2 output layers. We kept the numnber of hidden layers, neurons and epochs the same. We test the data and resulted in a 57.54% for loss and 72.00% for accuracy.    
   2. In our second attempt, we added additional neurons. We doubled the number of input features and split them evenly between the 2 hidden layers. As a result, we received 58.37% for loss and 71.91% for accuracy.     
   3. In our third attempt, since the model with additional neurons did not do so well, we reduced the number of neurons back to our original model with 43 neurons. However, instead of 2 hidden layers, we added a third one and split the neurons to 30 for the first layer, 10 for the second and 3 for the third. We received 58.04% for loss and 71.80% for accuracy.      
   4. In our fourth attempt, we kept everything consistent from our third model and added additional epochs to a total of 200. With this model, we received 57.79% for loss and 72.05% for accuracy.

### 3. If you were to implement a different model to solve this classification problem, which would you choose? Why?

Looking at the results we got from the model and the different steps we took to improve model performance, we decided to test the data with different models, such as the Random Forest Classifier and the Logistic Regression with different solvers. If we were to implement a different model, we would consider the Random Forest Classifier as the model accuracy was very close to the accuracy of our initial model before the changes. Not only were the accuracy similar between both models, running the Random Forest Classifier model is a lot faster and could allow us to spend more time on our dataset. 

One thing we may want to look at, aside from the dataset, is the minor overfitting for our models. When we run our model with the training data, we get 73% for accuracy. However, when we run our model with the testing data, the accuracy drops to 72%. In future models, we can spend some time to see what is affecting this slight overfitting.

## Usage

**Note:** Please ensure you have all the required and updated softwares on your computer.

1. Download the following files into the same folder for the project.

        - charity_data.csv
        - AlphabetSoupChallenge.ipynb

2. In your Anaconda prompt, activate your development environment and navigate to the folder holding the above files and run Jupyter Lab. 

3. You can now run all the cells to see their outputs. If you want to test out the results, you can change the parameters.

