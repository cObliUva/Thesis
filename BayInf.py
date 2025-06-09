import pandas as pd
import statistics as stat
from LoT import else_, streak_, conform_, get_, patternCont_, balance_

# get the data from the 
bestRules = pd.read_csv("../Data/BestlowAcc500k.csv", dtype={"sequence": str, "p_pred": str, "m_pred": str})

# Load in the whole Excel file
all_data = pd.ExcelFile("../Data/PredictingOutcomes_ParticipantPredictions.xlsx")

# extract the worksheet of study 1B, and ensure the type of the sequence and prediction is a string
data_1B = pd.read_excel(all_data, 'Study 2B', dtype={"sequence": str, "prediction_raw": str})

# rename the participant ID column
data_1B = data_1B.rename(columns = {'participant_id' : 'p_id', 'prediction_raw' : 'p_pred'})

# get the relavant data columns
rdata = data_1B.loc[:, ['p_id', 'sequence', 'p_pred']]

# get the participants for which a rule was made (the training data)
toExclude = pd.unique(bestRules['p_id'])

# get the data from the participants not part of the training data
testData = rdata[[p not in toExclude for p in rdata['p_id']]]

# This part contains the bayesian inference which makes predictions on the test data

# create a new column called for the infered predictions
Inf_pred = []

# go over all the participants in the test data
for i in range(testData.shape[0]):

    # initialize the a list to store the predictions of the different rules on the current sequence
    pred = []

    # get the sequence, and define it as x for the evaluation rules for the current sequence
    x = testData.iloc[i]['sequence']

    # get the current rules
    relRules = bestRules[(bestRules['sequence'] == x)]

    """ Only use if you need the rules to make new predictions on sequences they were not fit on. """
    # loop over all rules that were made with the current sequence as training data
    #for rule in pd.unique(relRules['rule']):
        # evaluate the rule on the current sequence and store the result
        #pred.append(int(eval(rule)))

    # enter the prediction as the mean prediction of all relevant rules for the current sequence three 1's, and two 0's would yield '1'
    Inf_pred.append('1' if stat.mean(relRules['m_pred'].astype(int)) > 0.5 else '0')


# add the infered predictions to the dataframe
testData.insert(3,"inf_pred", Inf_pred)

print(testData)

# print the mean
print(stat.mean(testData['p_pred'] == testData['inf_pred']))
