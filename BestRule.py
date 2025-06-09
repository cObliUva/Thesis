import pandas as pd
import re
import statistics as stat
import math
from LoT import else_, streak_, conform_, get_, patternCont_, balance_

def predictOutcome(results):
    # keep track of the predictions
    pred = []

    # go over each row
    for i in range(results.shape[0]):

        # define the sequence as x for the rule
        x = results['sequence'].iloc[i]

        # evaluate the rule on the current sequence and store the result
        pred.append(eval(results['rule'].iloc[i]))

    # load in the models predictions
    results['m_pred'] = pred

    # add a column to indicate whether the participants answer is congruent with our models prediction
    results['correct'] = results['m_pred'] == results['p_pred']

    return results


def bestRule(results):
    # initialize lists
    means = []
    par = []
    rull = []
    post = []

    # loop over all participants, and all their respective rules
    for p in pd.unique(results['p_id']):
        pard = results[results['p_id'] == p]
        for rule in pd.unique(pard['rule']):

            # get the data for the current rule
            ruled = pard[pard['rule'] == rule]

            # append the data to the created lists
            means.append(ruled['correct'].mean())
            par.append(p)
            rull.append(rule)
            post.append(pd.unique(ruled['posterior']).item())

    # fill a new dataframe with the extracted values
    bRule = pd.DataFrame({'p_id' : par, 'rule' : rull, 'posterior' : post, 'mean' : means})

    # sort values by mean and then posterior
    bRule = bRule.sort_values(by = ['mean', 'posterior'], ascending=False)

    # create a empty list for the best rules
    bestRules = pd.DataFrame({'p_id' : [], 'rule' : []})

    # loop over each participant
    for p in pd.unique(bRule['p_id']):

        # create a new dataframe with the id and best rule for the current participant, which is the first row due to sorting
        brule = pd.DataFrame({'p_id' : [p], 'rule' : [bRule[bRule['p_id'] == p].iloc[0]['rule']]})
        
        # add the dataframe of 1 row to the rest
        bestRules = pd.concat([bestRules, brule])

    # get the subset the results dataframe by only retaining the best rule for each participant
    bestRules =  pd.merge(results, bestRules, on=['p_id', 'rule'], how='inner')

    return bestRules

# only run this code if the program is run with this file as the main
if __name__ == '__main__':

    # load in LoT model results
    mvData = pd.read_csv("../Data/lowAcc500k.csv")

    # remove unnessesary string elements
    mvData['rule'] = [re.sub('"', '', rule)[10:] for rule in mvData['rule']]

    # Load in the whole Excel file
    all_data = pd.ExcelFile("../Data/PredictingOutcomes_ParticipantPredictions.xlsx")

    # extract the worksheet of study 1B, and ensure the type of the sequence and prediction is a string
    data_1B = pd.read_excel(all_data, 'Study 2B', dtype={"sequence": str, "prediction_raw": str})

    # rename the participant ID column
    data_1B = data_1B.rename(columns = {'participant_id' : 'p_id', 'prediction_raw' : 'p_pred'})

    # get the data from the sampled participants for which a rule was made using a boolean mask
    rdata = data_1B[[p in pd.unique(mvData['p_id']) for p in data_1B['p_id']]]
        
    # get the relevant data columns
    rdata = rdata.loc[:, ['p_id', 'sequence', 'p_pred']]

    # merge the dataframes
    results = pd.merge(rdata, mvData[['p_id', 'rule', 'posterior']], on = 'p_id')

    # use the found rules of each participant to predict the outcome for their sequences
    results_pred = predictOutcome(results)

    # get the best rules from the results
    BRule  = bestRule(results_pred)

    # print the total average accuracy
    print(BRule['correct'].mean())

    # print the mean accuracy of each participant
    print(BRule.groupby(['p_id'])['correct'].mean().to_string())

    print(BRule.to_string())

    # store the data of the best rules
    BRule.to_csv("../Data/BestlowAcc500k.csv", index=False)





        
            


