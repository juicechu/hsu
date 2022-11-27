import numpy as np
import pandas as pd
import datetime
from pymongo import MongoClient
from itertools import combinations
import pickle
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn import preprocessing
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class GroupProb:
    def __init__(self):
        self.probs = []

class WorldCUPPredictor:
    def __init__(self):
        ranking_df = pd.read_csv('./world_cup/fifa_ranking-2022-10-06.csv')
        ranking_df = ranking_df.replace({"IR Iran": "Iran"})
        ranking_df['rank_date'] = pd.to_datetime(ranking_df['rank_date'])
        ranking_df['country_full'] = ranking_df[['country_full']].replace({'Korea Republic': 'South Korea', 'Korea DPR': 'North Korea'})
        print(ranking_df.head())
        self.ranking_df = ranking_df;


    def fifa_rank_and_point(self, x, name='home_team'):
        ranking_df = self.ranking_df
        team_ranking = ranking_df[ranking_df['country_full'] == x[name]]
        team_ranking['date_diff'] = x['date'] - team_ranking['rank_date']
        team_ranking = team_ranking[(x['date'] - team_ranking['rank_date']) >= datetime.timedelta()]
        if team_ranking.shape[0] == 0 :
            return None
        team_ranking = team_ranking.sort_values(by=['date_diff'])
        #print(team_ranking.head())
        return team_ranking.iloc[0]
    def fifa_data(self, x):
        s = pd.Series(data={
            'home_team_fifa_rank': 0,
            'home_team_fifa_points': 0,
            'away_team_fifa_rank': 0,
            'away_team_fifa_points': 0
        })
        home = self.fifa_rank_and_point(x, 'home_team')
        away = self.fifa_rank_and_point(x, 'away_team')
        if home is not None:
            s['home_team_fifa_rank'] = home['rank']
            s['home_team_fifa_points'] = home['total_points']
        if away is not None:
            s['away_team_fifa_rank'] = away['rank']
            s['away_team_fifa_points'] = away['total_points']
        #return pd.concat(home, away)
        return s

    def write_to_mongodb(self):
        ranking_df = self.ranking_df
        match_df = pd.read_csv('./world_cup/results.csv')
        #match_df =  match_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
        match_df['date'] = pd.to_datetime(match_df['date'])
        match_df = match_df[match_df['date'] > datetime.datetime(1992,12,31)]
        print(match_df.loc[~match_df['home_team'].isin(ranking_df['country_full']), 'home_team'].value_counts())
        print(match_df.head())
        match_df['home_team'] = match_df[['home_team']].replace({
            'DR Congo': 'Congo DR',
            'United States': 'USA',
            'Saint Kitts and Nevis': 'St. Kitts and Nevis',
            'Romani people': 'Romania',
            'Cape Verde': 'Cape Verde Islands',
            'Kyrgyzstan': 'Kyrgyz Republic',
            'Saint Lucia': 'St. Lucia',
            'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
        })
        match_df['away_team'] = match_df[['away_team']].replace({
            'DR Congo': 'Congo DR',
            'United States': 'USA',
            'Saint Kitts and Nevis': 'St. Kitts and Nevis',
            'Romani people': 'Romania',
            'Cape Verde': 'Cape Verde Islands',
            'Kyrgyzstan': 'Kyrgyz Republic',
            'Saint Lucia': 'St. Lucia',
            'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
        })

        #print(match_df.loc[~match_df['home_team'].isin(ranking_df['country_full']), 'home_team'].value_counts())
        match_df = match_df[match_df['home_team'].isin(ranking_df['country_full'])]
        match_df = match_df[match_df['away_team'].isin(ranking_df['country_full'])]
        match_df = match_df.dropna()
        #match_df = match_df.iloc[0:100] #test for small dataset

        #match_df['home_team_fifa_rank'] = match_df.apply(lambda x: fifa_rank(x, 0), axis=1)
        fifa_df = match_df.apply(lambda x: self.fifa_data(x), axis=1, result_type='expand')
        match_df = pd.concat([match_df, fifa_df], axis=1)
        match_df = match_df[(match_df['home_team_fifa_rank'] > 0)
                            & (match_df['home_team_fifa_points'] > 0)
                            & (match_df['away_team_fifa_rank'] > 0)
                            & (match_df['away_team_fifa_points'] > 0)
                            ] #filter
        match_df['rank_diff'] = match_df['home_team_fifa_rank'] - match_df['away_team_fifa_rank']
        match_df['rank_avg'] = (match_df['home_team_fifa_rank'] + match_df['away_team_fifa_rank']) / 2
        match_df['points_diff'] = match_df['home_team_fifa_points'] - match_df['away_team_fifa_points']
        match_df['score_diff'] = match_df['home_score'] - match_df['away_score']
        def isWin(x):
            if x > 0 :
                return 1
            elif x == 0:
                return 0
            elif x < 0:
                return -1
        match_df['is_win'] = match_df['score_diff'].map(lambda x:isWin(x))
        match_df['not_friendly'] = match_df['tournament'] != 'Friendly'
        match_df['is_worldcup'] = 'World Cup' in match_df['tournament']

        ##insert to mongodb
        mongo = MongoClient()
        international_match = mongo.com6002.international_match
        international_match.drop()
        international_match.insert_many(match_df.to_dict('records'))

    def train_model(self):
        mongo = MongoClient()
        international_match = mongo.com6002.international_match
        colls = international_match.find()
        df = pd.DataFrame(colls)
        df['not_friendly'] = df['not_friendly'].astype(int)
        df['is_worldcup'] = df['is_worldcup'].astype(int)
        df['is_win'] = df['is_win'].astype(int)
        print(df['not_friendly'].value_counts())
        print(df['is_worldcup'].value_counts())
        print(df['is_win'].value_counts())
        print(df.isna().sum())
        df.info()
        df_X, df_Y = df.loc[:,['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup']], df['is_win']
        df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=12345)


        """
        # Logistic Regression
        logreg = LogisticRegression()
        logreg.fit(df_X_train, df_Y_train)
        accuracy = round(logreg.score(df_X_test, df_Y_test) * 100, 5) #Return the mean accuracy on the given test data and labels.
        print('Logistic Regression Accuracy: ' + str(accuracy))
        Y_pred = logreg.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))

        # Support Vector Machines
        svc = SVC()
        svc.fit(df_X_train, df_Y_train)
        accuracy = round(svc.score(df_X_test, df_Y_test) * 100, 5)
        print('SVC Accuracy: ' + str(accuracy))
        print('Confusion Matrix')
        Y_pred = svc.predict(df_X_test)
        print(confusion_matrix(df_Y_test, Y_pred))


        # KNN
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(df_X_train, df_Y_train)
        accuracy = round(knn.score(df_X_test, df_Y_test) * 100, 5)
        print('KNN Accuracy: ' + str(accuracy))
        Y_pred = knn.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))

        # Decision Tree
        dtree = DecisionTreeClassifier()
        dtree.fit(df_X_train, df_Y_train)
        accuracy = round(dtree.score(df_X_test, df_Y_test) * 100, 5)
        print('Decision Tree Accuracy: ' + str(accuracy))
        Y_pred = dtree.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))

        # Random Forest
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(df_X_train, df_Y_train)
        accuracy = round(forest.score(df_X_test, df_Y_test) * 100, 5)
        print('Random Forest Accuracy: ' + str(accuracy))
        Y_pred = forest.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(df_X_train, df_Y_train)
        accuracy = round(nb.score(df_X_test, df_Y_test) * 100, 5)
        print('Naive Bayes Accuracy: ' + str(accuracy))
        Y_pred = nb.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))
        """
        # Neural Network
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(3, 2), random_state=12345)
        scaler = preprocessing.StandardScaler()
        df_X_train = pd.DataFrame(scaler.fit_transform(df_X_train), columns = df_X_train.columns)
        df_X_test = pd.DataFrame(scaler.fit_transform(df_X_test), columns = df_X_test.columns)
        clf.fit(df_X_train, df_Y_train)
        accuracy = round(clf.score(df_X_test, df_Y_test) * 100, 5)
        print('Neural Network Accuracy: ' + str(accuracy))
        Y_pred = clf.predict(df_X_test)
        print('Confusion Matrix')
        print(confusion_matrix(df_Y_test, Y_pred))
        model = clf
        return model

    def team_fifa_data(self, x):
        ranking_df = self.ranking_df
        team_ranking = ranking_df[ranking_df['country_full'] == x['Team']]
        team_ranking = team_ranking[team_ranking['rank_date'] == team_ranking['rank_date'].max()]
        if team_ranking.shape[0] == 0 :
            return None
        #print(team_ranking.head())
        row = team_ranking.iloc[0]
        s = pd.Series(data={
            'fifa_rank': 0,
            'fifa_points': 0,
        })
        if row is not None:
            s['fifa_rank'] = int(row['rank'])
            s['fifa_points'] = int(row['total_points'])
        return s

    def predict_group(self, model):
        wc_df = pd.read_csv("./world_cup/Fifa_Worldcup_2022_Groups.csv")
        wc_df['Team'] = wc_df[['Team']].replace({'Korea Republic': 'South Korea', 'Korea DPR': 'North Korea'})
        wc_df['group_points'] = 0 #group stage points
        wc_df['weight_points'] = 0
        wc_df['group_prob'] = wc_df.apply(lambda x: GroupProb(), axis=1)
        fifa_df = wc_df.apply(lambda x: self.team_fifa_data(x), axis=1, result_type='expand')
        wc_df = pd.concat([wc_df, fifa_df], axis=1)
        wc_df.set_index(['Team'], inplace=True, drop=False) #set index
        print(wc_df.head())

        #margin = 0.05
        for group in wc_df['Group'].unique():
            print('___Starting group {}:___'.format(group))

            # group competition
            for home, away in combinations(wc_df.query('Group == "{}"'.format(group))['Team'].values, 2):
                print("{} vs. {}: ".format(home, away), end='')

                # Create a row for each match
                newdata = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, 0, 1]]), columns=['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup'])

                home_rank = wc_df.loc[home, 'fifa_rank']
                home_points = wc_df.loc[home, 'fifa_points']
                away_rank = wc_df.loc[away, 'fifa_rank']
                away_points = wc_df.loc[away, 'fifa_points']

                newdata['rank_diff'] = (home_rank - away_rank)
                newdata['rank_avg'] = (home_rank + away_rank)
                newdata['points_diff'] = (home_points - away_points)

                # Model Output
                win_prob = model.predict_proba(newdata)[0]  #predict
                #print(model.classes_) #print classes order
                #print(win_prob)
                home_win_prob = win_prob[2]
                away_win_prob = win_prob[0]

                wc_df.loc[home, 'group_prob'].probs.append(home_win_prob)
                wc_df.loc[away, 'group_prob'].probs.append(away_win_prob)

                # Determining Win / Draw / Lose based on home_win_prob
                points = 0
                if home_win_prob < away_win_prob:
                    print("{} wins with {:.2f}".format(away, away_win_prob))
                    wc_df.loc[away, 'group_points'] += 3
                    wc_df.loc[away, 'weight_points'] += (away_win_prob) * 3
                elif home_win_prob > away_win_prob:
                    points = 3
                    wc_df.loc[home, 'group_points'] += 3
                    wc_df.loc[home, 'weight_points'] += home_win_prob * 3
                    print("{} wins with {:.2f}".format(home, home_win_prob))
                else:
                    points = 1

                if points == 1:
                    wc_df.loc[home, 'group_points'] += 1
                    wc_df.loc[away, 'group_points'] += 1
                    wc_df.loc[home, 'weight_points'] += home_win_prob * 1
                    wc_df.loc[away, 'weight_points'] += away_win_prob * 1
        wc_df['group_prob_pickle'] = pickle.dumps(wc_df['group_prob'])
        ##insert to mongodb
        mongo = MongoClient()
        worldcup_2022_group = mongo.com6002.worldcup_2022_group
        worldcup_2022_group.drop()
        worldcup_2022_group.insert_many(wc_df.loc[:, wc_df.columns != 'group_prob'].to_dict('records'))
        return wc_df

    def predict_knockout(self, model, wc_df):
        country_total_prob = []
        for country in wc_df['Team']:
            win_prob_list = wc_df.loc[country, 'group_prob'].probs
            #print(win_prob_list)
            total_prob = 0
            prob = 1
            #win three game
            for i in range(3):
                prob = prob * win_prob_list[i]

            total_prob += prob

            #win two game
            for i in range(3):
                prob = 1
                for j in range(3):
                    if i == j:
                        prob = prob * (1 - win_prob_list[i])
                    else:
                        prob = prob * win_prob_list[i]
                        total_prob += prob

            country_total_prob.append((country, total_prob))
        country_total_prob = sorted(country_total_prob, key=lambda x: x[1], reverse=True)
        #print(country_total_prob)
        prob_df = pd.DataFrame(country_total_prob, columns =['Country', 'Probability'])
        prob_df.plot(x="Country", y="Probability", kind="bar", figsize=(20,10))
        plt.show()


matplotlib.use('TkAgg')
p = WorldCUPPredictor()
#p.write_to_mongodb()
model = p.train_model()
wc_df = p.predict_group(model)
p.predict_knockout(model, wc_df)
