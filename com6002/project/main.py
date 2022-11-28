import numpy as np
import pandas as pd
import datetime
from pymongo import MongoClient
from itertools import combinations
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
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
                #win
                return 1
            elif x == 0:
                #draw
                return 3
            elif x < 0:
                #lose
                return 2
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
        #df = df.loc[df['is_win'] != 3]
        df['rank_diff'] = df['rank_diff'].astype(int)
        df['rank_avg'] = df['rank_avg'].astype(int)
        df['points_diff'] = df['points_diff'].astype(int)
        df['not_friendly'] = df['not_friendly'].astype(int)
        df['is_worldcup'] = df['is_worldcup'].astype(int)
        df['is_win'] = df['is_win'].astype(str)
        print(df['not_friendly'].value_counts())
        print(df['is_worldcup'].value_counts())
        print(df['is_win'].value_counts())
        print(df.isna().sum())
        # data distribution
        features = ['rank_avg', 'rank_diff', 'points_diff']

        for feature in features:
            plt.figure()
            df[feature].plot.hist(bins=30)
            plt.title(feature)

        df.loc[:,['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup', 'is_win']].info()
        df_X, df_Y = df.loc[:,['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup']], df['is_win']
        df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=12345)

        # Logistic Regression
        logreg = LogisticRegression(random_state=12345, max_iter=1e6)
        features = PolynomialFeatures(degree=2)
        logmodel = Pipeline([
            ('polynomial_features', features),
            ('logistic_regression', logreg)
        ])
        logmodel.fit(df_X_train, df_Y_train)
        accuracy = round(logmodel.score(df_X_test, df_Y_test) * 100, 5) #Return the mean accuracy on the given test data and labels.
        Y_pred = logmodel.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=logmodel.classes_)
        disp.plot()
        disp.ax_.set_title('Logistic Regression Accuracy: ' + str(accuracy))
        #ROC curve
        #fpr, tpr, _ = roc_curve(df_Y_test, logmodel.predict_proba(df_X_test)[:,1])
        #plt.figure(figsize=(15,5))
        #plt.plot([0, 1], [0, 1], 'k--')
        #plt.plot(fpr, tpr)
        #plt.ax_.set_title('AUC score is {0:0.2}'.format(roc_auc_score(df_Y_test, model.predict_proba(df_X_test)[:,1])))
        #plt.ax_.set_aspect(1)

        """
        # Support Vector Machines
        svc = SVC(random_state=12345)
        svc.fit(df_X_train, df_Y_train)
        accuracy = round(svc.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = svc.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=svc.classes_)
        disp.plot()
        disp.ax_.set_title('SVC Accuracy: ' + str(accuracy))

        # KNN
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(df_X_train, df_Y_train)
        accuracy = round(knn.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = knn.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=knn.classes_)
        disp.plot()
        disp.ax_.set_title('KNN Accuracy: ' + str(accuracy))

        # Decision Tree
        dtree = DecisionTreeClassifier(random_state=12345)
        dtree.fit(df_X_train, df_Y_train)
        accuracy = round(dtree.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = dtree.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=dtree.classes_)
        disp.plot()
        disp.ax_.set_title('Decision Tree Accuracy: ' + str(accuracy))


        # Random Forest
        forest = RandomForestClassifier(random_state=12345, n_estimators=100)
        forest.fit(df_X_train, df_Y_train)
        accuracy = round(forest.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = forest.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=forest.classes_)
        disp.plot()
        disp.ax_.set_title('Random Forest Accuracy: ' + str(accuracy))

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(df_X_train, df_Y_train)
        accuracy = round(nb.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = nb.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=nb.classes_)
        disp.plot()
        disp.ax_.set_title('Naive Bayes Accuracy: ' + str(accuracy))
        # Neural Network
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(6, 3), random_state=12345, max_iter=1e6)
        scaler = preprocessing.StandardScaler()
        df_X_train = pd.DataFrame(scaler.fit_transform(df_X_train), columns = df_X_train.columns)
        df_X_test = pd.DataFrame(scaler.fit_transform(df_X_test), columns = df_X_test.columns)
        clf.fit(df_X_train, df_Y_train)
        accuracy = round(clf.score(df_X_test, df_Y_test) * 100, 5)
        Y_pred = clf.predict(df_X_test)
        cm = confusion_matrix(df_Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
        disp.plot()
        disp.ax_.set_title('Neural Network Accuracy: ' + str(accuracy))
        """
        model = logmodel
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

    def predict(self, model):
        wc_df = pd.read_csv("./world_cup/Fifa_Worldcup_2022_Groups.csv")
        wc_df['Team'] = wc_df[['Team']].replace({'Korea Republic': 'South Korea', 'Korea DPR': 'North Korea'})
        wc_df['group_points'] = 0 #group stage points
        wc_df['weight_points'] = 0
        wc_df['group_prob_obj'] = wc_df.apply(lambda x: GroupProb(), axis=1)
        wc_df['group_prob'] = 1
        fifa_df = wc_df.apply(lambda x: self.team_fifa_data(x), axis=1, result_type='expand')
        wc_df = pd.concat([wc_df, fifa_df], axis=1)
        wc_df.set_index(['Team'], inplace=True, drop=False) #set index
        #print(wc_df.head())

        print(model.classes_) #print classes order
        #margin = 0.05
        for group in wc_df['Group'].unique():
            print('___Starting group {}:___'.format(group))

            # group competition
            for home, away in combinations(wc_df.query('Group == "{}"'.format(group))['Team'].values, 2):
                print("{} vs. {}: ".format(home, away), end='')

                # Create a row for each match
                newdata = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, 1, 1]]), columns=['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup'])

                home_rank = wc_df.loc[home, 'fifa_rank']
                home_points = wc_df.loc[home, 'fifa_points']
                away_rank = wc_df.loc[away, 'fifa_rank']
                away_points = wc_df.loc[away, 'fifa_points']

                newdata['rank_diff'] = (home_rank - away_rank)
                newdata['rank_avg'] = (home_rank + away_rank) / 2
                newdata['points_diff'] = (home_points - away_points)

                # Model Output
                win_prob = model.predict_proba(newdata)[0]  #predict
                #print(win_prob)
                home_win_prob = win_prob[0]
                away_win_prob = win_prob[1]
                draw_prob = win_prob[2]

                # Determining Win / Draw / Lose based on home_win_prob
                points = 0
                if draw_prob >= 1/3:
                    points = 1
                elif home_win_prob < away_win_prob:
                    wc_df.loc[away, 'group_points'] += 3
                    wc_df.loc[away, 'weight_points'] += (away_win_prob) * 3
                    away_win_prob += draw_prob
                    wc_df.loc[away, 'group_prob'] *= away_win_prob
                    print("{} wins with {:.6f}".format(away, away_win_prob))
                elif home_win_prob > away_win_prob:
                    points = 3
                    wc_df.loc[home, 'group_points'] += 3
                    wc_df.loc[home, 'weight_points'] += home_win_prob * 3
                    home_win_prob += draw_prob
                    wc_df.loc[home, 'group_prob'] *= home_win_prob
                    print("{} wins with {:.6f}".format(home, home_win_prob))

                if points == 1:
                    wc_df.loc[home, 'group_points'] += 1
                    wc_df.loc[away, 'group_points'] += 1
                    wc_df.loc[home, 'weight_points'] += draw_prob * 1
                    wc_df.loc[away, 'weight_points'] += draw_prob * 1

                wc_df.loc[home, 'group_prob_obj'].probs.append(home_win_prob)
                wc_df.loc[away, 'group_prob_obj'].probs.append(away_win_prob)
        wc_df['group_prob_pickle'] = pickle.dumps(wc_df['group_prob_obj'])
        ##insert to mongodb
        mongo = MongoClient()
        worldcup_2022_group = mongo.com6002.worldcup_2022_group
        worldcup_2022_group.drop()
        worldcup_2022_group.insert_many(wc_df.loc[:, wc_df.columns != 'group_prob_obj'].to_dict('records'))
        return wc_df

    def predict_group(self, model, wc_df):
        wc_df['group_total_prob'] = 0
        for country in wc_df['Team']:
            win_prob_list = wc_df.loc[country, 'group_prob_obj'].probs
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

            wc_df.loc[country, 'group_total_prob'] = total_prob
        prob_df = wc_df.loc[:, ['Team', 'group_points', 'group_total_prob']]
        prob_df = prob_df.sort_values(by=['group_total_prob'], ascending=False)
        ax = prob_df.plot(x="Team", y="group_total_prob", kind="bar", figsize=(20,10), title='Group Stage')
        ax.set_ylabel("Probability")

    def predict_knockout(self, model, wc_df):
        pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14] #A1:B2, B1:A2, C1:D2, D1:C2, E1:F2, F1:E2, G1:H2, H1:G2

        wc_df = wc_df.sort_values(by=['Group', 'group_points', 'group_prob'], ascending=False)
        next_round_wc = wc_df.groupby('Group').nth([0, 1]) # select the top 2
        next_round_wc = next_round_wc.reset_index()
        next_round_wc = next_round_wc.loc[pairing]
        next_round_wc = next_round_wc.set_index('Team')

        finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

        labels = list()
        odds = list()

        for f in finals:
            print("___Starting of the {}___".format(f))
            iterations = int(len(next_round_wc) / 2)
            winners = []

            for i in range(iterations):
                home = next_round_wc.index[i*2]
                away = next_round_wc.index[i*2+1]
                print("{} vs. {}: ".format(home,
                                        away),
                                        end='')
                newdata = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, 1, 1]]), columns=['rank_diff', 'rank_avg', 'points_diff', 'not_friendly', 'is_worldcup'])
                home_rank = wc_df.loc[home, 'fifa_rank']
                home_points = wc_df.loc[home, 'fifa_points']
                away_rank = wc_df.loc[away, 'fifa_rank']
                away_points = wc_df.loc[away, 'fifa_points']
                newdata['rank_diff'] = (home_rank - away_rank)
                newdata['rank_avg'] = (home_rank + away_rank) / 2
                newdata['points_diff'] = (home_points - away_points)

                win_prob = model.predict_proba(newdata)[0]  #predict
                #print(win_prob)
                home_win_prob = win_prob[0]
                away_win_prob = win_prob[1]
                draw_prob = win_prob[2]

                if home_win_prob < away_win_prob:
                    print("{0} wins with probability {1:.6f}".format(away, away_win_prob + draw_prob))
                    winners.append(away)
                else:
                    print("{0} wins with probability {1:.6f}".format(home, home_win_prob + draw_prob))
                    winners.append(home)

                labels.append("{}({:.2f}) vs. {}({:.6f})".format(wc_df.loc[home, 'Team'],
                                                                1/home_win_prob,
                                                                wc_df.loc[away, 'Team'],
                                                                1/away_win_prob))
                odds.append([home_win_prob, away_win_prob])

            next_round_wc = next_round_wc.loc[winners]
            print("\n")


matplotlib.use('Qt5Agg')
p = WorldCUPPredictor()
#p.write_to_mongodb()
model = p.train_model()
wc_df = p.predict(model)
p.predict_group(model, wc_df)
p.predict_knockout(model, wc_df)
plt.show()
