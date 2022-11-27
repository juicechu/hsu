import numpy as np
import pandas as pd
import datetime

ranking_df = pd.read_csv('./world_cup/fifa_ranking-2022-10-06.csv')
ranking_df = ranking_df.replace({"IR Iran": "Iran"})
ranking_df['rank_date'] = pd.to_datetime(ranking_df['rank_date'])
print(ranking_df.head())

match_df = pd.read_csv('./world_cup/results.csv')
#match_df =  match_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
match_df['date'] = pd.to_datetime(match_df['date'])
match_df = match_df[match_df['date'] > datetime.datetime(1992,12,31)]
print(match_df.loc[~match_df['home_team'].isin(ranking_df['country_full']), 'home_team'].value_counts())
print(match_df.head())
ranking_df['country_full'] = ranking_df[['country_full']].replace({'Korea Republic': 'South Korea', 'Korea DPR': 'North Korea'})
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
def fifa_rank(x, away=1):
    name = 'away_team' if away == 1 else 'home_team'
    team_ranking = ranking_df[ranking_df['country_full'] == x[name]]
    team_ranking['date_diff'] = x['date'] - team_ranking['rank_date']
    team_ranking = team_ranking[(x['date'] - team_ranking['rank_date']) >= datetime.timedelta()]
    if team_ranking.shape[0] == 0 :
        return 0
    team_ranking = team_ranking.sort_values(by=['date_diff'])
    #print(team_ranking.head())
    return team_ranking.iloc[0]['rank']

match_df['home_team_fifa_rank'] = match_df.apply(lambda x: fifa_rank(x, 0), axis=1)
match_df['away_team_fifa_rank'] = match_df.apply(lambda x: fifa_rank(x, 1), axis=1)
match_df = match_df[match_df['home_team_fifa_rank'] > 0] #filter
match_df = match_df[match_df['away_team_fifa_rank'] > 0] #filter
match_df['rank_dff'] = match_df['home_team_fifa_rank'] - match_df['away_team_fifa_rank']
match_df['rank_average'] = (match_df['home_team_fifa_rank'] + match_df['away_team_fifa_rank']) / 2
match_df['score_diff'] = match_df['home_score'] + match_df['away_score']
match_df['is_win'] = match_df['score_diff'] > 0
match_df['not_friendly'] = match_df['tournament'] != 'Friendly'
match_df['is_worldcup'] = 'World Cup' in match_df['tournament']
print(match_df.tail())

