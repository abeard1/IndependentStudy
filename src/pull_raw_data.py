import pandas as pd
import time
from sportsref import euro

def pull_year(year):
    schedules = []

    for league in euro.LEAGUE_IDS:
        season = euro.Season(year, league)
        schedules.append(season.schedule(kind='R'))
    
    schedule_df = pd.concat(schedules)
    schedule_df.reset_index(drop=True)

    # remove duplicates if necessary
    schedule_df.drop_duplicates(subset=['boxscore_id'], inplace=True)


    # Create new df with players vs teams (by pulling boxscores)
    # Note: only using boxscore_id from schedule_df because schedule_df data is unreliable

    player_dfs_list = []

    print('total length {}'.format(len(schedule_df)))

    for index, row in schedule_df.iterrows():
        print('{} {}'.format(index, row['boxscore_id']))
        print(time.time())
        
        # commented out for safety
        #box = euro.BoxScore(row['boxscore_id'])
        
        # home specific
        home_df = box.get_home_stats()
        home_df['team_id'] = box.home()
        home_df['opp_id'] = box.away()
        home_df['team_pts'] = box.home_score()
        home_df['opp_pts'] = box.away_score()
       
        # away specific
        away_df = box.get_away_stats()
        away_df['team_id'] = box.away()
        away_df['opp_id'] = box.home()
        away_df['team_pts'] = box.away_score()
        away_df['opp_pts'] = box.home_score()
        
        both_df = pd.concat([home_df, away_df], ignore_index=True)
        
        # general
        both_df['boxscore_id'] = row['boxscore_id']
        both_df['is_playoffs'] = row['is_playoffs']
        
        player_dfs_list.append(both_df)

    # concatenate
    player_df = pd.concat(player_dfs_list, ignore_index=True)

    # save to csv
    file_path = 'raw_data/raw_data_{}.csv'.format(year)
    player_df.to_csv(file_path, index=False)

years = [2015, 2016, 2017]

for year in years:
    print('year {}'.format(year))
    pull_year(year)

